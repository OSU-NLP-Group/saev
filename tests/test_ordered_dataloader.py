# tests/test_ordered_dataloader.py
import dataclasses
import gc
import json
import pathlib
import time

import beartype
import numpy as np
import psutil
import pytest
import torch
import torch.multiprocessing as mp

from saev.data import (
    IndexedConfig,
    IndexedDataset,
    Metadata,
    OrderedConfig,
    OrderedDataLoader,
    datasets,
)

mp.set_start_method("spawn", force=True)


@pytest.fixture(scope="session")
def shards_dir(pytestconfig):
    shards = pytestconfig.getoption("--shards")
    if shards is None:
        pytest.skip("--shards not supplied")
    return pathlib.Path(shards)


@pytest.fixture(scope="session")
def ordered_cfg(shards_dir):
    md = Metadata.load(shards_dir)
    return OrderedConfig(
        shards=shards_dir, tokens="content", layer=md.layers[0], batch_size=128
    )


@pytest.fixture(scope="session")
def shards_dir_with_token_labels(shards_dir):
    """Fixture for shards that have a labels.bin file."""
    labels_path = shards_dir / "labels.bin"
    if not labels_path.exists():
        pytest.skip("--shards has no labels.bin")

    return shards_dir


@pytest.fixture(scope="session")
def ordered_cfg_with_token_labels(shards_dir_with_token_labels):
    md = Metadata.load(shards_dir)
    return OrderedConfig(
        shards=shards_dir_with_token_labels,
        tokens="content",
        layer=md.layers[0],
        batch_size=128,
    )


@beartype.beartype
def write_shards(shards_root: pathlib.Path, **kwargs) -> pathlib.Path:
    from saev.data import shards

    default_kwargs = dict(
        data=datasets.FakeImg(n_examples=2),
        family="clip",
        ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_model=128,
        layers=[0],
        content_tokens_per_example=16,
        cls_token=True,  # This model has a [CLS] token
        max_tokens_per_shard=1000,
        batch_size=4,
        n_workers=0,
        device="cpu",
        shards_root=shards_root,
    )
    default_kwargs.update(kwargs)
    return shards.worker_fn(**default_kwargs)


@beartype.beartype
def write_token_labels(shards_dir: pathlib.Path):
    """Create distinct labels for each patch position across all images"""
    md = Metadata.load(shards_dir)

    labels = np.memmap(
        shards_dir / "labels.bin",
        mode="w+",
        dtype=np.uint8,
        shape=(md.n_examples, md.content_tokens_per_example),
    )
    for example_idx in range(md.n_examples):
        for token_idx in range(md.content_tokens_per_example):
            # Create a unique label based on image and patch index
            labels[example_idx, token_idx] = (example_idx * 10 + token_idx) % 150

    labels.flush()


def peak_children():
    """Return set(pid) of live child processes."""
    return {p.pid: p.name() for p in psutil.Process().children(recursive=True)}


def test_init_smoke(ordered_cfg):
    """Test that we can instantiate the OrderedDataLoader."""
    OrderedDataLoader(ordered_cfg)


def test_len_smoke(ordered_cfg):
    """Test that we can get the length of the OrderedDataLoader."""
    dl = OrderedDataLoader(ordered_cfg)
    assert isinstance(len(dl), int)
    assert len(dl) > 0


def test_iter_smoke(ordered_cfg):
    """Test that we can iterate and get one batch."""
    dl = OrderedDataLoader(ordered_cfg)
    # simply iterating one element should succeed
    batch = next(iter(dl))
    assert "act" in batch
    assert "example_idx" in batch
    assert "token_idx" in batch
    assert batch["act"].ndim == 2  # [batch, d_model]
    assert batch["example_idx"].ndim == 1  # [batch]
    assert batch["token_idx"].ndim == 1  # [batch]


def test_batches(ordered_cfg):
    """Test that we can iterate through multiple batches."""
    dl = OrderedDataLoader(ordered_cfg)
    it = iter(dl)
    for _ in range(8):
        batch = next(it)
        assert "act" in batch
        assert "example_idx" in batch
        assert "token_idx" in batch


@pytest.mark.parametrize("bs", [8, 32, 128, 512, 2048])
def test_batch_size_matches(ordered_cfg, bs):
    """Test that batches have the correct size."""
    cfg = dataclasses.replace(ordered_cfg, batch_size=bs)
    dl = OrderedDataLoader(cfg)
    it = iter(dl)
    for _ in range(4):
        batch = next(it)
        # Last batch might be smaller
        assert batch["act"].shape[0] <= bs
        assert batch["example_idx"].shape[0] == batch["act"].shape[0]
        assert batch["token_idx"].shape[0] == batch["act"].shape[0]


@pytest.mark.xfail()
def test_no_child_leak(ordered_cfg):
    """Loader must clean up its workers after iteration terminates."""
    before = peak_children()

    dl = OrderedDataLoader(ordered_cfg)

    for _ in range(2):  # minimal work
        next(iter(dl))

    if hasattr(dl, "shutdown"):
        dl.shutdown()  # explicit close
    del dl
    gc.collect()
    time.sleep(5.0)  # give OS a tick
    gc.collect()

    after = peak_children()
    assert set(after.keys()).issubset(set(before.keys()))  # no new zombies


def test_compare_with_indexed_sequential(ordered_cfg):
    """
    Compare ordered dataloader output with indexed dataset. The ordered dataloader should produce data in the exact same order as iterating through the indexed dataset sequentially.
    """
    # Setup ordered dataloader
    dl = OrderedDataLoader(ordered_cfg)

    # Setup indexed dataset
    indexed_cfg = IndexedConfig(
        shards=ordered_cfg.shards, tokens=ordered_cfg.tokens, layer=ordered_cfg.layer
    )
    ds = IndexedDataset(indexed_cfg)

    # Iterate through ordered dataloader and compare with indexed dataset
    global_idx = 0
    it = iter(dl)

    for batch_idx in range(10):
        try:
            batch = next(it)
        except StopIteration:
            break

        batch_size = batch["act"].shape[0]

        # Compare each activation in the batch with indexed dataset
        for i in range(batch_size):
            if global_idx >= len(ds):
                break

            # Get from indexed dataset
            indexed_example = ds[global_idx]

            # Get from batch
            batch_act = batch["act"][i]
            batch_example_idx = batch["example_idx"][i].item()
            batch_token_idx = batch["token_idx"][i].item()

            # Verify metadata matches
            assert indexed_example["example_idx"] == batch_example_idx
            assert indexed_example["token_idx"] == batch_token_idx

            torch.testing.assert_close(
                indexed_example["act"],
                batch_act,
                rtol=1e-5,
                atol=1e-6,
                msg=f"Batch {batch_idx}, item {i} (global idx {global_idx}): activation mismatch",
            )

            global_idx += 1


def test_sequential_order(ordered_cfg):
    """Test that data comes out in sequential order."""
    dl = OrderedDataLoader(ordered_cfg)

    prev_example_idx = -1
    prev_token_idx = -1

    it = iter(dl)
    for batch_idx in range(5):  # Check first 5 batches
        batch = next(it)

        for i in range(batch["act"].shape[0]):
            example_idx = batch["example_idx"][i].item()
            token_idx = batch["token_idx"][i].item()

            # Check sequential order
            if example_idx == prev_example_idx:
                # Same image, token should be next
                assert token_idx == prev_token_idx + 1
            elif example_idx == prev_example_idx + 1:
                # Next image, token should be 0
                assert token_idx == 0
            elif prev_example_idx == -1:
                # First iteration
                assert example_idx == 0 and token_idx == 0
            else:
                # Should not skip images
                raise AssertionError(
                    f"Batch {batch_idx}, item {i}: images not sequential: prev=({prev_example_idx}, {prev_token_idx}), curr=({example_idx}, {token_idx})"
                )

            prev_example_idx = example_idx
            prev_token_idx = token_idx


def test_reproducibility(ordered_cfg):
    """Test that multiple iterations produce the same data in the same order."""
    dl = OrderedDataLoader(ordered_cfg)

    # Collect first few batches from first iteration
    first_batches = []
    it1 = iter(dl)
    for _ in range(3):
        batch = next(it1)
        first_batches.append({
            "act": batch["act"].clone(),
            "example_idx": batch["example_idx"].clone(),
            "token_idx": batch["token_idx"].clone(),
        })

    # Collect same batches from second iteration
    second_batches = []
    it2 = iter(dl)
    for _ in range(3):
        batch = next(it2)
        second_batches.append({
            "act": batch["act"].clone(),
            "example_idx": batch["example_idx"].clone(),
            "token_idx": batch["token_idx"].clone(),
        })

    # Compare batches
    for i, (b1, b2) in enumerate(zip(first_batches, second_batches)):
        torch.testing.assert_close(b1["act"], b2["act"], rtol=0, atol=0)
        torch.testing.assert_close(b1["example_idx"], b2["example_idx"], rtol=0, atol=0)
        torch.testing.assert_close(b1["token_idx"], b2["token_idx"], rtol=0, atol=0)


def test_constructor_validation(ordered_cfg):
    """Test that constructor validates inputs properly."""
    # Test with non-existent directory
    cfg = dataclasses.replace(ordered_cfg, shard_root=pathlib.Path("/nonexistent/path"))
    with pytest.raises(RuntimeError, match="Activations are not saved"):
        OrderedDataLoader(cfg)


def test_properties(ordered_cfg):
    """Test OrderedDataLoader properties."""
    dl = OrderedDataLoader(ordered_cfg)

    assert dl.n_batches == len(dl)
    assert dl.n_samples > 0
    assert dl.batch_size == ordered_cfg.batch_size
    assert dl.drop_last == ordered_cfg.drop_last

    # Calculate expected number of batches
    if ordered_cfg.drop_last:
        expected_batches = dl.n_samples // dl.batch_size
    else:
        expected_batches = (dl.n_samples + dl.batch_size - 1) // dl.batch_size

    assert dl.n_batches == expected_batches


def test_edge_cases(ordered_cfg):
    """Test edge cases like very small batch sizes."""
    # Test with batch_size = 1
    cfg = dataclasses.replace(ordered_cfg, batch_size=1)
    dl = OrderedDataLoader(cfg)

    it = iter(dl)
    for _ in range(10):
        batch = next(it)
        assert batch["act"].shape[0] == 1
        assert batch["example_idx"].shape[0] == 1
        assert batch["token_idx"].shape[0] == 1


def test_memory_stability(ordered_cfg):
    """Test that the dataloader doesn't leak memory over many iterations."""
    cfg = dataclasses.replace(ordered_cfg, batch_size=100)
    dl = OrderedDataLoader(cfg)

    # Force garbage collection
    gc.collect()

    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Iterate through many batches
    it = iter(dl)
    for _ in range(50):
        try:
            _ = next(it)
        except StopIteration:
            break

    # Force garbage collection
    gc.collect()
    time.sleep(0.1)

    # Check memory didn't grow too much (allow 100MB growth)
    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory
    assert memory_growth < 100 * 1024 * 1024, (
        f"Memory grew by {memory_growth / 1024 / 1024:.2f}MB"
    )


def test_cross_shard_batches(shards_path, layer, md):
    """Test that batches spanning multiple shards work correctly."""
    # Use a batch size likely to span shards
    patches_per_shard = md.examples_per_shard * md.n_patches_per_img / len(md.layers)
    batch_size = int(patches_per_shard * 1.5)  # Should span 2 shards

    cfg = OrderedConfig(
        shard_root=shards_path,
        patches="image",
        layer=layer,
        batch_size=batch_size,
        debug=True,
    )
    dl = OrderedDataLoader(cfg)

    # Just verify we can iterate without errors
    it = iter(dl)
    for _ in range(3):
        batch = next(it)
        assert batch["act"].shape[0] <= batch_size


def test_timeout_handling(ordered_cfg):
    """Test batch timeout handling."""
    # Use very short timeout
    cfg = dataclasses.replace(ordered_cfg, batch_timeout_s=0.001)
    dl = OrderedDataLoader(cfg)

    # Should still work, just with warnings
    it = iter(dl)
    batch = next(it)
    assert batch["act"].shape[0] > 0


@pytest.mark.slow
def test_ordered_dataloader_with_tiny_fake_dataset():
    """Test OrderedDataLoader with a very small fake dataset to ensure end behavior works."""
    with pytest.helpers.tmp_shards_root() as shards_root:
        # Generate the activation shards
        shards_dir = write_shards(shards_root)
        md = Metadata.load(shards_dir)

        # Test with batch_size = 7 (32 total samples, so batches of 7, 7, 7, 7, 4)
        ordered_cfg = OrderedConfig(
            shards=shards_dir, tokens="content", layer=0, batch_size=7, drop_last=False
        )

        # Check that we can calculate expected values
        dl = OrderedDataLoader(ordered_cfg)
        expected_samples = md.n_examples * md.content_tokens_per_example  # 2 * 16 = 32
        expected_batches = (expected_samples + 6) // 7  # ceil(32/7) = 5

        assert dl.n_samples == expected_samples
        assert dl.n_batches == expected_batches
        assert len(dl) == expected_batches

        for batch in dl:
            assert len(batch["token_idx"]) <= ordered_cfg.batch_size

        # The actual iteration might still fail due to multiprocessing,
        # but at least we've tested the calculation logic
        dl.shutdown()


@pytest.mark.slow
def test_missing_shard_file_not_detected_at_init():
    """Test that missing shard files are NOT detected at initialization - exposes the validation gap."""
    # Use small max_patches_per_shard to force multiple shards
    # Each image has 17 tokens (16 patches + 1 CLS), so with 2 images per shard we get 34 patches per shard
    max_patches_per_shard = 34  # This will create ~5 shards for 10 images
    with pytest.helpers.tmp_shards_root() as shards_root:
        # Generate the activation shards
        shards_dir = write_shards(
            shards_root,
            max_patches_per_shard=max_patches_per_shard,
            data=datasets.FakeImg(n_examples=10),
        )

        # Verify we have multiple shards
        shard_files = [f for f in shards_dir.iterdir() if f.suffix == ".bin"]
        assert len(shard_files) > 1, f"Expected multiple shards, got {len(shard_files)}"

        # Delete one of the middle shard files (not the first one)
        missing_shard = "acts000001.bin"
        missing_file_path = shards_dir / missing_shard
        assert missing_file_path.exists(), (
            f"Shard file {missing_shard} should exist before deletion"
        )
        missing_file_path.unlink()
        assert not missing_file_path.exists(), (
            f"Shard file {missing_shard} should be deleted"
        )

        # Verify shards.json still lists the deleted file
        with open(shards_dir / "shards.json") as fd:
            shards_data = json.load(fd)
        shard_names = [s["name"] for s in shards_data]
        assert missing_shard in shard_names, (
            f"shards.json should still list {missing_shard}"
        )

        # Create dataloader. this should raise an error at initialization because missing files should be detected early
        with pytest.raises(FileNotFoundError):
            cfg = OrderedConfig(shards_dir=shards_dir, tokens="content", layer=0)
            OrderedDataLoader(cfg)


@pytest.mark.slow
def test_patch_labels_returned_when_available(tmp_path):
    """Test that patch labels are returned in batches when labels.bin exists."""

    with pytest.helpers.tmp_shards_root() as shards_root:
        # Generate the activation shards
        shards_dir = write_shards(shards_root)
        md = Metadata.load(shards_dir)

        # Create synthetic labels.bin file
        labels_path = shards_dir / "labels.bin"
        # Create distinct labels for each patch position across all images
        # Shape: (n_examples, n_patches_per_img)
        labels = np.zeros(
            (md.n_examples, md.content_tokens_per_example), dtype=np.uint8
        )
        for example_idx in range(md.n_examples):
            for token_idx in range(md.content_tokens_per_example):
                # Create a unique label based on image and patch index
                labels[example_idx, token_idx] = (example_idx * 10 + token_idx) % 150

        # Save labels to disk
        labels.tofile(labels_path)

        # Create OrderedDataLoader
        cfg = OrderedConfig(shards=shards_dir, tokens="content", layer=0, batch_size=8)
        dl = OrderedDataLoader(cfg)

        # Iterate and check that patch labels are returned
        for batch_idx, batch in enumerate(dl):
            assert "token_labels" in batch, f"Batch {batch_idx} missing token_labels"

            # Check shape matches other batch elements
            assert batch["token_labels"].shape[0] == batch["act"].shape[0]
            assert batch["token_labels"].shape[0] == batch["example_idx"].shape[0]
            assert batch["token_labels"].shape[0] == batch["token_idx"].shape[0]

            # Verify the labels match what we expect
            for i in range(batch["act"].shape[0]):
                img_i = batch["example_idx"][i].item()
                token_idx = batch["token_idx"][i].item()
                expected_label = (img_i * 10 + token_idx) % 150
                actual_label = batch["token_labels"][i].item()
                assert actual_label == expected_label, (
                    f"Batch {batch_idx}, item {i}: expected label {expected_label}, "
                    f"got {actual_label} for img={img_i}, patch={token_idx}"
                )

            # Test first 3 batches only for speed
            if batch_idx >= 2:
                break


@pytest.mark.slow
def test_patch_labels_not_returned_when_missing():
    """Test that patch labels are NOT returned when labels.bin doesn't exist."""

    with pytest.helpers.tmp_shards_root() as shards_root:
        # Generate the activation shards
        shards_dir = write_shards(shards_root)

        # Ensure labels.bin doesn't exist
        labels_path = shards_dir / "labels.bin"
        assert not labels_path.exists(), "labels.bin shouldn't exist for this test"

        # Create OrderedDataLoader
        cfg = OrderedConfig(
            shards=shards_dir, tokens="content", layer=0, batch_size=8, drop_last=False
        )
        dl = OrderedDataLoader(cfg)

        # Iterate and check that patch labels are NOT in the batch
        for batch_idx, batch in enumerate(dl):
            assert "token_labels" not in batch, (
                f"Batch {batch_idx} should not have token_labels when labels.bin is missing"
            )

            # Ensure other expected keys are still present
            assert "act" in batch
            assert "example_idx" in batch
            assert "token_idx" in batch

            # Test first 2 batches only
            if batch_idx >= 1:
                break


@pytest.mark.slow
def test_no_patch_filtering_occurs(tmp_path):
    """Test that OrderedDataLoader does NOT filter patches based on labels, unlike ShuffledDataLoader."""

    with pytest.helpers.tmp_shards_root() as shards_root:
        # Generate the activation shards
        shards_dir = write_shards(shards_root)
        write_token_labels(shards_dir)
        md = Metadata.load(shards_dir)

        # Create OrderedDataLoader
        cfg = OrderedConfig(
            shards=shards_dir,
            tokens="content",
            layer=0,
            batch_size=100,  # Large batch to get all samples
            drop_last=False,
        )
        dl = OrderedDataLoader(cfg)

        # Collect all samples
        all_samples = []
        for batch in dl:
            batch_size = batch["act"].shape[0]
            for i in range(batch_size):
                all_samples.append({
                    "example_idx": batch["example_idx"][i].item(),
                    "token_idx": batch["token_idx"][i].item(),
                    "token_label": batch["token_label"][i].item(),
                })

        # Verify we got ALL patches, including those with label 255
        expected_total = md.n_examples * md.content_tokens_per_example
        assert len(all_samples) == expected_total, (
            f"Expected {expected_total} samples, got {len(all_samples)}. "
            "OrderedDataLoader should NOT filter patches."
        )


@pytest.mark.slow
def test_patch_labels_consistency_across_batches(tmp_path):
    """Test that patch labels are consistent across multiple iterations."""

    with pytest.helpers.tmp_shards_root() as shards_root:
        # Generate the activation shards
        shards_dir = write_shards(shards_root)
        write_token_labels(shards_dir)

        # Create OrderedDataLoader
        cfg = OrderedConfig(shards=shards_dir, tokens="content", layer=0, batch_size=4)
        dl = OrderedDataLoader(cfg)

        # Collect labels from first iteration
        first_iter_labels = {}
        for batch in dl:
            for i in range(batch["act"].shape[0]):
                img_i = batch["example_idx"][i].item()
                token_idx = batch["token_idx"][i].item()
                label = batch["token_labels"][i].item()
                first_iter_labels[(img_i, token_idx)] = label

        # Collect labels from second iteration
        second_iter_labels = {}
        for batch in dl:
            for i in range(batch["act"].shape[0]):
                img_i = batch["example_idx"][i].item()
                token_idx = batch["token_idx"][i].item()
                label = batch["token_labels"][i].item()
                second_iter_labels[(img_i, token_idx)] = label

        # Verify labels are consistent
        assert first_iter_labels == second_iter_labels


@pytest.mark.slow
def test_patch_labels_dtype_and_range(tmp_path):
    """Test that patch labels have correct dtype and value range."""

    with pytest.helpers.tmp_shards_root() as shards_root:
        # Generate the activation shards
        shards_dir = write_shards(shards_root)
        md = Metadata.load(shards_dir)

        labels = np.memmap(
            shards_dir / "labels.bin",
            mode="w+",
            dtype=np.uint8,
            shape=(md.n_examples, md.content_tokens_per_example),
        )
        test_values = [0, 1, 127, 128, 150, 254, 255]
        for example_idx in range(md.n_examples):
            for token_idx in range(md.content_tokens_per_example):
                labels[example_idx, token_idx] = test_values[
                    (example_idx * md.content_tokens_per_example + token_idx)
                    % len(test_values)
                ]
        labels.flush()

        cfg = OrderedConfig(shards=shards_dir, tokens="content", layer=0, batch_size=4)
        dl = OrderedDataLoader(cfg)

        # Check labels in batches
        all_labels = []
        for batch in dl:
            assert "token_labels" in batch

            # Check dtype
            assert batch["token_labels"].dtype == torch.long

            # Collect all labels
            for i in range(batch["act"].shape[0]):
                label = batch["token_labels"][i].item()
                all_labels.append(label)

                # Check range
                assert 0 <= label <= 255

        # Verify we saw all label values
        assert set(all_labels) == set(test_values)


# @pytest.mark.slow
# def test_patch_labels_with_multiple_shards(tmp_path):
#     """Test that patch labels work correctly when data spans multiple shards."""

#     with pytest.helpers.tmp_shards_root() as shards_root:
#         # Generate the activation shards
#         shards_dir = write_shards(
#             shards_root, datasets.FakeImg(n_examples=6), max_patches_per_shard=34
#         )
#         write_token_labels(shards_dir)

#         # Verify we have multiple shards
#         shard_files = [
#             f
#             for f in os.listdir(shards_dir)
#             if f.startswith("acts") and f.endswith(".bin")
#         ]
#         assert len(shard_files) > 1, f"Expected multiple shards, got {len(shard_files)}"

#         # Create synthetic labels
#         labels_path = os.path.join(shard_root, "labels.bin")
#         labels = np.zeros((n_examples, n_patches), dtype=np.uint8)
#         for img_i in range(n_examples):
#             for patch_i in range(n_patches):
#                 # Create unique label per position
#                 labels[img_i, patch_i] = (img_i * 20 + patch_i) % 150

#         labels.tofile(labels_path)

#         # Create OrderedDataLoader with batch size that spans shards
#         cfg = OrderedConfig(
#             shard_root=shard_root,
#             patches="image",
#             layer=layers[0],
#             batch_size=40,  # Large enough to span shards
#             drop_last=False,
#         )
#         dl = OrderedDataLoader(cfg)

#         # Verify labels are correct across shard boundaries
#         for batch_idx, batch in enumerate(dl):
#             assert "token_labels" in batch

#             for i in range(batch["act"].shape[0]):
#                 img_i = batch["example_idx"][i].item()
#                 patch_i = batch["patch_i"][i].item()
#                 expected_label = (img_i * 20 + patch_i) % 150
#                 actual_label = batch["token_labels"][i].item()

#                 assert actual_label == expected_label, (
#                     f"Batch {batch_idx}, item {i}: label mismatch across shards. "
#                     f"Expected {expected_label}, got {actual_label} for img={img_i}, patch={patch_i}"
#                 )

#             # Test first 2 batches
#             if batch_idx >= 1:
#                 break


@pytest.mark.slow
def test_real_shards_with_labels(shards_dir_with_token_labels):
    """Test OrderedDataLoader with real shards that have labels.bin."""
    # Load metadata to get dimensions
    md = Metadata.load(shards_dir_with_token_labels)

    # Load labels to understand the data
    labels_mmap = np.memmap(
        shards_dir_with_token_labels / "labels.bin",
        mode="r",
        dtype=np.uint8,
        shape=(md.n_examples, md.content_tokens_per_example),
    )

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=shards_dir_with_token_labels,
        tokens="content",
        layer=md.layers[0],
        batch_size=256,
        drop_last=False,
    )
    dl = OrderedDataLoader(cfg)

    # Collect samples from first few batches
    samples_collected = 0
    max_samples = min(1000, dl.n_samples)  # Test first 1000 samples

    for batch_idx, batch in enumerate(dl):
        assert "token_labels" in batch

        # Check shapes are consistent
        batch_size = batch["act"].shape[0]
        assert batch["token_labels"].shape == (batch_size,)
        assert batch["example_idx"].shape == (batch_size,)
        assert batch["token_idx"].shape == (batch_size,)

        # Verify labels match what's in the file
        for i in range(batch_size):
            example_idx = batch["example_idx"][i].item()
            token_idx = batch["token_idx"][i].item()

            # Get expected label from mmap
            expected_label = labels_mmap[example_idx, token_idx]
            actual_label = batch["token_labels"][i].item()

            assert actual_label == expected_label, (
                f"Label mismatch at img={example_idx}, patch={token_idx}: "
                f"expected {expected_label}, got {actual_label}"
            )

            samples_collected += 1
            if samples_collected >= max_samples:
                break

        if samples_collected >= max_samples:
            break

    # Ensure we tested a reasonable number of samples
    assert samples_collected > 0, "No samples were tested"
    print(f"Verified {samples_collected} samples with correct labels")


@pytest.mark.slow
def test_real_shards_sequential_order_with_labels(patch_labeled_shards_path):
    """Test that real shards maintain sequential order while returning labels."""
    md = Metadata.load(patch_labeled_shards_path)
    layer = md.layers[0]

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=patch_labeled_shards_path,
        patches="image",
        layer=layer,
        batch_size=128,
        drop_last=False,
    )
    dl = OrderedDataLoader(cfg)

    # Track sequential order
    prev_img_i = -1
    prev_patch_i = -1
    samples_checked = 0
    max_samples = 1000

    for batch_idx, batch in enumerate(dl):
        assert "token_labels" in batch

        for i in range(batch["act"].shape[0]):
            img_i = batch["example_idx"][i].item()
            token_idx = batch["token_idx"][i].item()
            _ = batch["token_labels"][i].item()  # Verify labels exist

            # Check sequential order
            if prev_img_i >= 0:  # Skip first sample
                if img_i == prev_img_i:
                    # Same image, patch index should increment
                    assert token_idx == prev_patch_i + 1, (
                        f"Patches not sequential within image: "
                        f"prev=({prev_img_i},{prev_patch_i}), curr=({img_i},{token_idx})"
                    )
                else:
                    # Next image
                    assert img_i == prev_img_i + 1, (
                        f"Images not sequential: prev={prev_img_i}, curr={img_i}"
                    )
                    assert token_idx == 0, (
                        f"First patch of new image should be 0, got {token_idx}"
                    )

            prev_img_i = img_i
            prev_patch_i = token_idx
            samples_checked += 1

            if samples_checked >= max_samples:
                break

        if samples_checked >= max_samples:
            break

    print(f"Verified sequential order for {samples_checked} samples with labels")
    assert samples_checked > 0, "No samples were checked"


@pytest.mark.slow
def test_real_shards_no_filtering(shards_dir_with_token_labels):
    """Verify that real shards with labels don't filter any patches."""

    md = Metadata.load(shards_dir_with_token_labels)

    # Load labels to check for potential "ignore" labels
    labels_mmap = np.memmap(
        shards_dir_with_token_labels / "labels.bin",
        mode="r",
        dtype=np.uint8,
        shape=(md.n_examples, md.content_tokens_per_example),
    )

    # Count occurrences of each label in the first 100 samples.
    unique_labels, counts = np.unique(labels_mmap[:100], return_counts=True)
    label_counts = dict(zip(unique_labels, counts))

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=shards_dir_with_token_labels,
        content="token",
        layer=md.layers[0],
        batch_size=100,
        drop_last=False,
    )
    dl = OrderedDataLoader(cfg)

    # Count labels seen in dataloader for same images
    label_counts_dl = {}
    images_to_check = min(100, md.n_examples)
    expected_samples = images_to_check * md.content_tokens_per_example
    actual_samples = 0

    for batch in dl:
        assert "token_labels" in batch

        for i in range(batch["act"].shape[0]):
            img_i = batch["example_idx"][i].item()
            if img_i >= images_to_check:
                break

            label = batch["token_labels"][i].item()
            label_counts_dl[label] = label_counts_dl.get(label, 0) + 1
            actual_samples += 1

        # Stop if we've seen all images we're checking
        if actual_samples >= expected_samples:
            break

    # Verify we got all samples (no filtering)
    assert actual_samples == expected_samples, (
        f"Expected {expected_samples} samples, got {actual_samples}. "
        "OrderedDataLoader should not filter any patches."
    )

    # Verify label distribution matches
    for label, count in label_counts.items():
        assert label in label_counts_dl, f"Label {label} missing from dataloader output"
        assert label_counts_dl[label] == count, (
            f"Label {label} count mismatch: file has {count}, dataloader returned {label_counts_dl[label]}"
        )

    print(f"Verified all {actual_samples} samples present with no filtering")
    print(f"Unique labels found: {sorted(label_counts_dl.keys())}")


@pytest.mark.slow
def test_real_shards_reproducibility_with_labels(ordered_cfg_with_token_labels):
    """Test that multiple iterations over real shards produce identical results."""

    dl = OrderedDataLoader(ordered_cfg_with_token_labels)

    # Collect data from first iteration
    first_iter_data = []
    samples_to_check = 500

    for batch in dl:
        assert "token_labels" in batch

        for i in range(batch["act"].shape[0]):
            first_iter_data.append({
                "act": batch["act"][i].clone(),
                "example_idx": batch["example_idx"][i].item(),
                "token_idx": batch["token_idx"][i].item(),
                "token_labels": batch["token_labels"][i].item(),
            })

            if len(first_iter_data) >= samples_to_check:
                break

        if len(first_iter_data) >= samples_to_check:
            break

    # Collect data from second iteration
    second_iter_data = []
    for batch in dl:
        assert "token_labels" in batch

        for i in range(batch["act"].shape[0]):
            second_iter_data.append({
                "act": batch["act"][i].clone(),
                "example_idx": batch["example_idx"][i].item(),
                "token_idx": batch["token_idx"][i].item(),
                "token_labels": batch["token_labels"][i].item(),
            })

            if len(second_iter_data) >= samples_to_check:
                break

        if len(second_iter_data) >= samples_to_check:
            break

    # Compare iterations
    assert len(first_iter_data) == len(second_iter_data)

    for idx, (first, second) in enumerate(zip(first_iter_data, second_iter_data)):
        assert first["example_idx"] == second["example_idx"]
        assert first["token_idx"] == second["token_idx"]
        assert first["token_labels"] == second["token_labels"]
        torch.testing.assert_close(
            first["act"],
            second["act"],
            rtol=1e-5,
            atol=1e-6,
            msg=f"Sample {idx}: activation mismatch",
        )

    print(f"Verified {len(first_iter_data)} samples are identical across iterations")
