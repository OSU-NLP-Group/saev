# tests/test_ordered_dataloader.py
import dataclasses
import gc
import os
import time

import psutil
import pytest
import torch
import torch.multiprocessing as mp

import saev.data
from saev.data.indexed import Config as IndexedConfig
from saev.data.indexed import Dataset as IndexedDataset
from saev.data.ordered import Config as OrderedConfig
from saev.data.ordered import DataLoader

mp.set_start_method("spawn", force=True)

N_SAMPLES = 25_000  # quick but representative
BATCH_SIZE = 4_096
N_BATCHES_TO_TEST = 10  # Number of batches to compare


@pytest.fixture(scope="session")
def ordered_cfg(pytestconfig):
    shards = pytestconfig.getoption("--shards")
    if shards is None:
        pytest.skip("--shards not supplied")
    metadata = saev.data.Metadata.load(shards)
    layer = metadata.layers[0]
    cfg = OrderedConfig(shard_root=shards, patches="image", layer=layer)
    return cfg


@pytest.fixture(scope="session")
def shards_path(pytestconfig):
    shards = pytestconfig.getoption("--shards")
    if shards is None:
        pytest.skip("--shards not supplied")
    return shards


@pytest.fixture(scope="session")
def metadata(shards_path):
    return saev.data.Metadata.load(shards_path)


@pytest.fixture(scope="session")
def layer(metadata):
    return metadata.layers[0]


def test_init_smoke(ordered_cfg):
    """Test that we can instantiate the DataLoader."""
    DataLoader(ordered_cfg)


def test_len_smoke(ordered_cfg):
    """Test that we can get the length of the DataLoader."""
    dl = DataLoader(ordered_cfg)
    assert isinstance(len(dl), int)
    assert len(dl) > 0


def test_iter_smoke(ordered_cfg):
    """Test that we can iterate and get one batch."""
    dl = DataLoader(ordered_cfg)
    # simply iterating one element should succeed
    batch = next(iter(dl))
    assert "act" in batch and "image_i" in batch and "patch_i" in batch
    assert batch["act"].ndim == 2  # [batch, d_vit]
    assert batch["image_i"].ndim == 1  # [batch]
    assert batch["patch_i"].ndim == 1  # [batch]


def test_batches(ordered_cfg):
    """Test that we can iterate through multiple batches."""
    dl = DataLoader(ordered_cfg)
    it = iter(dl)
    for _ in range(8):
        batch = next(it)
        assert "act" in batch and "image_i" in batch and "patch_i" in batch


@pytest.mark.parametrize("bs", [8, 32, 128, 512, 2048])
def test_batch_size_matches(ordered_cfg, bs):
    """Test that batches have the correct size."""
    cfg = dataclasses.replace(ordered_cfg, batch_size=bs)
    dl = DataLoader(cfg)
    it = iter(dl)
    for _ in range(4):
        batch = next(it)
        # Last batch might be smaller
        assert batch["act"].shape[0] <= bs
        assert batch["image_i"].shape[0] == batch["act"].shape[0]
        assert batch["patch_i"].shape[0] == batch["act"].shape[0]


def peak_children():
    """Return set(pid) of live child processes."""
    return {p.pid: p.name() for p in psutil.Process().children(recursive=True)}


def test_no_child_leak(ordered_cfg):
    """Loader must clean up its workers after iteration terminates."""
    before = peak_children()

    dl = DataLoader(ordered_cfg)

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


def test_compare_with_indexed_sequential(shards_path, layer):
    """
    Compare ordered dataloader output with indexed dataset. The ordered dataloader should produce data in the exact same order as iterating through the indexed dataset sequentially.
    """
    # Setup ordered dataloader
    ordered_cfg = OrderedConfig(
        shard_root=shards_path,
        patches="image",
        layer=layer,
        batch_size=BATCH_SIZE,
    )
    dl = DataLoader(ordered_cfg)

    # Setup indexed dataset
    indexed_cfg = IndexedConfig(
        shard_root=shards_path,
        patches="image",
        layer=layer,
    )
    ds = IndexedDataset(indexed_cfg)

    # Iterate through ordered dataloader and compare with indexed dataset
    global_idx = 0
    it = iter(dl)

    for batch_idx in range(N_BATCHES_TO_TEST):
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
            batch_img_i = batch["image_i"][i].item()
            batch_patch_i = batch["patch_i"][i].item()

            # Verify metadata matches
            assert indexed_example["image_i"] == batch_img_i, (
                f"Batch {batch_idx}, item {i} (global idx {global_idx}): image_i mismatch: indexed={indexed_example['image_i']}, ordered={batch_img_i}"
            )
            assert indexed_example["patch_i"] == batch_patch_i, (
                f"Batch {batch_idx}, item {i} (global idx {global_idx}): patch_i mismatch: indexed={indexed_example['patch_i']}, ordered={batch_patch_i}"
            )

            # Compare activations with tolerance for float32 precision
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
    dl = DataLoader(ordered_cfg)

    prev_img_i = -1
    prev_patch_i = -1

    it = iter(dl)
    for batch_idx in range(5):  # Check first 5 batches
        batch = next(it)

        for i in range(batch["act"].shape[0]):
            img_i = batch["image_i"][i].item()
            patch_i = batch["patch_i"][i].item()

            # Check sequential order
            if img_i == prev_img_i:
                # Same image, patch should be next
                assert patch_i == prev_patch_i + 1, (
                    f"Batch {batch_idx}, item {i}: patches not sequential: "
                    f"prev=({prev_img_i}, {prev_patch_i}), curr=({img_i}, {patch_i})"
                )
            elif img_i == prev_img_i + 1:
                # Next image, patch should be 0
                assert patch_i == 0, (
                    f"Batch {batch_idx}, item {i}: first patch of new image should be 0: "
                    f"prev=({prev_img_i}, {prev_patch_i}), curr=({img_i}, {patch_i})"
                )
            elif prev_img_i == -1:
                # First iteration
                assert img_i == 0 and patch_i == 0, (
                    f"First sample should be (0, 0), got ({img_i}, {patch_i})"
                )
            else:
                # Should not skip images
                raise AssertionError(
                    f"Batch {batch_idx}, item {i}: images not sequential: "
                    f"prev=({prev_img_i}, {prev_patch_i}), curr=({img_i}, {patch_i})"
                )

            prev_img_i = img_i
            prev_patch_i = patch_i


def test_reproducibility(ordered_cfg):
    """Test that multiple iterations produce the same data in the same order."""
    dl = DataLoader(ordered_cfg)

    # Collect first few batches from first iteration
    first_batches = []
    it1 = iter(dl)
    for _ in range(3):
        batch = next(it1)
        first_batches.append({
            "act": batch["act"].clone(),
            "image_i": batch["image_i"].clone(),
            "patch_i": batch["patch_i"].clone(),
        })

    # Collect same batches from second iteration
    second_batches = []
    it2 = iter(dl)
    for _ in range(3):
        batch = next(it2)
        second_batches.append({
            "act": batch["act"].clone(),
            "image_i": batch["image_i"].clone(),
            "patch_i": batch["patch_i"].clone(),
        })

    # Compare batches
    for i, (b1, b2) in enumerate(zip(first_batches, second_batches)):
        torch.testing.assert_close(b1["act"], b2["act"], rtol=0, atol=0)
        torch.testing.assert_close(b1["image_i"], b2["image_i"], rtol=0, atol=0)
        torch.testing.assert_close(b1["patch_i"], b2["patch_i"], rtol=0, atol=0)


def test_constructor_validation(ordered_cfg):
    """Test that constructor validates inputs properly."""
    # Test with non-existent directory
    cfg = dataclasses.replace(ordered_cfg, shard_root="/nonexistent/path")
    with pytest.raises(RuntimeError, match="Activations are not saved"):
        DataLoader(cfg)


def test_properties(ordered_cfg):
    """Test DataLoader properties."""
    dl = DataLoader(ordered_cfg)

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
    dl = DataLoader(cfg)

    it = iter(dl)
    for _ in range(10):
        batch = next(it)
        assert batch["act"].shape[0] == 1
        assert batch["image_i"].shape[0] == 1
        assert batch["patch_i"].shape[0] == 1


def test_memory_stability(ordered_cfg):
    """Test that the dataloader doesn't leak memory over many iterations."""
    cfg = dataclasses.replace(ordered_cfg, batch_size=100)
    dl = DataLoader(cfg)

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


def test_cross_shard_batches(shards_path, layer, metadata):
    """Test that batches spanning multiple shards work correctly."""
    # Use a batch size likely to span shards
    patches_per_shard = metadata.n_imgs_per_shard * metadata.n_patches_per_img
    batch_size = int(patches_per_shard * 1.5)  # Should span 2 shards

    cfg = OrderedConfig(
        shard_root=shards_path,
        patches="image",
        layer=layer,
        batch_size=batch_size,
    )
    dl = DataLoader(cfg)

    # Just verify we can iterate without errors
    it = iter(dl)
    for _ in range(3):
        batch = next(it)
        assert batch["act"].shape[0] <= batch_size


def test_timeout_handling(ordered_cfg):
    """Test batch timeout handling."""
    # Use very short timeout
    cfg = dataclasses.replace(ordered_cfg, batch_timeout_s=0.001)
    dl = DataLoader(cfg)

    # Should still work, just with warnings
    it = iter(dl)
    batch = next(it)
    assert batch["act"].shape[0] > 0


@pytest.mark.slow
def test_ordered_dataloader_with_tiny_fake_dataset(tmp_path):
    """Test OrderedDataLoader with a very small fake dataset to ensure end behavior works."""
    from saev.data import images, writers

    # Create a tiny dataset - just 2 images
    # The tiny-open-clip-model uses 16 patches + 1 CLS token
    n_imgs = 2
    d_vit = 128
    n_patches = 16  # Standard for this model
    layers = [0]

    # Create activation shards using the fake dataset
    cfg = writers.Config(
        data=images.Fake(n_imgs=n_imgs),
        dump_to=str(tmp_path),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=d_vit,
        vit_layers=layers,
        n_patches_per_img=n_patches,
        cls_token=True,  # This model has CLS token
        max_patches_per_shard=1000,
        vit_batch_size=n_imgs,  # Process all images in one batch
        n_workers=0,
        device="cpu",
    )

    # Generate the activation shards
    writers.worker_fn(cfg)

    # Get the actual shard directory
    metadata = writers.Metadata.from_cfg(cfg)
    shard_root = os.path.join(str(tmp_path), metadata.hash)

    # Test with batch_size = 7 (32 total samples, so batches of 7, 7, 7, 7, 4)
    ordered_cfg = OrderedConfig(
        shard_root=shard_root,
        patches="image",
        layer=layers[0],
        batch_size=7,
        drop_last=False,
    )

    # Check that we can calculate expected values
    dl = DataLoader(ordered_cfg)
    expected_samples = n_imgs * n_patches  # 2 * 16 = 32
    expected_batches = (expected_samples + 6) // 7  # ceil(32/7) = 5

    assert dl.n_samples == expected_samples
    assert dl.n_batches == expected_batches
    assert len(dl) == expected_batches

    for batch in dl:
        assert len(batch["patch_i"]) <= ordered_cfg.batch_size

    # The actual iteration might still fail due to multiprocessing,
    # but at least we've tested the calculation logic
    dl.shutdown()
