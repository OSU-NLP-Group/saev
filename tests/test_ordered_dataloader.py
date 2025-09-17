# tests/test_ordered_dataloader.py
import dataclasses
import gc
import json
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


@pytest.fixture(scope="session")
def patch_labeled_shards_path(pytestconfig, tmp_path_factory):
    """Fixture for shards that have a labels.bin file.

    First checks if --shards has labels.bin, otherwise creates test shards with labels.
    """
    shards = pytestconfig.getoption("--shards")
    if shards is not None:
        labels_path = os.path.join(shards, "labels.bin")
        if os.path.exists(labels_path):
            return shards

    # Create test shards with labels if no real shards available
    import numpy as np

    from saev.data import datasets, writers

    tmp_path = tmp_path_factory.mktemp("labeled_shards")

    # Create a dataset with meaningful size
    n_imgs = 20
    n_patches = 16  # 4x4 patches for tiny model

    # Create activation shards
    cfg = writers.Config(
        data=datasets.Fake(n_imgs=n_imgs),
        dump_to=str(tmp_path),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=128,  # Match the tiny model's dimension
        vit_layers=[0],  # Use first layer for speed
        n_patches_per_img=16,  # Smaller for the tiny model
        cls_token=True,
        max_patches_per_shard=100,  # Force multiple shards
        vit_batch_size=4,
        n_workers=0,
        device="cpu",
    )

    # Generate the activation shards
    writers.worker_fn(cfg)

    # Get the actual shard directory
    metadata = writers.Metadata.from_cfg(cfg)
    shard_root = os.path.join(str(tmp_path), metadata.hash)

    # Create realistic labels (simulate semantic segmentation)
    labels_path = os.path.join(shard_root, "labels.bin")
    labels = np.zeros((n_imgs, n_patches), dtype=np.uint8)

    # Simulate different semantic classes with spatial coherence
    np.random.seed(42)  # For reproducibility
    for img_i in range(n_imgs):
        # Create patches of different "classes" for 4x4 grid
        for patch_i in range(n_patches):
            row = patch_i // 4
            col = patch_i % 4

            # Create regions with different labels
            if row < 2:  # Top half
                labels[img_i, patch_i] = 10 + (img_i % 3)  # Vary types
            else:  # Bottom half
                labels[img_i, patch_i] = 20 + (col % 4)  # Different objects

            # Add some "ignore" labels randomly (5% of patches)
            if np.random.random() < 0.05:
                labels[img_i, patch_i] = 255

    labels.tofile(labels_path)

    return shard_root


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
    patches_per_shard = (
        metadata.n_imgs_per_shard * metadata.n_patches_per_img / len(metadata.layers)
    )
    batch_size = int(patches_per_shard * 1.5)  # Should span 2 shards

    cfg = OrderedConfig(
        shard_root=shards_path,
        patches="image",
        layer=layer,
        batch_size=batch_size,
        debug=True,
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
    from saev.data import datasets, writers

    # Create a tiny dataset - just 2 images
    # The tiny-open-clip-model uses 16 patches + 1 CLS token
    n_imgs = 2
    d_vit = 128
    n_patches = 16  # Standard for this model
    layers = [0]

    # Create activation shards using the fake dataset
    cfg = writers.Config(
        data=datasets.Fake(n_imgs=n_imgs),
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


@pytest.mark.slow
def test_missing_shard_file_not_detected_at_init(tmp_path):
    """Test that missing shard files are NOT detected at initialization - exposes the validation gap."""
    from saev.data import datasets, writers

    # Create a small dataset with multiple shards
    n_imgs = 10
    d_vit = 128
    n_patches = 16
    layers = [0]

    # Use small max_patches_per_shard to force multiple shards
    # Each image has 17 tokens (16 patches + 1 CLS), so with 2 images per shard we get 34 patches per shard
    max_patches_per_shard = 34  # This will create ~5 shards for 10 images

    # Create activation shards
    cfg = writers.Config(
        data=datasets.Fake(n_imgs=n_imgs),
        dump_to=str(tmp_path),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=d_vit,
        vit_layers=layers,
        n_patches_per_img=n_patches,
        cls_token=True,
        max_patches_per_shard=max_patches_per_shard,
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
    )

    # Generate the activation shards
    writers.worker_fn(cfg)

    # Get the actual shard directory
    metadata = writers.Metadata.from_cfg(cfg)
    shard_root = os.path.join(str(tmp_path), metadata.hash)

    # Verify we have multiple shards
    shard_files = [f for f in os.listdir(shard_root) if f.endswith(".bin")]
    assert len(shard_files) > 1, f"Expected multiple shards, got {len(shard_files)}"

    # Delete one of the middle shard files (not the first one)
    missing_shard = "acts000001.bin"
    missing_file_path = os.path.join(shard_root, missing_shard)
    assert os.path.exists(missing_file_path), (
        f"Shard file {missing_shard} should exist before deletion"
    )
    os.remove(missing_file_path)
    assert not os.path.exists(missing_file_path), (
        f"Shard file {missing_shard} should be deleted"
    )

    # Verify shards.json still lists the deleted file
    with open(os.path.join(shard_root, "shards.json")) as fd:
        shards_data = json.load(fd)
    shard_names = [s["name"] for s in shards_data]
    assert missing_shard in shard_names, (
        f"shards.json should still list {missing_shard}"
    )

    # Create dataloader. this should raise an error at initialization because missing files should be detected early
    with pytest.raises(FileNotFoundError):
        cfg = OrderedConfig(
            shard_root=shard_root, patches="image", layer=layers[0], drop_last=False
        )
        DataLoader(cfg)


@pytest.mark.slow
def test_patch_labels_returned_when_available(tmp_path):
    """Test that patch labels are returned in batches when labels.bin exists."""
    import numpy as np

    from saev.data import datasets, writers

    # Create a small dataset
    n_imgs = 4
    d_vit = 128
    n_patches = 16
    layers = [0]

    # Create activation shards
    cfg = writers.Config(
        data=datasets.Fake(n_imgs=n_imgs),
        dump_to=str(tmp_path),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=d_vit,
        vit_layers=layers,
        n_patches_per_img=n_patches,
        cls_token=True,
        vit_batch_size=n_imgs,
        n_workers=0,
        device="cpu",
    )

    # Generate the activation shards
    writers.worker_fn(cfg)

    # Get the actual shard directory
    metadata = writers.Metadata.from_cfg(cfg)
    shard_root = os.path.join(str(tmp_path), metadata.hash)

    # Create synthetic labels.bin file
    labels_path = os.path.join(shard_root, "labels.bin")
    # Create distinct labels for each patch position across all images
    # Shape: (n_imgs, n_patches_per_img)
    labels = np.zeros((n_imgs, n_patches), dtype=np.uint8)
    for img_i in range(n_imgs):
        for patch_i in range(n_patches):
            # Create a unique label based on image and patch index
            labels[img_i, patch_i] = (img_i * 10 + patch_i) % 150

    # Save labels to disk
    labels.tofile(labels_path)

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=shard_root,
        patches="image",
        layer=layers[0],
        batch_size=8,
        drop_last=False,
    )
    dl = DataLoader(cfg)

    # Iterate and check that patch labels are returned
    for batch_idx, batch in enumerate(dl):
        assert "patch_labels" in batch, f"Batch {batch_idx} missing patch_labels"

        # Check shape matches other batch elements
        assert batch["patch_labels"].shape[0] == batch["act"].shape[0]
        assert batch["patch_labels"].shape[0] == batch["image_i"].shape[0]
        assert batch["patch_labels"].shape[0] == batch["patch_i"].shape[0]

        # Verify the labels match what we expect
        for i in range(batch["act"].shape[0]):
            img_i = batch["image_i"][i].item()
            patch_i = batch["patch_i"][i].item()
            expected_label = (img_i * 10 + patch_i) % 150
            actual_label = batch["patch_labels"][i].item()
            assert actual_label == expected_label, (
                f"Batch {batch_idx}, item {i}: expected label {expected_label}, "
                f"got {actual_label} for img={img_i}, patch={patch_i}"
            )

        # Test first 3 batches only for speed
        if batch_idx >= 2:
            break


@pytest.mark.slow
def test_patch_labels_not_returned_when_missing(tmp_path):
    """Test that patch labels are NOT returned when labels.bin doesn't exist."""
    from saev.data import datasets, writers

    # Create a small dataset
    n_imgs = 2
    d_vit = 128
    n_patches = 16
    layers = [0]

    # Create activation shards
    cfg = writers.Config(
        data=datasets.Fake(n_imgs=n_imgs),
        dump_to=str(tmp_path),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=d_vit,
        vit_layers=layers,
        n_patches_per_img=n_patches,
        cls_token=True,
        vit_batch_size=n_imgs,
        n_workers=0,
        device="cpu",
    )

    # Generate the activation shards
    writers.worker_fn(cfg)

    # Get the actual shard directory
    metadata = writers.Metadata.from_cfg(cfg)
    shard_root = os.path.join(str(tmp_path), metadata.hash)

    # Ensure labels.bin doesn't exist
    labels_path = os.path.join(shard_root, "labels.bin")
    assert not os.path.exists(labels_path), "labels.bin shouldn't exist for this test"

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=shard_root,
        patches="image",
        layer=layers[0],
        batch_size=8,
        drop_last=False,
    )
    dl = DataLoader(cfg)

    # Iterate and check that patch labels are NOT in the batch
    for batch_idx, batch in enumerate(dl):
        assert "patch_labels" not in batch, (
            f"Batch {batch_idx} should not have patch_labels when labels.bin is missing"
        )

        # Ensure other expected keys are still present
        assert "act" in batch
        assert "image_i" in batch
        assert "patch_i" in batch

        # Test first 2 batches only
        if batch_idx >= 1:
            break


@pytest.mark.slow
def test_no_patch_filtering_occurs(tmp_path):
    """Test that OrderedDataLoader does NOT filter patches based on labels, unlike ShuffledDataLoader."""
    import numpy as np

    from saev.data import datasets, writers

    # Create a small dataset
    n_imgs = 3
    d_vit = 128
    n_patches = 16
    layers = [0]

    # Create activation shards
    cfg = writers.Config(
        data=datasets.Fake(n_imgs=n_imgs),
        dump_to=str(tmp_path),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=d_vit,
        vit_layers=layers,
        n_patches_per_img=n_patches,
        cls_token=True,
        vit_batch_size=n_imgs,
        n_workers=0,
        device="cpu",
    )

    # Generate the activation shards
    writers.worker_fn(cfg)

    # Get the actual shard directory
    metadata = writers.Metadata.from_cfg(cfg)
    shard_root = os.path.join(str(tmp_path), metadata.hash)

    # Create synthetic labels.bin file with some patches marked for "filtering"
    labels_path = os.path.join(shard_root, "labels.bin")
    labels = np.zeros((n_imgs, n_patches), dtype=np.uint8)

    # Mark some patches with label 255 (typically used for "ignore")
    # We'll mark half the patches of each image
    for img_i in range(n_imgs):
        for patch_i in range(n_patches):
            if patch_i % 2 == 0:
                labels[img_i, patch_i] = 255  # "ignore" label
            else:
                labels[img_i, patch_i] = patch_i  # normal label

    labels.tofile(labels_path)

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=shard_root,
        patches="image",
        layer=layers[0],
        batch_size=100,  # Large batch to get all samples
        drop_last=False,
    )
    dl = DataLoader(cfg)

    # Collect all samples
    all_samples = []
    for batch in dl:
        batch_size = batch["act"].shape[0]
        for i in range(batch_size):
            all_samples.append({
                "image_i": batch["image_i"][i].item(),
                "patch_i": batch["patch_i"][i].item(),
                "label": batch["patch_labels"][i].item()
                if "patch_labels" in batch
                else None,
            })

    # Verify we got ALL patches, including those with label 255
    expected_total = n_imgs * n_patches
    assert len(all_samples) == expected_total, (
        f"Expected {expected_total} samples, got {len(all_samples)}. "
        "OrderedDataLoader should NOT filter patches."
    )

    # Verify patches with label 255 are included
    ignore_label_count = sum(1 for s in all_samples if s["label"] == 255)
    expected_ignore_count = n_imgs * (n_patches // 2)  # Half of patches per image
    assert ignore_label_count == expected_ignore_count, (
        f"Expected {expected_ignore_count} patches with label 255, got {ignore_label_count}. "
        "OrderedDataLoader should include all patches regardless of label."
    )


@pytest.mark.slow
def test_patch_labels_consistency_across_batches(tmp_path):
    """Test that patch labels are consistent across multiple iterations."""
    import numpy as np

    from saev.data import datasets, writers

    # Create a small dataset
    n_imgs = 2
    d_vit = 128
    n_patches = 16
    layers = [0]

    # Create activation shards
    cfg = writers.Config(
        data=datasets.Fake(n_imgs=n_imgs),
        dump_to=str(tmp_path),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=d_vit,
        vit_layers=layers,
        n_patches_per_img=n_patches,
        cls_token=True,
        vit_batch_size=n_imgs,
        n_workers=0,
        device="cpu",
    )

    # Generate the activation shards
    writers.worker_fn(cfg)

    # Get the actual shard directory
    metadata = writers.Metadata.from_cfg(cfg)
    shard_root = os.path.join(str(tmp_path), metadata.hash)

    # Create synthetic labels with specific patterns
    labels_path = os.path.join(shard_root, "labels.bin")
    labels = np.zeros((n_imgs, n_patches), dtype=np.uint8)
    for img_i in range(n_imgs):
        for patch_i in range(n_patches):
            labels[img_i, patch_i] = (img_i + patch_i * 2) % 100

    labels.tofile(labels_path)

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=shard_root,
        patches="image",
        layer=layers[0],
        batch_size=4,
        drop_last=False,
    )
    dl = DataLoader(cfg)

    # Collect labels from first iteration
    first_iter_labels = {}
    for batch in dl:
        for i in range(batch["act"].shape[0]):
            img_i = batch["image_i"][i].item()
            patch_i = batch["patch_i"][i].item()
            label = batch["patch_labels"][i].item()
            first_iter_labels[(img_i, patch_i)] = label

    # Collect labels from second iteration
    second_iter_labels = {}
    for batch in dl:
        for i in range(batch["act"].shape[0]):
            img_i = batch["image_i"][i].item()
            patch_i = batch["patch_i"][i].item()
            label = batch["patch_labels"][i].item()
            second_iter_labels[(img_i, patch_i)] = label

    # Verify labels are consistent
    assert first_iter_labels == second_iter_labels, (
        "Patch labels should be consistent across iterations"
    )


@pytest.mark.slow
def test_patch_labels_dtype_and_range(tmp_path):
    """Test that patch labels have correct dtype and value range."""
    import numpy as np

    from saev.data import datasets, writers

    # Create a small dataset
    n_imgs = 2
    d_vit = 128
    n_patches = 16
    layers = [0]

    # Create activation shards
    cfg = writers.Config(
        data=datasets.Fake(n_imgs=n_imgs),
        dump_to=str(tmp_path),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=d_vit,
        vit_layers=layers,
        n_patches_per_img=n_patches,
        cls_token=True,
        vit_batch_size=n_imgs,
        n_workers=0,
        device="cpu",
    )

    # Generate the activation shards
    writers.worker_fn(cfg)

    # Get the actual shard directory
    metadata = writers.Metadata.from_cfg(cfg)
    shard_root = os.path.join(str(tmp_path), metadata.hash)

    # Create labels with full uint8 range
    labels_path = os.path.join(shard_root, "labels.bin")
    labels = np.zeros((n_imgs, n_patches), dtype=np.uint8)
    # Use various values including edge cases
    test_values = [0, 1, 127, 128, 150, 254, 255]
    for img_i in range(n_imgs):
        for patch_i in range(n_patches):
            labels[img_i, patch_i] = test_values[
                (img_i * n_patches + patch_i) % len(test_values)
            ]

    labels.tofile(labels_path)

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=shard_root,
        patches="image",
        layer=layers[0],
        batch_size=8,
        drop_last=False,
    )
    dl = DataLoader(cfg)

    # Check labels in batches
    all_labels = []
    for batch in dl:
        assert "patch_labels" in batch

        # Check dtype
        assert batch["patch_labels"].dtype == torch.long, (
            f"Expected torch.long dtype for patch_labels, got {batch['patch_labels'].dtype}"
        )

        # Collect all labels
        for i in range(batch["act"].shape[0]):
            label = batch["patch_labels"][i].item()
            all_labels.append(label)

            # Check range
            assert 0 <= label <= 255, f"Label {label} out of uint8 range [0, 255]"

    # Verify we saw various label values
    unique_labels = set(all_labels)
    assert len(unique_labels) == len(test_values), (
        f"Expected {len(test_values)} unique labels, got {len(unique_labels)}"
    )


@pytest.mark.slow
def test_patch_labels_with_multiple_shards(tmp_path):
    """Test that patch labels work correctly when data spans multiple shards."""
    import numpy as np

    from saev.data import datasets, writers

    # Create dataset that will span multiple shards
    n_imgs = 6
    d_vit = 128
    n_patches = 16
    layers = [0]

    # Force multiple shards by setting small max_patches_per_shard
    max_patches_per_shard = 34  # ~2 images per shard

    # Create activation shards
    cfg = writers.Config(
        data=datasets.Fake(n_imgs=n_imgs),
        dump_to=str(tmp_path),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=d_vit,
        vit_layers=layers,
        n_patches_per_img=n_patches,
        cls_token=True,
        max_patches_per_shard=max_patches_per_shard,
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
    )

    # Generate the activation shards
    writers.worker_fn(cfg)

    # Get the actual shard directory
    metadata = writers.Metadata.from_cfg(cfg)
    shard_root = os.path.join(str(tmp_path), metadata.hash)

    # Verify we have multiple shards
    shard_files = [
        f for f in os.listdir(shard_root) if f.startswith("acts") and f.endswith(".bin")
    ]
    assert len(shard_files) > 1, f"Expected multiple shards, got {len(shard_files)}"

    # Create synthetic labels
    labels_path = os.path.join(shard_root, "labels.bin")
    labels = np.zeros((n_imgs, n_patches), dtype=np.uint8)
    for img_i in range(n_imgs):
        for patch_i in range(n_patches):
            # Create unique label per position
            labels[img_i, patch_i] = (img_i * 20 + patch_i) % 150

    labels.tofile(labels_path)

    # Create OrderedDataLoader with batch size that spans shards
    cfg = OrderedConfig(
        shard_root=shard_root,
        patches="image",
        layer=layers[0],
        batch_size=40,  # Large enough to span shards
        drop_last=False,
    )
    dl = DataLoader(cfg)

    # Verify labels are correct across shard boundaries
    for batch_idx, batch in enumerate(dl):
        assert "patch_labels" in batch

        for i in range(batch["act"].shape[0]):
            img_i = batch["image_i"][i].item()
            patch_i = batch["patch_i"][i].item()
            expected_label = (img_i * 20 + patch_i) % 150
            actual_label = batch["patch_labels"][i].item()

            assert actual_label == expected_label, (
                f"Batch {batch_idx}, item {i}: label mismatch across shards. "
                f"Expected {expected_label}, got {actual_label} for img={img_i}, patch={patch_i}"
            )

        # Test first 2 batches
        if batch_idx >= 1:
            break


@pytest.mark.slow
def test_real_shards_with_labels(patch_labeled_shards_path):
    """Test OrderedDataLoader with real shards that have labels.bin."""
    import numpy as np

    # Load metadata to get dimensions
    metadata = saev.data.Metadata.load(patch_labeled_shards_path)
    layer = metadata.layers[0]

    # Load labels to understand the data
    labels_path = os.path.join(patch_labeled_shards_path, "labels.bin")
    labels_mmap = np.memmap(
        labels_path,
        mode="r",
        dtype=np.uint8,
        shape=(metadata.n_imgs, metadata.n_patches_per_img),
    )

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=patch_labeled_shards_path,
        patches="image",
        layer=layer,
        batch_size=256,
        drop_last=False,
    )
    dl = DataLoader(cfg)

    # Collect samples from first few batches
    samples_collected = 0
    max_samples = min(1000, dl.n_samples)  # Test first 1000 samples

    for batch_idx, batch in enumerate(dl):
        # Verify patch_labels is present
        assert "patch_labels" in batch, f"Batch {batch_idx} missing patch_labels"

        # Check shapes are consistent
        batch_size = batch["act"].shape[0]
        assert batch["patch_labels"].shape == (batch_size,)
        assert batch["image_i"].shape == (batch_size,)
        assert batch["patch_i"].shape == (batch_size,)

        # Verify labels match what's in the file
        for i in range(batch_size):
            img_i = batch["image_i"][i].item()
            patch_i = batch["patch_i"][i].item()

            # Get expected label from mmap
            expected_label = labels_mmap[img_i, patch_i]
            actual_label = batch["patch_labels"][i].item()

            assert actual_label == expected_label, (
                f"Label mismatch at img={img_i}, patch={patch_i}: "
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
def test_real_shards_label_distribution(patch_labeled_shards_path):
    """Test the distribution and properties of labels in real shards."""
    from collections import Counter

    metadata = saev.data.Metadata.load(patch_labeled_shards_path)
    layer = metadata.layers[0]

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=patch_labeled_shards_path,
        patches="image",
        layer=layer,
        batch_size=512,
        drop_last=False,
    )
    dl = DataLoader(cfg)

    # Collect label statistics
    label_counter = Counter()
    total_samples = 0
    max_samples = min(5000, dl.n_samples)  # Analyze first 5000 samples

    for batch in dl:
        assert "patch_labels" in batch

        # Collect labels
        for i in range(batch["act"].shape[0]):
            label = batch["patch_labels"][i].item()
            label_counter[label] += 1
            total_samples += 1

            # Verify label is in valid range
            assert 0 <= label <= 255, f"Label {label} out of valid uint8 range"

        if total_samples >= max_samples:
            break

    # Analyze distribution
    print(f"\nAnalyzed {total_samples} samples")
    print(f"Found {len(label_counter)} unique labels")

    # Get most common labels
    most_common = label_counter.most_common(10)
    print("\nTop 10 most common labels:")
    for label, count in most_common:
        percentage = (count / total_samples) * 100
        print(f"  Label {label:3d}: {count:5d} samples ({percentage:5.2f}%)")

    # Check if there's an ignore label (often 255 or 0)
    if 255 in label_counter:
        print(f"\nLabel 255 (often 'ignore'): {label_counter[255]} samples")
    if 0 in label_counter:
        print(f"Label 0: {label_counter[0]} samples")

    assert total_samples > 0, "No samples were analyzed"


@pytest.mark.slow
def test_real_shards_sequential_order_with_labels(patch_labeled_shards_path):
    """Test that real shards maintain sequential order while returning labels."""
    metadata = saev.data.Metadata.load(patch_labeled_shards_path)
    layer = metadata.layers[0]

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=patch_labeled_shards_path,
        patches="image",
        layer=layer,
        batch_size=128,
        drop_last=False,
    )
    dl = DataLoader(cfg)

    # Track sequential order
    prev_img_i = -1
    prev_patch_i = -1
    samples_checked = 0
    max_samples = 1000

    for batch_idx, batch in enumerate(dl):
        assert "patch_labels" in batch

        for i in range(batch["act"].shape[0]):
            img_i = batch["image_i"][i].item()
            patch_i = batch["patch_i"][i].item()
            _ = batch["patch_labels"][i].item()  # Verify labels exist

            # Check sequential order
            if prev_img_i >= 0:  # Skip first sample
                if img_i == prev_img_i:
                    # Same image, patch index should increment
                    assert patch_i == prev_patch_i + 1, (
                        f"Patches not sequential within image: "
                        f"prev=({prev_img_i},{prev_patch_i}), curr=({img_i},{patch_i})"
                    )
                else:
                    # Next image
                    assert img_i == prev_img_i + 1, (
                        f"Images not sequential: prev={prev_img_i}, curr={img_i}"
                    )
                    assert patch_i == 0, (
                        f"First patch of new image should be 0, got {patch_i}"
                    )

            prev_img_i = img_i
            prev_patch_i = patch_i
            samples_checked += 1

            if samples_checked >= max_samples:
                break

        if samples_checked >= max_samples:
            break

    print(f"Verified sequential order for {samples_checked} samples with labels")
    assert samples_checked > 0, "No samples were checked"


@pytest.mark.slow
def test_real_shards_no_filtering(patch_labeled_shards_path):
    """Verify that real shards with labels don't filter any patches."""
    import numpy as np

    metadata = saev.data.Metadata.load(patch_labeled_shards_path)
    layer = metadata.layers[0]

    # Load labels to check for potential "ignore" labels
    labels_path = os.path.join(patch_labeled_shards_path, "labels.bin")
    labels_mmap = np.memmap(
        labels_path,
        mode="r",
        dtype=np.uint8,
        shape=(metadata.n_imgs, metadata.n_patches_per_img),
    )

    # Count occurrences of each label in the file
    unique_labels, counts = np.unique(
        labels_mmap[:100], return_counts=True
    )  # Sample first 100 images
    label_counts_file = dict(zip(unique_labels, counts))

    # Create OrderedDataLoader
    cfg = OrderedConfig(
        shard_root=patch_labeled_shards_path,
        patches="image",
        layer=layer,
        batch_size=256,
        drop_last=False,
    )
    dl = DataLoader(cfg)

    # Count labels seen in dataloader for same images
    label_counts_dl = {}
    images_to_check = min(100, metadata.n_imgs)
    expected_samples = images_to_check * metadata.n_patches_per_img
    actual_samples = 0

    for batch in dl:
        assert "patch_labels" in batch

        for i in range(batch["act"].shape[0]):
            img_i = batch["image_i"][i].item()
            if img_i >= images_to_check:
                break

            label = batch["patch_labels"][i].item()
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
    for label, count in label_counts_file.items():
        assert label in label_counts_dl, f"Label {label} missing from dataloader output"
        assert label_counts_dl[label] == count, (
            f"Label {label} count mismatch: file has {count}, dataloader returned {label_counts_dl[label]}"
        )

    print(f"Verified all {actual_samples} samples present with no filtering")
    print(f"Unique labels found: {sorted(label_counts_dl.keys())}")


@pytest.mark.slow
def test_real_shards_reproducibility_with_labels(patch_labeled_shards_path):
    """Test that multiple iterations over real shards produce identical results."""
    metadata = saev.data.Metadata.load(patch_labeled_shards_path)
    layer = metadata.layers[0]

    cfg = OrderedConfig(
        shard_root=patch_labeled_shards_path,
        patches="image",
        layer=layer,
        batch_size=256,
        drop_last=False,
    )
    dl = DataLoader(cfg)

    # Collect data from first iteration
    first_iter_data = []
    samples_to_check = 500

    for batch in dl:
        assert "patch_labels" in batch

        for i in range(batch["act"].shape[0]):
            first_iter_data.append({
                "act": batch["act"][i].clone(),
                "image_i": batch["image_i"][i].item(),
                "patch_i": batch["patch_i"][i].item(),
                "label": batch["patch_labels"][i].item(),
            })

            if len(first_iter_data) >= samples_to_check:
                break

        if len(first_iter_data) >= samples_to_check:
            break

    # Collect data from second iteration
    second_iter_data = []
    for batch in dl:
        assert "patch_labels" in batch

        for i in range(batch["act"].shape[0]):
            second_iter_data.append({
                "act": batch["act"][i].clone(),
                "image_i": batch["image_i"][i].item(),
                "patch_i": batch["patch_i"][i].item(),
                "label": batch["patch_labels"][i].item(),
            })

            if len(second_iter_data) >= samples_to_check:
                break

        if len(second_iter_data) >= samples_to_check:
            break

    # Compare iterations
    assert len(first_iter_data) == len(second_iter_data)

    for idx, (first, second) in enumerate(zip(first_iter_data, second_iter_data)):
        assert first["image_i"] == second["image_i"], f"Sample {idx}: image_i mismatch"
        assert first["patch_i"] == second["patch_i"], f"Sample {idx}: patch_i mismatch"
        assert first["label"] == second["label"], f"Sample {idx}: label mismatch"
        torch.testing.assert_close(
            first["act"],
            second["act"],
            rtol=1e-5,
            atol=1e-6,
            msg=f"Sample {idx}: activation mismatch",
        )

    print(f"Verified {len(first_iter_data)} samples are identical across iterations")
