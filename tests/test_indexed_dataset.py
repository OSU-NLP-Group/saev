# tests/test_indexed_dataset.py
import gc
import time

import numpy as np
import psutil
import pytest
import torch
import torch.multiprocessing as mp

import saev.data
from saev.data.indexed import Config as IndexedConfig
from saev.data.indexed import Dataset as IndexedDataset
from saev.data.iterable import Config as IterableConfig
from saev.data.iterable import DataLoader

mp.set_start_method("spawn", force=True)

N_SAMPLES = 25_000  # quick but representative
BATCH_SIZE = 4_096
N_BATCHES_TO_TEST = 5  # Number of batches to compare


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


def test_init_smoke(shards_path, layer):
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    IndexedDataset(cfg)


def test_len_smoke(shards_path, layer):
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)
    assert isinstance(len(ds), int)


def test_getitem_smoke(shards_path, layer):
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)
    # simply accessing one element should succeed
    example = ds[0]
    assert "act" in example and "image_i" in example and "patch_i" in example


@pytest.mark.parametrize("seed", [17, 42, 123, 999, 2024])
def test_compare_with_iterable(shards_path, layer, seed):
    """Compare first N_BATCHES_TO_TEST batches from iterable dataloader with indexed dataset."""
    # Setup iterable dataloader
    iterable_cfg = IterableConfig(
        shard_root=shards_path,
        patches="image",
        layer=layer,
        batch_size=BATCH_SIZE,
        seed=seed,
    )
    dl = DataLoader(iterable_cfg)

    # Setup indexed dataset
    indexed_cfg = IndexedConfig(
        shard_root=shards_path,
        patches="image",
        layer=layer,
        seed=seed,
    )
    ds = IndexedDataset(indexed_cfg)

    # Collect batches from iterable dataloader
    iterable_batches = []
    it = iter(dl)
    for _ in range(N_BATCHES_TO_TEST):
        batch = next(it)
        iterable_batches.append({
            "act": batch["act"].clone(),
            "image_i": batch["image_i"].clone(),
            "patch_i": batch["patch_i"].clone(),
        })

    # Compare each activation with indexed dataset
    for batch_idx, batch in enumerate(iterable_batches):
        for i in range(batch["act"].shape[0]):
            img_i = batch["image_i"][i].item()
            patch_i = batch["patch_i"][i].item()

            # Calculate the global index for this (image, patch) pair
            global_idx = img_i * ds.metadata.n_patches_per_img + patch_i

            # Get the same example from indexed dataset
            indexed_example = ds[global_idx]

            # Compare values
            assert indexed_example["image_i"] == img_i, (
                f"Batch {batch_idx}, item {i}: image_i mismatch: indexed={indexed_example['image_i']}, iterable={img_i}"
            )
            assert indexed_example["patch_i"] == patch_i, (
                f"Batch {batch_idx}, item {i}: patch_i mismatch: indexed={indexed_example['patch_i']}, iterable={patch_i}"
            )

            # Compare activations with tolerance for float32 precision
            torch.testing.assert_close(
                indexed_example["act"],
                batch["act"][i],
                rtol=1e-5,
                atol=1e-6,
                msg=f"Batch {batch_idx}, item {i}: activation mismatch",
            )


def test_random_access_consistency(shards_path, layer):
    """Test that repeated access to the same index returns the same data."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Test several random indices
    rng = np.random.default_rng(42)
    test_indices = rng.integers(0, len(ds), size=20)

    for idx in test_indices:
        # Get the same example multiple times
        example1 = ds[idx]
        example2 = ds[idx]
        example3 = ds[idx]

        # All should be identical
        assert example1["image_i"] == example2["image_i"] == example3["image_i"]
        assert example1["patch_i"] == example2["patch_i"] == example3["patch_i"]
        torch.testing.assert_close(example1["act"], example2["act"], rtol=0, atol=0)
        torch.testing.assert_close(example1["act"], example3["act"], rtol=0, atol=0)


def test_boundary_indices(shards_path, layer):
    """Test accessing first, last, and boundary indices."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Test first index
    example = ds[0]
    assert example["image_i"] == 0
    assert example["patch_i"] == 0

    # Test last index
    last_idx = len(ds) - 1
    example = ds[last_idx]
    expected_img_i = last_idx // ds.metadata.n_patches_per_img
    expected_patch_i = last_idx % ds.metadata.n_patches_per_img
    assert example["image_i"] == expected_img_i
    assert example["patch_i"] == expected_patch_i

    # Test out of bounds should raise IndexError
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_cls_token_mode(shards_path, layer):
    """Test indexed dataset in CLS token mode."""
    cfg = IndexedConfig(shard_root=shards_path, patches="cls", layer=layer)
    ds = IndexedDataset(cfg)

    # In CLS mode, length should be number of images
    assert len(ds) == ds.metadata.n_imgs

    # Test a few examples
    for i in range(min(10, len(ds))):
        example = ds[i]
        assert example["image_i"] == i
        assert example["patch_i"] == -1  # CLS tokens have patch_i = -1
        assert example["act"].shape == (ds.d_vit,)


def test_memory_usage(shards_path, layer):
    """Test that indexed dataset doesn't leak memory on repeated access."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Force garbage collection
    gc.collect()

    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Access many random indices
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(ds), size=1000)

    for idx in indices:
        _ = ds[idx]

    # Force garbage collection again
    gc.collect()
    time.sleep(0.1)

    # Check memory didn't grow too much (allow 50MB growth)
    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory
    assert memory_growth < 50 * 1024 * 1024, (
        f"Memory grew by {memory_growth / 1024 / 1024:.2f}MB"
    )
