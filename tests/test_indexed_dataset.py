# tests/test_indexed_dataset.py
import gc
import os
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


def test_negative_indices(shards_path, layer):
    """Test that negative indices raise IndexError."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Test various negative indices
    negative_indices = [-1, -10, -100, -len(ds), -len(ds) - 1]
    for idx in negative_indices:
        with pytest.raises(IndexError):
            ds[idx]


def test_shard_boundary_access(shards_path, layer, metadata):
    """Test accessing data at shard boundaries."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Calculate images per shard
    n_imgs_per_shard = (
        metadata.max_patches_per_shard
        // len(metadata.layers)
        // (metadata.n_patches_per_img + int(metadata.cls_token))
    )

    # Test around shard boundaries
    for shard_idx in range(min(3, metadata.n_shards - 1)):
        # Last image in current shard
        last_img_in_shard = (shard_idx + 1) * n_imgs_per_shard - 1
        if last_img_in_shard < metadata.n_imgs:
            # Last patch of last image in shard
            idx = (
                last_img_in_shard * metadata.n_patches_per_img
                + metadata.n_patches_per_img
                - 1
            )
            if idx < len(ds):
                example = ds[idx]
                assert example["image_i"] == last_img_in_shard
                assert example["patch_i"] == metadata.n_patches_per_img - 1

        # First image in next shard
        first_img_in_next_shard = (shard_idx + 1) * n_imgs_per_shard
        if first_img_in_next_shard < metadata.n_imgs:
            idx = first_img_in_next_shard * metadata.n_patches_per_img
            if idx < len(ds):
                example = ds[idx]
                assert example["image_i"] == first_img_in_next_shard
                assert example["patch_i"] == 0


def test_different_layers(shards_path, metadata):
    """Test that different layers return different activations."""
    if len(metadata.layers) < 2:
        pytest.skip("Need at least 2 layers for this test")

    layer1 = metadata.layers[0]
    layer2 = metadata.layers[1]

    cfg1 = IndexedConfig(shard_root=shards_path, patches="image", layer=layer1)
    cfg2 = IndexedConfig(shard_root=shards_path, patches="image", layer=layer2)

    ds1 = IndexedDataset(cfg1)
    ds2 = IndexedDataset(cfg2)

    # Same index should give same image/patch but different activations
    test_indices = [0, 100, 1000] if len(ds1) > 1000 else [0]

    for idx in test_indices:
        if idx < len(ds1):
            example1 = ds1[idx]
            example2 = ds2[idx]

            assert example1["image_i"] == example2["image_i"]
            assert example1["patch_i"] == example2["patch_i"]

            # Activations should be different between layers
            # Allow for small chance they might be similar
            if not torch.allclose(example1["act"], example2["act"], rtol=1e-3):
                break
    else:
        # If all test indices had similar activations, that's suspicious
        pytest.fail("All tested indices had similar activations across layers")


def test_dataset_properties(shards_path, layer):
    """Test various dataset properties and edge cases."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Test d_vit property
    assert ds.d_vit == ds.metadata.d_vit
    assert ds.d_vit > 0

    # Test that transform returns a tensor
    test_array = np.random.randn(ds.d_vit).astype(np.float32)
    transformed = ds.transform(test_array)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (ds.d_vit,)
    assert transformed.dtype == torch.float32


def test_all_patches_mode_not_implemented(shards_path, layer):
    """Test that 'all' patches mode raises NotImplementedError."""
    cfg = IndexedConfig(shard_root=shards_path, patches="all", layer=layer)
    ds = IndexedDataset(cfg)

    # Should raise assert_never error
    with pytest.raises(Exception):  # Could be AssertionError or other
        ds[0]


def test_all_layers_not_implemented(shards_path):
    """Test that layer='all' mode raises NotImplementedError."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer="all")
    ds = IndexedDataset(cfg)

    # Should raise assert_never error
    with pytest.raises(Exception):  # Could be AssertionError or other
        ds[0]


def test_sequential_access_pattern(shards_path, layer):
    """Test sequential access pattern for performance."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Access first 100 indices sequentially
    n_test = min(100, len(ds))
    prev_img_i = -1

    for i in range(n_test):
        example = ds[i]

        # Verify indices are sequential
        expected_img_i = i // ds.metadata.n_patches_per_img
        expected_patch_i = i % ds.metadata.n_patches_per_img

        assert example["image_i"] == expected_img_i
        assert example["patch_i"] == expected_patch_i

        # Check image indices are non-decreasing
        assert example["image_i"] >= prev_img_i
        prev_img_i = example["image_i"]


def test_cls_mode_with_different_seeds(shards_path, layer):
    """Test CLS mode with different seeds (should give same results)."""
    cfg1 = IndexedConfig(shard_root=shards_path, patches="cls", layer=layer, seed=17)
    cfg2 = IndexedConfig(shard_root=shards_path, patches="cls", layer=layer, seed=42)

    ds1 = IndexedDataset(cfg1)
    ds2 = IndexedDataset(cfg2)

    # CLS mode should return same data regardless of seed
    for i in range(min(10, len(ds1))):
        example1 = ds1[i]
        example2 = ds2[i]

        assert example1["image_i"] == example2["image_i"]
        assert example1["patch_i"] == example2["patch_i"]
        torch.testing.assert_close(example1["act"], example2["act"], rtol=0, atol=0)


def test_invalid_layer(shards_path):
    """Test that invalid layer raises assertion error."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=99999)

    with pytest.raises(AssertionError):
        IndexedDataset(cfg)


def test_nonexistent_shard_root():
    """Test that non-existent shard root raises RuntimeError."""
    cfg = IndexedConfig(shard_root="/nonexistent/path", patches="image", layer=0)

    with pytest.raises(RuntimeError, match="Activations are not saved"):
        IndexedDataset(cfg)


def test_edge_case_single_image(shards_path, layer, metadata):
    """Test edge case where we access patches from the very last image."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Access patches from the last image
    last_img_idx = metadata.n_imgs - 1

    for patch_i in range(metadata.n_patches_per_img):
        global_idx = last_img_idx * metadata.n_patches_per_img + patch_i
        if global_idx < len(ds):
            example = ds[global_idx]
            assert example["image_i"] == last_img_idx
            assert example["patch_i"] == patch_i


def test_patch_indices_within_bounds(shards_path, layer):
    """Test that all patch indices are within valid bounds."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Sample random indices
    rng = np.random.default_rng(42)
    test_indices = rng.integers(0, len(ds), size=min(100, len(ds)))

    for idx in test_indices:
        example = ds[idx]

        # Patch index should be within bounds
        assert 0 <= example["patch_i"] < ds.metadata.n_patches_per_img

        # Image index should be within bounds
        assert 0 <= example["image_i"] < ds.metadata.n_imgs

        # Activation should have correct shape
        assert example["act"].shape == (ds.d_vit,)


def test_dtype_consistency(shards_path, layer):
    """Test that data types are consistent."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    example = ds[0]

    # Check types
    assert isinstance(example["image_i"], int)
    assert isinstance(example["patch_i"], int)
    assert isinstance(example["act"], torch.Tensor)
    assert example["act"].dtype == torch.float32


def test_memmap_file_access(shards_path, layer, metadata):
    """Test that memmap files are accessed correctly."""
    cfg = IndexedConfig(shard_root=shards_path, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Calculate expected shape
    n_imgs_per_shard = (
        metadata.max_patches_per_shard
        // len(metadata.layers)
        // (metadata.n_patches_per_img + int(metadata.cls_token))
    )

    # Test first shard file exists
    acts_fpath = os.path.join(shards_path, "acts000000.bin")
    assert os.path.exists(acts_fpath)

    # Verify we can access data from first and last patch of first image
    example_first = ds[0]
    example_last = ds[metadata.n_patches_per_img - 1]

    assert example_first["image_i"] == 0
    assert example_first["patch_i"] == 0
    assert example_last["image_i"] == 0
    assert example_last["patch_i"] == metadata.n_patches_per_img - 1
