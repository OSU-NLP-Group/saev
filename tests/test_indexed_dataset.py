# tests/test_indexed_dataset.py
import gc
import json
import os
import pathlib
import time

import numpy as np
import psutil
import pytest
import torch
import torch.multiprocessing as mp

import saev.data
from saev.data.indexed import Config as IndexedConfig
from saev.data.indexed import Dataset as IndexedDataset
from saev.data.shuffled import Config as IterableConfig
from saev.data.shuffled import DataLoader

mp.set_start_method("spawn", force=True)

N_SAMPLES = 25_000  # quick but representative
BATCH_SIZE = 4_096
N_BATCHES_TO_TEST = 5  # Number of batches to compare


@pytest.fixture(scope="session")
def shards_dir(pytestconfig) -> pathlib.Path:
    shards = pytestconfig.getoption("--shards")
    if shards is None:
        pytest.skip("--shards not supplied")
    return pathlib.Path(shards)


@pytest.fixture(scope="session")
def metadata(shards_dir: pathlib.Path):
    return saev.data.Metadata.load(shards_dir)


@pytest.fixture(scope="session")
def layer(metadata):
    return metadata.layers[0]


def test_init_smoke(shards_dir, layer):
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    IndexedDataset(cfg)


def test_len_smoke(shards_dir, layer):
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    ds = IndexedDataset(cfg)
    assert isinstance(len(ds), int)


def test_getitem_smoke(shards_dir, layer):
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    ds = IndexedDataset(cfg)
    # simply accessing one element should succeed
    example = ds[0]
    assert "act" in example and "ex_i" in example and "patch_i" in example


@pytest.mark.parametrize("seed", [17, 42, 123, 999, 2024])
def test_compare_with_iterable(shards_dir, layer, seed):
    """Compare first N_BATCHES_TO_TEST batches from iterable dataloader with indexed dataset."""
    # Setup iterable dataloader
    iterable_cfg = IterableConfig(
        shards=shards_dir,
        patches="image",
        layer=layer,
        batch_size=BATCH_SIZE,
        seed=seed,
    )
    dl = DataLoader(iterable_cfg)

    # Setup indexed dataset
    indexed_cfg = IndexedConfig(
        shards=shards_dir,
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
            "ex_i": batch["ex_i"].clone(),
            "patch_i": batch["patch_i"].clone(),
        })

    # Compare each activation with indexed dataset
    for batch_idx, batch in enumerate(iterable_batches):
        for i in range(batch["act"].shape[0]):
            ex_i = batch["ex_i"][i].item()
            patch_i = batch["patch_i"][i].item()

            # Calculate the global index for this (image, patch) pair
            global_idx = ex_i * ds.metadata.patches_per_ex + patch_i

            # Get the same example from indexed dataset
            indexed_example = ds[global_idx]

            # Compare values
            assert indexed_example["ex_i"] == ex_i, (
                f"Batch {batch_idx}, item {i}: ex_i mismatch: indexed={indexed_example['ex_i']}, iterable={ex_i}"
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


def test_random_access_consistency(shards_dir, layer):
    """Test that repeated access to the same index returns the same data."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
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
        assert example1["ex_i"] == example2["ex_i"] == example3["ex_i"]
        assert example1["patch_i"] == example2["patch_i"] == example3["patch_i"]
        torch.testing.assert_close(example1["act"], example2["act"], rtol=0, atol=0)
        torch.testing.assert_close(example1["act"], example3["act"], rtol=0, atol=0)


def test_boundary_indices(shards_dir, layer):
    """Test accessing first, last, and boundary indices."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Test first index
    example = ds[0]
    assert example["ex_i"] == 0
    assert example["patch_i"] == 0

    # Test last index
    last_idx = len(ds) - 1
    example = ds[last_idx]
    expected_ex_i = last_idx // ds.metadata.patches_per_ex
    expected_patch_i = last_idx % ds.metadata.patches_per_ex
    assert example["ex_i"] == expected_ex_i
    assert example["patch_i"] == expected_patch_i

    # Test out of bounds should raise IndexError
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_cls_token_mode(shards_dir, layer):
    """Test indexed dataset in CLS token mode."""
    cfg = IndexedConfig(shards=shards_dir, patches="cls", layer=layer)
    ds = IndexedDataset(cfg)

    # In CLS mode, length should be number of images
    assert len(ds) == ds.metadata.n_examples

    # Test a few examples
    for i in range(min(10, len(ds))):
        example = ds[i]
        assert example["ex_i"] == i
        assert example["patch_i"] == -1  # CLS tokens have patch_i = -1
        assert example["act"].shape == (ds.d_model,)


def test_memory_usage(shards_dir, layer):
    """Test that indexed dataset doesn't leak memory on repeated access."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
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


def test_negative_indices(shards_dir, layer):
    """Test that negative indices raise IndexError."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Test various negative indices
    negative_indices = [-1, -10, -100, -len(ds), -len(ds) - 1]
    for idx in negative_indices:
        with pytest.raises(IndexError):
            ds[idx]


def test_shard_boundary_access(shards_dir, layer, metadata):
    """Test accessing data at shard boundaries."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Calculate images per shard
    ex_per_shard = (
        metadata.max_patches_per_shard
        // len(metadata.layers)
        // (metadata.patches_per_ex + int(metadata.cls_token))
    )

    # Test around shard boundaries
    for shard_idx in range(min(3, metadata.n_shards - 1)):
        # Last image in current shard
        last_ex_in_shard = (shard_idx + 1) * ex_per_shard - 1
        if last_ex_in_shard < metadata.n_examples:
            # Last patch of last image in shard
            idx = (
                last_ex_in_shard * metadata.patches_per_ex + metadata.patches_per_ex - 1
            )
            if idx < len(ds):
                example = ds[idx]
                assert example["ex_i"] == last_ex_in_shard
                assert example["patch_i"] == metadata.patches_per_ex - 1

        # First image in next shard
        first_ex_in_next_shard = (shard_idx + 1) * ex_per_shard
        if first_ex_in_next_shard < metadata.n_examples:
            idx = first_ex_in_next_shard * metadata.patches_per_ex
            if idx < len(ds):
                example = ds[idx]
                assert example["ex_i"] == first_ex_in_next_shard
                assert example["patch_i"] == 0


def test_different_layers(shards_dir, metadata):
    """Test that different layers return different activations."""
    if len(metadata.layers) < 2:
        pytest.skip("Need at least 2 layers for this test")

    layer1 = metadata.layers[0]
    layer2 = metadata.layers[1]

    cfg1 = IndexedConfig(shards=shards_dir, patches="image", layer=layer1)
    cfg2 = IndexedConfig(shards=shards_dir, patches="image", layer=layer2)

    ds1 = IndexedDataset(cfg1)
    ds2 = IndexedDataset(cfg2)

    # Same index should give same image/patch but different activations
    test_indices = [0, 100, 1000] if len(ds1) > 1000 else [0]

    for idx in test_indices:
        if idx < len(ds1):
            example1 = ds1[idx]
            example2 = ds2[idx]

            assert example1["ex_i"] == example2["ex_i"]
            assert example1["patch_i"] == example2["patch_i"]

            # Activations should be different between layers
            # Allow for small chance they might be similar
            if not torch.allclose(example1["act"], example2["act"], rtol=1e-3):
                break
    else:
        # If all test indices had similar activations, that's suspicious
        pytest.fail("All tested indices had similar activations across layers")


def test_dataset_properties(shards_dir, layer):
    """Test various dataset properties and edge cases."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Test d_model property
    assert ds.d_model == ds.metadata.d_model
    assert ds.d_model > 0

    # Test that transform returns a tensor
    test_array = np.random.randn(ds.d_model).astype(np.float32)
    transformed = ds.transform(test_array)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (ds.d_model,)
    assert transformed.dtype == torch.float32


def test_all_patches_mode_not_implemented(shards_dir, layer):
    """Test that 'all' patches mode raises NotImplementedError."""
    cfg = IndexedConfig(shards=shards_dir, patches="all", layer=layer)
    ds = IndexedDataset(cfg)

    # Should raise assert_never error
    with pytest.raises(Exception):  # Could be AssertionError or other
        ds[0]


def test_all_layers_not_implemented(shards_dir):
    """Test that layer='all' mode raises NotImplementedError."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer="all")
    ds = IndexedDataset(cfg)

    # Should raise assert_never error
    with pytest.raises(Exception):  # Could be AssertionError or other
        ds[0]


def test_sequential_access_pattern(shards_dir, layer):
    """Test sequential access pattern for performance."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Access first 100 indices sequentially
    n_test = min(100, len(ds))
    prev_ex_i = -1

    for i in range(n_test):
        example = ds[i]

        # Verify indices are sequential
        expected_ex_i = i // ds.metadata.patches_per_ex
        expected_patch_i = i % ds.metadata.patches_per_ex

        assert example["ex_i"] == expected_ex_i
        assert example["patch_i"] == expected_patch_i

        # Check image indices are non-decreasing
        assert example["ex_i"] >= prev_ex_i
        prev_ex_i = example["ex_i"]


def test_cls_mode_with_different_seeds(shards_dir, layer):
    """Test CLS mode with different seeds (should give same results)."""
    cfg1 = IndexedConfig(shards=shards_dir, patches="cls", layer=layer, seed=17)
    cfg2 = IndexedConfig(shards=shards_dir, patches="cls", layer=layer, seed=42)

    ds1 = IndexedDataset(cfg1)
    ds2 = IndexedDataset(cfg2)

    # CLS mode should return same data regardless of seed
    for i in range(min(10, len(ds1))):
        example1 = ds1[i]
        example2 = ds2[i]

        assert example1["ex_i"] == example2["ex_i"]
        assert example1["patch_i"] == example2["patch_i"]
        torch.testing.assert_close(example1["act"], example2["act"], rtol=0, atol=0)


def test_invalid_layer(shards_dir):
    """Test that invalid layer raises assertion error."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=99999)

    with pytest.raises(AssertionError):
        IndexedDataset(cfg)


def test_nonexistent_shard_dir():
    """Test that non-existent shard root raises RuntimeError."""
    cfg = IndexedConfig(
        shards=pathlib.Path("/nonexistent/path"), patches="image", layer=0
    )

    with pytest.raises(RuntimeError, match="Activations are not saved"):
        IndexedDataset(cfg)


def test_edge_case_single_image(shards_dir, layer, metadata):
    """Test edge case where we access patches from the very last image."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Access patches from the last image
    last_ex_idx = metadata.n_examples - 1

    for patch_i in range(metadata.patches_per_ex):
        global_idx = last_ex_idx * metadata.patches_per_ex + patch_i
        if global_idx < len(ds):
            example = ds[global_idx]
            assert example["ex_i"] == last_ex_idx
            assert example["patch_i"] == patch_i


def test_patch_indices_within_bounds(shards_dir, layer):
    """Test that all patch indices are within valid bounds."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Sample random indices
    rng = np.random.default_rng(42)
    test_indices = rng.integers(0, len(ds), size=min(100, len(ds)))

    for idx in test_indices:
        example = ds[idx]

        # Patch index should be within bounds
        assert 0 <= example["patch_i"] < ds.metadata.patches_per_ex

        # Image index should be within bounds
        assert 0 <= example["ex_i"] < ds.metadata.n_examples

        # Activation should have correct shape
        assert example["act"].shape == (ds.d_model,)


def test_dtype_consistency(shards_dir, layer):
    """Test that data types are consistent."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    example = ds[0]

    # Check types
    assert isinstance(example["ex_i"], int)
    assert isinstance(example["patch_i"], int)
    assert isinstance(example["act"], torch.Tensor)
    assert example["act"].dtype == torch.float32


def test_memmap_file_access(shards_dir, layer, metadata):
    """Test that memmap files are accessed correctly."""
    cfg = IndexedConfig(shards=shards_dir, patches="image", layer=layer)
    ds = IndexedDataset(cfg)

    # Test first shard file exists
    acts_fpath = os.path.join(shards_dir, "acts000000.bin")
    assert os.path.exists(acts_fpath)

    # Verify we can access data from first and last patch of first image
    example_first = ds[0]
    example_last = ds[metadata.patches_per_ex - 1]

    assert example_first["ex_i"] == 0
    assert example_first["patch_i"] == 0
    assert example_last["ex_i"] == 0
    assert example_last["patch_i"] == metadata.patches_per_ex - 1


@pytest.mark.slow
def test_missing_shard_file_not_detected_at_init(tmp_path):
    """Test that missing shard files are NOT detected at initialization - exposes the validation gap."""
    from saev.data import datasets, shards

    # Create a small dataset with multiple shards
    n_examples = 10
    d_model = 128
    n_patches = 16
    layers = [0]

    # Use small max_patches_per_shard to force multiple shards
    # Each image has 17 tokens (16 patches + 1 CLS), so with 2 images per shard we get 34 patches per shard
    max_patches_per_shard = 34  # This will create ~5 shards for 10 images

    # Create activation shards
    cfg = shards.Config(
        data=datasets.Fake(n_examples=n_examples),
        dump_to=str(tmp_path),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_model=d_model,
        vit_layers=layers,
        patches_per_ex=n_patches,
        cls_token=True,
        max_patches_per_shard=max_patches_per_shard,
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
    )

    # Generate the activation shards
    shards.worker_fn(cfg)

    # Get the actual shard directory
    metadata = shards.Metadata.from_cfg(cfg)
    shard_dir = os.path.join(str(tmp_path), metadata.hash)

    # Verify we have multiple shards
    shard_files = [f for f in os.listdir(shard_dir) if f.endswith(".bin")]
    assert len(shard_files) > 1, f"Expected multiple shards, got {len(shard_files)}"

    # Delete one of the middle shard files (not the first one)
    missing_shard = "acts000001.bin"
    missing_file_path = os.path.join(shard_dir, missing_shard)
    assert os.path.exists(missing_file_path), (
        f"Shard file {missing_shard} should exist before deletion"
    )
    os.remove(missing_file_path)
    assert not os.path.exists(missing_file_path), (
        f"Shard file {missing_shard} should be deleted"
    )

    # Verify shards.json still lists the deleted file
    with open(os.path.join(shard_dir, "shards.json")) as fd:
        shards_data = json.load(fd)
    shard_names = [s["name"] for s in shards_data]
    assert missing_shard in shard_names, (
        f"shards.json should still list {missing_shard}"
    )

    # Create indexed dataset. This should raise an error at initialization because missing files should be detected early
    with pytest.raises(FileNotFoundError):
        cfg = IndexedConfig(shards=shard_dir, patches="image", layer=layers[0])
        IndexedDataset(cfg)
