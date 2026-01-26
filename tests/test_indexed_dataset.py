# tests/test_indexed_dataset.py
import dataclasses
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

from saev.data import (
    IndexedConfig,
    IndexedDataset,
    Metadata,
    ShuffledConfig,
    ShuffledDataLoader,
    datasets,
)

mp.set_start_method("spawn", force=True)


@pytest.fixture()
def md(shards_dir) -> IndexedConfig:
    return Metadata.load(shards_dir)


@pytest.fixture()
def cfg(shards_dir, md) -> IndexedConfig:
    return IndexedConfig(shards=shards_dir, tokens="content", layer=md.layers[0])


def test_init_smoke(cfg):
    IndexedDataset(cfg)


def test_len_smoke(cfg):
    ds = IndexedDataset(cfg)
    assert isinstance(len(ds), int)


def test_getitem_smoke(cfg):
    # simply accessing one element should succeed
    ds = IndexedDataset(cfg)
    example = ds[0]
    assert "act" in example
    assert "example_idx" in example
    assert "token_idx" in example


@pytest.mark.parametrize("seed", [17, 42, 123])
def test_compare_with_shuffled(shards_dir, seed):
    """Compare first 4 batches from shuffled dataloader with indexed dataset."""
    md = Metadata.load(shards_dir)
    layer = md.layers[0]
    # Setup shuffled dataloader
    shuffled_cfg = ShuffledConfig(
        shards=shards_dir,
        tokens="content",
        layer=layer,
        batch_size=128,
        seed=seed,
    )
    dl = ShuffledDataLoader(shuffled_cfg)

    # Setup indexed dataset
    indexed_cfg = IndexedConfig(shards=shards_dir, tokens="content", layer=layer)
    ds = IndexedDataset(indexed_cfg)

    # Collect batches from shuffled dataloader
    shuffled_batches = []
    it = iter(dl)
    for _ in range(4):
        batch = next(it)
        shuffled_batches.append({
            "act": batch["act"].clone(),
            "example_idx": batch["example_idx"].clone(),
            "token_idx": batch["token_idx"].clone(),
        })

    # Compare each activation with indexed dataset
    for batch_idx, batch in enumerate(shuffled_batches):
        for i in range(batch["act"].shape[0]):
            example_idx = batch["example_idx"][i].item()
            token_idx = batch["token_idx"][i].item()

            # Calculate the global index for this (image, patch) pair
            global_idx = example_idx * md.content_tokens_per_example + token_idx

            # Get the same example from indexed dataset
            indexed_example = ds[global_idx]

            # Compare values
            assert indexed_example["example_idx"] == example_idx
            assert indexed_example["token_idx"] == token_idx

            # Compare activations with tolerance for float32 precision
            torch.testing.assert_close(
                indexed_example["act"],
                batch["act"][i],
                rtol=1e-5,
                atol=1e-6,
                msg=f"Batch {batch_idx}, item {i}: activation mismatch",
            )


def test_random_access_consistency(cfg):
    """Test that repeated access to the same index returns the same data."""
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
        assert (
            example1["example_idx"]
            == example2["example_idx"]
            == example3["example_idx"]
        )
        assert example1["token_idx"] == example2["token_idx"] == example3["token_idx"]
        torch.testing.assert_close(example1["act"], example2["act"], rtol=0, atol=0)
        torch.testing.assert_close(example1["act"], example3["act"], rtol=0, atol=0)


def test_boundary_indices(cfg, md):
    """Test accessing first, last, and boundary indices."""
    ds = IndexedDataset(cfg)

    # Test first index
    example = ds[0]
    assert example["example_idx"] == 0
    assert example["token_idx"] == 0

    # Test last index
    last_idx = len(ds) - 1
    example = ds[last_idx]
    expected_ex_i = last_idx // md.content_tokens_per_example
    expected_token_i = last_idx % md.content_tokens_per_example
    assert example["example_idx"] == expected_ex_i
    assert example["token_idx"] == expected_token_i

    # Test out of bounds should raise IndexError
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_cls_token_mode(cfg, md):
    """Test indexed dataset in CLS token mode."""
    cfg = dataclasses.replace(cfg, tokens="special")
    ds = IndexedDataset(cfg)

    # In CLS mode, length should be number of images
    assert len(ds) == md.n_examples

    # Test a few examples
    for i in range(min(10, len(ds))):
        example = ds[i]
        assert example["example_idx"] == i
        assert example["token_idx"] == -1  # CLS tokens have token_idx = -1
        assert example["act"].shape == (ds.d_model,)


def test_memory_usage(cfg):
    """Test that indexed dataset doesn't leak memory on repeated access."""
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


def test_negative_indices(cfg):
    """Test that negative indices raise IndexError."""
    ds = IndexedDataset(cfg)

    # Test various negative indices
    negative_indices = [-1, -10, -100, -len(ds), -len(ds) - 1]
    for idx in negative_indices:
        with pytest.raises(IndexError):
            ds[idx]


def test_shard_boundary_access(cfg, md):
    """Test accessing data at shard boundaries."""
    ds = IndexedDataset(cfg)

    # Test around shard boundaries
    for shard_idx in range(min(3, md.n_shards - 1)):
        # Last image in current shard
        last_ex_in_shard = (shard_idx + 1) * md.examples_per_shard - 1
        if last_ex_in_shard < md.n_examples:
            # Last patch of last image in shard
            i = (
                last_ex_in_shard * md.content_tokens_per_example
                + md.content_tokens_per_example
                - 1
            )
            if i < len(ds):
                example = ds[i]
                assert example["example_idx"] == last_ex_in_shard
                assert example["token_idx"] == md.content_tokens_per_example - 1

        # First image in next shard
        first_ex_in_next_shard = (shard_idx + 1) * md.examples_per_shard
        if first_ex_in_next_shard < md.n_examples:
            i = first_ex_in_next_shard * md.content_tokens_per_example
            if i < len(ds):
                example = ds[i]
                assert example["example_idx"] == first_ex_in_next_shard
                assert example["token_idx"] == 0


def test_different_layers(shards_dir, md):
    """Test that different layers return different activations."""
    if len(md.layers) < 2:
        pytest.skip("Need at least 2 layers for this test")

    layer1 = md.layers[0]
    layer2 = md.layers[1]

    cfg1 = IndexedConfig(shards=shards_dir, tokens="content", layer=layer1)
    cfg2 = IndexedConfig(shards=shards_dir, tokens="content", layer=layer2)

    ds1 = IndexedDataset(cfg1)
    ds2 = IndexedDataset(cfg2)

    # Same index should give same image/patch but different activations
    test_indices = [0, 100, 1000] if len(ds1) > 1000 else [0]

    for idx in test_indices:
        if idx < len(ds1):
            example1 = ds1[idx]
            example2 = ds2[idx]

            assert example1["example_idx"] == example2["example_idx"]
            assert example1["token_idx"] == example2["token_idx"]

            # Activations should be different between layers
            # Allow for small chance they might be similar
            if not torch.allclose(example1["act"], example2["act"], rtol=1e-3):
                break
    else:
        # If all test indices had similar activations, that's suspicious
        pytest.fail("All tested indices had similar activations across layers")


def test_dataset_properties(cfg, md):
    """Test various dataset properties and edge cases."""
    ds = IndexedDataset(cfg)

    # Test d_model property
    assert ds.d_model == md.d_model
    assert ds.d_model > 0

    # Test that transform returns a tensor
    test_array = np.random.randn(ds.d_model).astype(np.float32)
    transformed = ds.transform(test_array)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (ds.d_model,)
    assert transformed.dtype == torch.float32


def test_all_patches_mode_not_implemented(cfg):
    """Test that 'all' tokens mode raises NotImplementedError."""
    cfg = dataclasses.replace(cfg, tokens="all")
    ds = IndexedDataset(cfg)

    # Should raise assert_never error
    with pytest.raises(Exception):  # Could be AssertionError or other
        ds[0]


def test_all_layers_not_implemented(shards_dir):
    """Test that layer='all' mode raises NotImplementedError."""
    cfg = IndexedConfig(shards=shards_dir, tokens="content", layer="all")
    ds = IndexedDataset(cfg)

    # Should raise assert_never error
    with pytest.raises(Exception):  # Could be AssertionError or other
        ds[0]


def test_sequential_access_pattern(cfg, md):
    """Test sequential access pattern for performance."""
    ds = IndexedDataset(cfg)

    # Access first 100 indices sequentially
    n_test = min(100, len(ds))
    prev_ex_i = -1

    for i in range(n_test):
        example = ds[i]

        # Verify indices are sequential
        expected_ex_i = i // md.content_tokens_per_example
        expected_token_i = i % md.content_tokens_per_example

        assert example["example_idx"] == expected_ex_i
        assert example["token_idx"] == expected_token_i

        # Check image indices are non-decreasing
        assert example["example_idx"] >= prev_ex_i
        prev_ex_i = example["example_idx"]


def test_invalid_layer(shards_dir):
    """Test that invalid layer raises assertion error."""
    cfg = IndexedConfig(shards=shards_dir, tokens="content", layer=99999)

    with pytest.raises(AssertionError):
        IndexedDataset(cfg)


def test_nonexistent_shard_dir():
    """Test that non-existent shard root raises RuntimeError."""
    cfg = IndexedConfig(
        shards=pathlib.Path("/nonexistent/path"), tokens="content", layer=0
    )

    with pytest.raises(RuntimeError, match="Activations are not saved"):
        IndexedDataset(cfg)


def test_edge_case_single_image(cfg, md):
    """Test edge case where we access tokens from the very last image."""
    ds = IndexedDataset(cfg)

    # Access tokens from the last image
    last_ex_idx = md.n_examples - 1

    for token_idx in range(md.content_tokens_per_example):
        global_idx = last_ex_idx * md.content_tokens_per_example + token_idx
        if global_idx < len(ds):
            example = ds[global_idx]
            assert example["example_idx"] == last_ex_idx
            assert example["token_idx"] == token_idx


def test_token_indices_within_bounds(cfg, md):
    """Test that all patch indices are within valid bounds."""
    ds = IndexedDataset(cfg)

    # Sample random indices
    rng = np.random.default_rng(42)
    test_indices = rng.integers(0, len(ds), size=min(100, len(ds)))

    for idx in test_indices:
        example = ds[idx]

        # Patch index should be within bounds
        assert 0 <= example["token_idx"] < md.content_tokens_per_example

        # Image index should be within bounds
        assert 0 <= example["example_idx"] < md.n_examples

        # Activation should have correct shape
        assert example["act"].shape == (ds.d_model,)


def test_dtype_consistency(cfg):
    """Test that data types are consistent."""
    ds = IndexedDataset(cfg)

    example = ds[0]

    # Check types
    assert isinstance(example["example_idx"], int)
    assert isinstance(example["token_idx"], int)
    assert isinstance(example["act"], torch.Tensor)
    assert example["act"].dtype == torch.float32


def test_memmap_file_access(cfg, md):
    """Test that memmap files are accessed correctly."""
    ds = IndexedDataset(cfg)

    # Test first shard file exists
    acts_fpath = os.path.join(cfg.shards, "acts000000.bin")
    assert os.path.exists(acts_fpath)

    # Verify we can access data from first and last patch of first image
    example_first = ds[0]
    example_last = ds[md.content_tokens_per_example - 1]

    assert example_first["example_idx"] == 0
    assert example_first["token_idx"] == 0
    assert example_last["example_idx"] == 0
    assert example_last["token_idx"] == md.content_tokens_per_example - 1


@pytest.mark.slow
def test_missing_shard_file_detected_at_init():
    """Test that missing shard files are detected at initialization."""
    # Use small max_tokens_per_shard to force multiple shards
    # Each image has 17 tokens (16 patches + 1 CLS), so with 2 images per shard we get 34 patches per shard
    max_tokens_per_shard = 34  # This will create ~5 shards for 10 images
    with pytest.helpers.tmp_shards_root() as shards_root:
        # Generate the activation shards
        shards_dir = pytest.helpers.write_shards(
            shards_root,
            max_tokens_per_shard=max_tokens_per_shard,
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

        # Create indexed dataset. This should raise an error at initialization because missing files should be detected early
        with pytest.raises(FileNotFoundError):
            cfg = IndexedConfig(shards=shards_dir, tokens="content", layer=0)
            IndexedDataset(cfg)
