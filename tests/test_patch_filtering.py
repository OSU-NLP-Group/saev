# tests/test_patch_filtering.py
import dataclasses

import numpy as np
import pytest
import torch.multiprocessing as mp

from saev.data import (
    IndexedConfig,
    IndexedDataset,
    Metadata,
    ShuffledConfig,
    ShuffledDataLoader,
)

mp.set_start_method("spawn", force=True)


@pytest.fixture(scope="session")
def md(shards_dir_with_token_labels):
    return Metadata.load(shards_dir_with_token_labels)


@pytest.fixture(scope="session")
def labels(shards_dir_with_token_labels, md):
    return np.memmap(
        shards_dir_with_token_labels / "labels.bin",
        mode="r",
        dtype=np.uint8,
        shape=(md.n_examples, md.content_tokens_per_example),
    )


def test_patch_filtering_no_filter(shards_dir_with_token_labels, md):
    """Test that dataloader works without filtering."""
    cfg = ShuffledConfig(
        shards=shards_dir_with_token_labels,
        tokens="content",
        layer=md.layers[0],
        batch_size=64,
        ignore_labels=[],  # No filtering - empty list
    )

    dl = ShuffledDataLoader(cfg)

    # Should be able to iterate without errors
    batch = next(iter(dl))
    assert "act" in batch
    assert "example_idx" in batch
    assert "token_idx" in batch
    assert batch["act"].shape[0] <= 64


def test_patch_filtering_ignore_single_label(shards_dir_with_token_labels, md, labels):
    """Test ignoring a single label value."""
    # Ignore background (label 0)
    ignore_label = 0

    cfg = ShuffledConfig(
        shards=shards_dir_with_token_labels,
        tokens="content",
        layer=md.layers[0],
        batch_size=32,
        ignore_labels=[ignore_label],  # Ignore label 0 (background)
    )

    dl = ShuffledDataLoader(cfg)

    # Iterate and collect some batches
    batches_collected = 0
    samples_collected = 0

    for batch in dl:
        batches_collected += 1
        samples_collected += batch["act"].shape[0]

        # Verify none of the labels are the ignored one
        for i in range(batch["act"].shape[0]):
            example_idx = batch["example_idx"][i].item()
            token_idx = batch["token_idx"][i].item()
            actual_label = labels[example_idx, token_idx]
            assert actual_label != ignore_label

        if batches_collected >= 3:  # Just check a few batches
            break

    assert batches_collected > 0
    assert samples_collected > 0


def test_patch_filtering_ignore_multiple_labels(
    shards_dir_with_token_labels, md, labels
):
    """Test ignoring multiple label values."""

    # Ignore background and class 2
    ignore_labels = [0, 2]

    cfg = ShuffledConfig(
        shards=shards_dir_with_token_labels,
        tokens="content",
        layer=md.layers[0],
        batch_size=64,
        ignore_labels=ignore_labels,
    )

    dl = ShuffledDataLoader(cfg)

    # Iterate and check
    for i, batch in enumerate(dl):
        # Verify none of the labels are in the ignored set
        for j in range(batch["act"].shape[0]):
            example_idx = batch["example_idx"][j].item()
            token_idx = batch["token_idx"][j].item()
            actual_label = labels[example_idx, token_idx]
            assert actual_label not in ignore_labels

        if i >= 2:  # Check a few batches
            break


def test_patch_filtering_ignore_all_labels(shards_dir_with_token_labels, md, labels):
    """Test ignoring all possible labels (should get no data)."""
    # Ignore all possible labels (0 through n_classes-1)
    ignore_labels = list(set(np.unique(labels).tolist()))

    cfg = ShuffledConfig(
        shards=shards_dir_with_token_labels,
        tokens="content",
        layer=md.layers[0],
        batch_size=32,
        ignore_labels=ignore_labels,
        batch_timeout_s=2.0,  # Shorter timeout for test
    )

    dl = ShuffledDataLoader(cfg)

    # Should timeout or return no data
    try:
        next(iter(dl))
        # If we get here, no patches should remain after filtering
        assert False, "Should not have gotten any data when ignoring all labels"
    except (StopIteration, TimeoutError):
        # Expected - no data matches
        pass
    finally:
        dl.shutdown()


def test_patch_filtering_missing_labels_file(shards_dir, md):
    """Test that error is raised when filtering is requested but labels.bin is missing."""
    labels_path = shards_dir / "labels.bin"
    if labels_path.exists():
        pytest.skip("--shards has labels.bin")
        return

    with pytest.raises(FileNotFoundError, match="labels.bin not found"):
        cfg = ShuffledConfig(
            shards=shards_dir,
            tokens="content",
            layer=md.layers[0],
            ignore_labels=[0, 1],  # Request filtering
        )
        ShuffledDataLoader(cfg)


def test_patch_filtering_preserves_shuffling(shards_dir_with_token_labels, md):
    """Test that filtering still maintains shuffling of data."""

    # Ignore background (0), keep all other labels
    cfg = ShuffledConfig(
        shards=shards_dir_with_token_labels,
        tokens="content",
        layer=md.layers[0],
        batch_size=16,
        ignore_labels=[0],  # Ignore background
        seed=42,
    )

    dl1 = ShuffledDataLoader(cfg)

    # Collect first batch from first iteration
    batch1 = next(iter(dl1))
    examples1 = batch1["example_idx"].numpy().copy()
    tokens1 = batch1["token_idx"].numpy().copy()
    dl1.shutdown()

    # Create new loader with different seed
    cfg2 = dataclasses.replace(cfg, seed=99)
    dl2 = ShuffledDataLoader(cfg2)

    # Collect first batch from second iteration
    batch2 = next(iter(dl2))
    examples2 = batch2["example_idx"].numpy().copy()
    tokens2 = batch2["token_idx"].numpy().copy()
    dl2.shutdown()

    # The order should be different (due to different seeds). Check that at least some elements are different
    if len(examples1) == len(examples2):  # Only compare if same length
        same_examples = np.array_equal(examples1, examples2)
        same_tokens = np.array_equal(tokens1, tokens2)
        assert not (same_examples and same_tokens)


@pytest.mark.skip(
    reason="iterating over an entire dataset is too slow; we need to use custom_shards_dir instead."
)
def test_indexed_vs_shuffled_filtering(shards_dir_with_token_labels, md, labels):
    """Test that indexed dataset correctly identifies patches that shuffled dataset would filter."""

    # Ignore background (0)
    ignore_labels = [0]

    # Count patches that should be valid using numpy
    valid_patches = set()
    for example_idx in range(md.n_examples):
        for token_idx in range(md.content_tokens_per_example):
            if labels[example_idx, token_idx] not in ignore_labels:
                valid_patches.add((example_idx, token_idx))

    # Create indexed dataset to check individual patches
    indexed_cfg = IndexedConfig(
        shards=shards_dir_with_token_labels, tokens="content", layer=md.layers[0]
    )
    indexed_ds = IndexedDataset(indexed_cfg)

    # Verify indexed dataset returns correct example_idx and token_idx for each global index
    indexed_valid = set()
    for example_idx in range(md.n_examples):
        for token_idx in range(md.content_tokens_per_example):
            # Calculate global index for this example/token combo
            global_idx = example_idx * md.content_tokens_per_example + token_idx

            # Get the example from indexed dataset
            example = indexed_ds[global_idx]

            # Verify example_idx and token_idx match our expectations
            assert example["example_idx"] == example_idx
            assert example["token_idx"] == token_idx

            # Verify token_label is included and correct
            assert "token_label" in example
            assert example["token_label"] == labels[example_idx, token_idx]

            # Also verify we can look up the correct label using returned indices
            returned_label = labels[example["example_idx"], example["token_idx"]]
            expected_label = labels[example_idx, token_idx]
            assert returned_label == expected_label

            # Check if this patch would be filtered
            if labels[example_idx, token_idx] not in ignore_labels:
                indexed_valid.add((example_idx, token_idx))

    # Verify indexed dataset identifies same valid patches
    assert indexed_valid == valid_patches

    # Now verify shuffled dataset filters correctly
    shuffled_cfg = ShuffledConfig(
        shards=shards_dir_with_token_labels,
        tokens="content",
        layer=md.layers[0],
        batch_size=8,
        ignore_labels=ignore_labels,
        seed=42,
    )
    shuffled_dl = ShuffledDataLoader(shuffled_cfg)

    # Collect patches from one full epoch of shuffled loader
    shuffled_seen = set()
    total_samples = 0

    for batch in shuffled_dl:
        batch_size = batch["act"].shape[0]
        for i in range(batch_size):
            example_idx = batch["example_idx"][i].item()
            token_idx = batch["token_idx"][i].item()

            # Verify this patch should not be filtered
            assert (example_idx, token_idx) in valid_patches

            # Double-check the label is correct
            actual_label = labels[example_idx, token_idx]
            assert actual_label not in ignore_labels

            shuffled_seen.add((example_idx, token_idx))

        total_samples += batch_size
        if total_samples >= len(valid_patches):
            break

    shuffled_dl.shutdown()

    # All seen patches should be in the valid set
    assert shuffled_seen.issubset(valid_patches)


def test_indexed_dataset_token_labels(shards_dir_with_token_labels, md, labels):
    """Test that indexed dataset correctly returns token_label field when labels.bin exists."""

    # Test with image patches
    indexed_cfg = IndexedConfig(
        shards=shards_dir_with_token_labels, tokens="content", layer=md.layers[0]
    )
    indexed_ds = IndexedDataset(indexed_cfg)

    # Test a few random indices
    test_indices = [0, 10, 50, 100, 150]  # Various indices across the dataset
    for global_idx in test_indices:
        if global_idx >= len(indexed_ds):
            continue

        example = indexed_ds[global_idx]

        # Verify token_label is present
        assert "token_label" in example

        # Verify the label value is correct
        example_idx = example["example_idx"]
        token_idx = example["token_idx"]
        expected_label = labels[example_idx, token_idx]
        assert example["token_label"] == expected_label

    # Test CLS token mode (should not have token_label)
    cls_cfg = IndexedConfig(
        shards=shards_dir_with_token_labels, tokens="special", layer=md.layers[0]
    )
    cls_ds = IndexedDataset(cls_cfg)

    # CLS tokens don't have patch labels
    cls_example = cls_ds[0]
    assert "token_label" not in cls_example


def test_indexed_dataset_no_labels_file(shards_dir, md):
    """Test that indexed dataset works without labels.bin (token_label not included)."""
    labels_path = shards_dir / "labels.bin"
    if labels_path.exists():
        pytest.skip("--shards has labels.bin")
        return

    # Create indexed dataset without labels.bin
    indexed_cfg = IndexedConfig(shards=shards_dir, tokens="content", layer=md.layers[0])
    indexed_ds = IndexedDataset(indexed_cfg)

    # Get an example - should work but not have token_label
    example = indexed_ds[0]
    assert "act" in example
    assert "example_idx" in example
    assert "token_idx" in example
    assert "token_label" not in example


@pytest.mark.skip(
    reason="iterating over an entire dataset is too slow; we need to use custom_shards_dir instead."
)
def test_patch_filtering_sees_all_valid_patches(
    shards_dir_with_token_labels, md, labels
):
    """Test that all patches with non-ignored labels are seen during iteration."""

    # Ignore background (0) and class 3
    ignore_labels = [0, 3]

    # Count how many patches should be valid
    expected_patches = set()
    for example_idx in range(md.n_examples):
        for token_idx in range(md.content_tokens_per_example):
            if labels[example_idx, token_idx] not in ignore_labels:
                expected_patches.add((example_idx, token_idx))

    # Small batch size to ensure we need multiple iterations
    cfg = ShuffledConfig(
        shards=shards_dir_with_token_labels,
        tokens="content",
        layer=md.layers[0],
        batch_size=8,
        ignore_labels=ignore_labels,
        seed=42,
    )

    dl = ShuffledDataLoader(cfg)

    # Collect all patches seen during one epoch
    seen_patches = set()
    total_samples = 0
    max_samples = len(expected_patches) * 2  # Safety limit to avoid infinite loop

    for batch in dl:
        batch_size = batch["act"].shape[0]
        for i in range(batch_size):
            example_idx = batch["example_idx"][i].item()
            token_idx = batch["token_idx"][i].item()
            seen_patches.add((example_idx, token_idx))

        total_samples += batch_size

        # Stop after seeing enough samples for one full epoch
        if total_samples >= len(expected_patches):
            break

        # Safety check
        if total_samples >= max_samples:
            break

    dl.shutdown()

    # Verify we saw exactly the expected patches
    assert seen_patches == expected_patches
