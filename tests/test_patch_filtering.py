# tests/test_shuffled_patch_filtering.py
import dataclasses
import os
import tempfile

import numpy as np
import pytest
import torch.multiprocessing as mp

from saev.data import Dataset as IndexedDataset
from saev.data import (
    IndexedConfig,
    ShuffledConfig,
    ShuffledDataLoader,
    datasets,
    writers,
)

mp.set_start_method("spawn", force=True)


@pytest.fixture
def segmentation_shards(tmp_path):
    """Create test shards with segmentation labels."""
    # Create a small dataset with segmentation
    n_imgs = 20
    n_patches_per_img = 16
    n_classes = 5

    cfg = writers.Config(
        data=datasets.FakeSeg(
            n_imgs=n_imgs,
            n_patches_per_img=n_patches_per_img,
            n_classes=n_classes,
            bg_label=0,
        ),
        dump_to=str(tmp_path),
        n_patches_per_img=n_patches_per_img,
        vit_layers=[-2],
        d_model=128,
        cls_token=False,
        vit_batch_size=5,
        n_workers=0,
        device="cpu",
        vit_family="fake-clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        max_patches_per_shard=n_imgs * n_patches_per_img,  # Single shard for simplicity
    )

    # Generate the activation shards
    writers.worker_fn(cfg)

    # Get the actual shard directory
    metadata = writers.Metadata.from_cfg(cfg)
    shard_root = os.path.join(str(tmp_path), metadata.hash)

    return shard_root, n_imgs, n_patches_per_img, n_classes


def test_patch_filtering_no_filter(segmentation_shards):
    """Test that dataloader works without filtering."""
    shard_root, n_imgs, n_patches_per_img, _ = segmentation_shards

    cfg = ShuffledConfig(
        shard_root=shard_root,
        patches="image",
        layer=-2,
        batch_size=64,
        ignore_labels=[],  # No filtering - empty list
    )

    dl = ShuffledDataLoader(cfg)

    # Should be able to iterate without errors
    batch = next(iter(dl))
    assert "act" in batch
    assert "image_i" in batch
    assert "patch_i" in batch
    assert batch["act"].shape[0] == 64


def test_patch_filtering_ignore_single_label(segmentation_shards):
    """Test ignoring a single label value."""
    shard_root, n_imgs, n_patches_per_img, _ = segmentation_shards

    # Load labels to see what's available
    labels_path = os.path.join(shard_root, "labels.bin")
    labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(
        n_imgs, n_patches_per_img
    )

    # Ignore background (label 0)
    ignore_label = 0

    cfg = ShuffledConfig(
        shard_root=shard_root,
        patches="image",
        layer=-2,
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

        # Check that we got valid data
        assert "act" in batch
        assert "image_i" in batch
        assert "patch_i" in batch

        # Verify none of the labels are the ignored one
        for i in range(batch["act"].shape[0]):
            img_i = batch["image_i"][i].item()
            patch_i = batch["patch_i"][i].item()
            actual_label = labels[img_i, patch_i]
            assert actual_label != ignore_label, (
                f"Should not have label {ignore_label}, but got it"
            )

        if batches_collected >= 3:  # Just check a few batches
            break

    assert batches_collected > 0, "Should have gotten at least some batches"
    assert samples_collected > 0, "Should have gotten at least some samples"


def test_patch_filtering_ignore_multiple_labels(segmentation_shards):
    """Test ignoring multiple label values."""
    shard_root, n_imgs, n_patches_per_img, n_classes = segmentation_shards

    # Load labels to see what's available
    labels_path = os.path.join(shard_root, "labels.bin")
    labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(
        n_imgs, n_patches_per_img
    )

    # Ignore background and class 2
    ignore_labels = [0, 2]

    cfg = ShuffledConfig(
        shard_root=shard_root,
        patches="image",
        layer=-2,
        batch_size=32,
        ignore_labels=ignore_labels,
    )

    dl = ShuffledDataLoader(cfg)

    # Iterate and check
    for i, batch in enumerate(dl):
        # Verify none of the labels are in the ignored set
        for j in range(batch["act"].shape[0]):
            img_i = batch["image_i"][j].item()
            patch_i = batch["patch_i"][j].item()
            actual_label = labels[img_i, patch_i]
            assert actual_label not in ignore_labels, (
                f"Label {actual_label} should be ignored (in {ignore_labels})"
            )

        if i >= 2:  # Check a few batches
            break


def test_patch_filtering_ignore_all_labels(segmentation_shards):
    """Test ignoring all possible labels (should get no data)."""
    shard_root, n_imgs, n_patches_per_img, n_classes = segmentation_shards

    # Ignore all possible labels (0 through n_classes-1)
    ignore_labels = list(range(n_classes))

    cfg = ShuffledConfig(
        shard_root=shard_root,
        patches="image",
        layer=-2,
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


def test_patch_filtering_missing_labels_file():
    """Test that error is raised when filtering is requested but labels.bin is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a minimal shard setup WITHOUT labels.bin
        metadata = {
            "vit_family": "fake-clip",
            "vit_ckpt": "test",
            "layers": [-2],
            "n_patches_per_img": 16,
            "cls_token": False,
            "d_model": 128,
            "n_imgs": 10,
            "max_patches_per_shard": 160,
            "data": {"n_imgs": 10},
            "dtype": "float32",
            "protocol": "1.1",
        }

        # Write metadata
        import json

        with open(os.path.join(tmp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Write shards.json
        shards = [{"name": f"acts{i:06d}.bin", "n_imgs": 10} for i in range(1)]
        with open(os.path.join(tmp_dir, "shards.json"), "w") as f:
            json.dump(shards, f)

        # Create a dummy acts file
        acts = np.zeros((10, 1, 17, 128), dtype=np.float32)
        acts.tofile(os.path.join(tmp_dir, "acts000000.bin"))

        # Try to create dataloader with filtering - should fail
        with pytest.raises(FileNotFoundError, match="labels.bin not found"):
            cfg = ShuffledConfig(
                shard_root=tmp_dir,
                patches="image",
                layer=-2,
                ignore_labels=[0, 1],  # Request filtering
            )
            ShuffledDataLoader(cfg)


def test_patch_filtering_preserves_shuffling(segmentation_shards):
    """Test that filtering still maintains shuffling of data."""
    shard_root, n_imgs, n_patches_per_img, _ = segmentation_shards

    # Ignore background (0), keep all other labels
    cfg = ShuffledConfig(
        shard_root=shard_root,
        patches="image",
        layer=-2,
        batch_size=16,
        ignore_labels=[0],  # Ignore background
        seed=42,
    )

    dl1 = ShuffledDataLoader(cfg)

    # Collect first batch from first iteration
    batch1 = next(iter(dl1))
    images1 = batch1["image_i"].numpy().copy()
    patches1 = batch1["patch_i"].numpy().copy()
    dl1.shutdown()

    # Create new loader with different seed
    cfg2 = dataclasses.replace(cfg, seed=99)
    dl2 = ShuffledDataLoader(cfg2)

    # Collect first batch from second iteration
    batch2 = next(iter(dl2))
    images2 = batch2["image_i"].numpy().copy()
    patches2 = batch2["patch_i"].numpy().copy()
    dl2.shutdown()

    # The order should be different (due to different seeds)
    # Check that at least some elements are different
    if len(images1) == len(images2):  # Only compare if same length
        same_images = np.array_equal(images1, images2)
        same_patches = np.array_equal(patches1, patches2)
        assert not (same_images and same_patches), (
            "Data should be shuffled differently with different seeds"
        )


def test_indexed_vs_shuffled_filtering(segmentation_shards):
    """Test that indexed dataset correctly identifies patches that shuffled dataset would filter."""
    shard_root, n_imgs, n_patches_per_img, _ = segmentation_shards

    # Load labels to identify expected patches
    labels_path = os.path.join(shard_root, "labels.bin")
    labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(
        n_imgs, n_patches_per_img
    )

    # Ignore background (0)
    ignore_labels = [0]

    # Count patches that should be valid using numpy
    valid_patches = set()
    for img_i in range(n_imgs):
        for patch_i in range(n_patches_per_img):
            if labels[img_i, patch_i] not in ignore_labels:
                valid_patches.add((img_i, patch_i))

    # Create indexed dataset to check individual patches
    indexed_cfg = IndexedConfig(
        shard_root=shard_root,
        patches="image",
        layer=-2,
    )
    indexed_ds = IndexedDataset(indexed_cfg)

    # Verify indexed dataset returns correct image_i and patch_i for each global index
    indexed_valid = set()
    for img_i in range(n_imgs):
        for patch_i in range(n_patches_per_img):
            # Calculate global index for this image/patch combo
            global_idx = img_i * n_patches_per_img + patch_i

            # Get the example from indexed dataset
            example = indexed_ds[global_idx]

            # Verify image_i and patch_i match our expectations
            assert example["image_i"] == img_i, (
                f"Global index {global_idx}: expected image_i={img_i}, got {example['image_i']}"
            )
            assert example["patch_i"] == patch_i, (
                f"Global index {global_idx}: expected patch_i={patch_i}, got {example['patch_i']}"
            )

            # Verify patch_label is included and correct
            assert "patch_label" in example, (
                "patch_label should be included when labels.bin exists"
            )
            assert example["patch_label"] == labels[img_i, patch_i], (
                f"patch_label mismatch for ({img_i}, {patch_i}): "
                f"expected {labels[img_i, patch_i]}, got {example['patch_label']}"
            )

            # Also verify we can look up the correct label using returned indices
            returned_label = labels[example["image_i"], example["patch_i"]]
            expected_label = labels[img_i, patch_i]
            assert returned_label == expected_label, (
                f"Label mismatch for ({img_i}, {patch_i}): "
                f"direct lookup gives {expected_label}, "
                f"indexed lookup gives {returned_label}"
            )

            # Check if this patch would be filtered
            if labels[img_i, patch_i] not in ignore_labels:
                indexed_valid.add((img_i, patch_i))

    # Verify indexed dataset identifies same valid patches
    assert indexed_valid == valid_patches, (
        f"Indexed dataset should identify same valid patches. "
        f"Expected {len(valid_patches)}, got {len(indexed_valid)}"
    )

    # Now verify shuffled dataset filters correctly
    shuffled_cfg = ShuffledConfig(
        shard_root=shard_root,
        patches="image",
        layer=-2,
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
            img_i = batch["image_i"][i].item()
            patch_i = batch["patch_i"][i].item()

            # Verify this patch should not be filtered
            assert (img_i, patch_i) in valid_patches, (
                f"Shuffled loader returned filtered patch ({img_i}, {patch_i})"
            )

            # Double-check the label is correct
            actual_label = labels[img_i, patch_i]
            assert actual_label not in ignore_labels, (
                f"Patch ({img_i}, {patch_i}) has label {actual_label} which should be filtered"
            )

            shuffled_seen.add((img_i, patch_i))

        total_samples += batch_size
        if total_samples >= len(valid_patches):
            break

    shuffled_dl.shutdown()

    # All seen patches should be in the valid set
    assert shuffled_seen.issubset(valid_patches), (
        "Shuffled loader should only return non-filtered patches"
    )


def test_indexed_dataset_patch_labels(segmentation_shards):
    """Test that indexed dataset correctly returns patch_label field when labels.bin exists."""
    shard_root, n_imgs, n_patches_per_img, n_classes = segmentation_shards

    # Load labels for verification
    labels_path = os.path.join(shard_root, "labels.bin")
    labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(
        n_imgs, n_patches_per_img
    )

    # Test with image patches
    indexed_cfg = IndexedConfig(
        shard_root=shard_root,
        patches="image",
        layer=-2,
    )
    indexed_ds = IndexedDataset(indexed_cfg)

    # Test a few random indices
    test_indices = [0, 10, 50, 100, 150]  # Various indices across the dataset
    for global_idx in test_indices:
        if global_idx >= len(indexed_ds):
            continue

        example = indexed_ds[global_idx]

        # Verify patch_label is present
        assert "patch_label" in example, (
            f"patch_label should be present for index {global_idx}"
        )

        # Verify the label value is correct
        img_i = example["image_i"]
        patch_i = example["patch_i"]
        expected_label = labels[img_i, patch_i]
        assert example["patch_label"] == expected_label, (
            f"Label mismatch at index {global_idx}: "
            f"expected {expected_label}, got {example['patch_label']}"
        )

    # Test CLS token mode (should not have patch_label)
    cls_cfg = IndexedConfig(
        shard_root=shard_root,
        patches="cls",
        layer=-2,
    )
    cls_ds = IndexedDataset(cls_cfg)

    # CLS tokens don't have patch labels
    cls_example = cls_ds[0]
    assert "patch_label" not in cls_example, (
        "CLS tokens should not have patch_label field"
    )


def test_indexed_dataset_no_labels_file():
    """Test that indexed dataset works without labels.bin (patch_label not included)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a minimal shard setup WITHOUT labels.bin
        metadata = {
            "vit_family": "fake-clip",
            "vit_ckpt": "test",
            "layers": [-2],
            "n_patches_per_img": 16,
            "cls_token": False,
            "d_model": 128,
            "n_imgs": 10,
            "max_patches_per_shard": 160,
            "data": {"n_imgs": 10},
            "dtype": "float32",
            "protocol": "1.1",
        }

        # Write metadata
        import json

        with open(os.path.join(tmp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Write shards.json
        shards = [{"name": f"acts{i:06d}.bin", "n_imgs": 10} for i in range(1)]
        with open(os.path.join(tmp_dir, "shards.json"), "w") as f:
            json.dump(shards, f)

        # Create a dummy acts file
        acts = np.zeros((10, 1, 16, 128), dtype=np.float32)
        acts.tofile(os.path.join(tmp_dir, "acts000000.bin"))

        # Create indexed dataset without labels.bin
        indexed_cfg = IndexedConfig(
            shard_root=tmp_dir,
            patches="image",
            layer=-2,
        )
        indexed_ds = IndexedDataset(indexed_cfg)

        # Get an example - should work but not have patch_label
        example = indexed_ds[0]
        assert "act" in example
        assert "image_i" in example
        assert "patch_i" in example
        assert "patch_label" not in example, (
            "patch_label should not be present when labels.bin doesn't exist"
        )


def test_patch_filtering_sees_all_valid_patches(segmentation_shards):
    """Test that all patches with non-ignored labels are seen during iteration."""
    shard_root, n_imgs, n_patches_per_img, n_classes = segmentation_shards

    # Load labels to identify expected patches
    labels_path = os.path.join(shard_root, "labels.bin")
    labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(
        n_imgs, n_patches_per_img
    )

    # Ignore background (0) and class 3
    ignore_labels = [0, 3]

    # Count how many patches should be valid
    expected_patches = set()
    for img_i in range(n_imgs):
        for patch_i in range(n_patches_per_img):
            if labels[img_i, patch_i] not in ignore_labels:
                expected_patches.add((img_i, patch_i))

    # Small batch size to ensure we need multiple iterations
    cfg = ShuffledConfig(
        shard_root=shard_root,
        patches="image",
        layer=-2,
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
            img_i = batch["image_i"][i].item()
            patch_i = batch["patch_i"][i].item()
            seen_patches.add((img_i, patch_i))

        total_samples += batch_size

        # Stop after seeing enough samples for one full epoch
        if total_samples >= len(expected_patches):
            break

        # Safety check
        if total_samples >= max_samples:
            break

    dl.shutdown()

    # Verify we saw exactly the expected patches
    assert seen_patches == expected_patches, (
        f"Should see all valid patches. "
        f"Expected {len(expected_patches)}, got {len(seen_patches)}. "
        f"Missing: {expected_patches - seen_patches}. "
        f"Extra: {seen_patches - expected_patches}"
    )
