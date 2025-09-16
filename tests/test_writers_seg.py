import os

import numpy as np
import torch
from PIL import Image

import saev.data.datasets as dsets
import saev.data.writers as writers


def test_labels_bin_is_generated(tmp_path):
    """
    Expect writers to generate a labels.bin file for segmentation datasets.

    Uses a tiny FakeSeg dataset (no real files) and a temporary shard root.
    """
    # Use fake-clip which properly handles the tiny test model
    # The tiny model uses 8x8 images with 2x2 patches = 4x4 = 16 patches
    cfg = writers.Config(
        data=dsets.FakeSeg(n_imgs=4, n_patches_per_img=16, n_classes=3, bg_label=0),
        dump_to=str(tmp_path),
        n_patches_per_img=16,
        vit_layers=[-2],
        d_vit=128,
        cls_token=False,
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
        vit_family="fake-clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
    )

    # Actually run the writer to generate files
    writers.worker_fn(cfg)

    # Get the activations directory
    acts_dir = writers.get_acts_dir(cfg)
    labels_path = os.path.join(acts_dir, "labels.bin")

    # Check that labels.bin was created
    assert os.path.exists(labels_path), (
        "labels.bin should be created for FakeSeg dataset"
    )

    # Check that the file has content
    file_size = os.path.getsize(labels_path)
    expected_size = cfg.data.n_imgs * cfg.n_patches_per_img  # 4 * 16 = 64 bytes
    assert file_size == expected_size, (
        f"labels.bin size {file_size} != expected {expected_size}"
    )


def test_labels_bin_shape_and_dtype(tmp_path):
    """Test that labels.bin has the correct shape and dtype."""
    cfg = writers.Config(
        data=dsets.FakeSeg(n_imgs=2, n_patches_per_img=16, n_classes=3),
        dump_to=str(tmp_path),
        n_patches_per_img=16,
        vit_layers=[-2],
        d_vit=128,
        cls_token=False,
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
        vit_family="fake-clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
    )

    writers.worker_fn(cfg)

    acts_dir = writers.get_acts_dir(cfg)
    labels_path = os.path.join(acts_dir, "labels.bin")

    # Load the labels
    labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(2, 16)

    assert labels.shape == (2, 16)
    assert labels.dtype == np.uint8


def test_labels_bin_value_range(tmp_path):
    """Test that labels.bin contains valid class indices."""
    n_classes = 5
    cfg = writers.Config(
        data=dsets.FakeSeg(n_imgs=3, n_patches_per_img=16, n_classes=n_classes),
        dump_to=str(tmp_path),
        n_patches_per_img=16,
        vit_layers=[-2],
        d_vit=128,
        cls_token=False,
        vit_batch_size=4,
        n_workers=0,
        device="cpu",
        vit_family="fake-clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
    )

    writers.worker_fn(cfg)

    acts_dir = writers.get_acts_dir(cfg)
    labels_path = os.path.join(acts_dir, "labels.bin")

    # Load the labels
    labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(3, 16)

    # Check that all values are in valid range [0, n_classes)
    assert labels.min() >= 0
    assert labels.max() < n_classes


def test_labels_bin_with_cls_token(tmp_path):
    """Test that labels.bin works correctly even when CLS token is used."""
    cfg = writers.Config(
        data=dsets.FakeSeg(n_imgs=2, n_patches_per_img=16, n_classes=3),
        dump_to=str(tmp_path),
        n_patches_per_img=16,  # This is image patches, not including CLS
        vit_layers=[-2],
        d_vit=128,
        cls_token=True,  # Enable CLS token
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
        vit_family="fake-clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
    )

    writers.worker_fn(cfg)

    acts_dir = writers.get_acts_dir(cfg)
    labels_path = os.path.join(acts_dir, "labels.bin")

    # Load the labels - should still be (n_imgs, n_patches_per_img)
    labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(2, 16)

    # Labels should not include CLS token, so shape should be (2, 16)
    assert labels.shape == (2, 16)


def test_labels_bin_multi_shard(tmp_path):
    """Test that labels.bin works correctly with multiple shards."""
    cfg = writers.Config(
        data=dsets.FakeSeg(n_imgs=10, n_patches_per_img=16, n_classes=4),
        dump_to=str(tmp_path),
        n_patches_per_img=16,
        vit_layers=[-2, -1],  # Multiple layers (tiny model only has 2 layers)
        d_vit=128,
        cls_token=False,
        vit_batch_size=10,
        n_workers=0,
        device="cpu",
        vit_family="fake-clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        max_patches_per_shard=80,  # Force multiple shards
    )

    writers.worker_fn(cfg)

    acts_dir = writers.get_acts_dir(cfg)
    labels_path = os.path.join(acts_dir, "labels.bin")

    # Load the labels
    labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(10, 16)

    # Should have all 10 images worth of labels
    assert labels.shape == (10, 16)
    assert labels.min() >= 0
    assert labels.max() < 4


def test_no_labels_bin_for_non_seg_dataset(tmp_path):
    """Test that labels.bin is NOT created for non-segmentation datasets."""
    cfg = writers.Config(
        data=dsets.Fake(n_imgs=2),  # Regular Fake, not FakeSeg
        dump_to=str(tmp_path),
        n_patches_per_img=16,
        vit_layers=[-2],
        d_vit=128,
        cls_token=False,
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
        vit_family="fake-clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
    )

    writers.worker_fn(cfg)

    acts_dir = writers.get_acts_dir(cfg)
    labels_path = os.path.join(acts_dir, "labels.bin")

    # Check that labels.bin was NOT created
    assert not os.path.exists(labels_path), (
        "labels.bin should not be created for non-segmentation datasets"
    )


def test_pixel_to_patch_labels_mode():
    """Test pixel_to_patch_labels with mode transformation."""

    # Create a 4x4 image with 4 2x2 patches
    # Patch layout:
    # [0, 0, 1, 1]
    # [0, 0, 1, 1]  -> Patch 0=[0,0,0,0], mode=0
    # [2, 2, 3, 3]  -> Patch 1=[1,1,1,1], mode=1
    # [2, 2, 3, 3]  -> Patch 2=[2,2,2,2], mode=2
    #                  Patch 3=[3,3,3,3], mode=3
    seg_array = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.uint8
    )
    segmentation = Image.fromarray(seg_array)

    patch_labels = writers.pixel_to_patch_labels(
        segmentation, n_patches=4, patch_size=2, label_transform="mode"
    )

    expected = torch.tensor([0, 1, 2, 3], dtype=torch.uint8)
    torch.testing.assert_close(patch_labels, expected)


def test_pixel_to_patch_labels_no_bg():
    """Test pixel_to_patch_labels with no-bg transformation."""

    # Create a 4x4 image with 4 2x2 patches
    # Use 0 as background
    # Patch layout:
    # [0, 0, 1, 1]
    # [0, 2, 1, 1]  -> Patch 0 has [0,0,0,2], no-bg mode=2 (ignores bg=0)
    # [0, 0, 3, 3]  -> Patch 1 has [1,1,1,1], no-bg mode=1
    # [0, 0, 3, 3]  -> Patch 2 has [0,0,0,0], no-bg mode=0 (all bg)
    #                  Patch 3 has [3,3,3,3], no-bg mode=3
    seg_array = np.array(
        [[0, 0, 1, 1], [0, 2, 1, 1], [0, 0, 3, 3], [0, 0, 3, 3]], dtype=np.uint8
    )
    segmentation = Image.fromarray(seg_array)

    patch_labels = writers.pixel_to_patch_labels(
        segmentation,
        n_patches=4,
        patch_size=2,
        label_transform="no-bg",
        bg_label=0,
    )

    expected = torch.tensor([2, 1, 0, 3], dtype=torch.uint8)
    torch.testing.assert_close(patch_labels, expected)


def test_pixel_to_patch_labels_pil_image():
    """Test pixel_to_patch_labels with PIL Image input (as required)."""
    import torch

    # Create a 4x4 PIL Image with 4 2x2 patches
    seg_array = np.array(
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.uint8
    )
    segmentation = Image.fromarray(seg_array)

    patch_labels = writers.pixel_to_patch_labels(
        segmentation, n_patches=4, patch_size=2
    )

    expected = torch.tensor([0, 1, 2, 3], dtype=torch.uint8)
    torch.testing.assert_close(patch_labels, expected)


def test_pixel_to_patch_labels_with_background():
    """Test pixel_to_patch_labels handles background label correctly."""
    import torch

    # Create an image where background (0) appears frequently
    seg_array = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 1], [0, 2, 2, 2], [2, 2, 2, 2]], dtype=np.uint8
    )
    segmentation = Image.fromarray(seg_array)

    patch_labels = writers.pixel_to_patch_labels(
        segmentation, n_patches=4, patch_size=2
    )

    expected = torch.tensor([0, 1, 2, 2], dtype=torch.uint8)
    torch.testing.assert_close(patch_labels, expected)


def test_pixel_to_patch_labels_larger_grid():
    """Test pixel_to_patch_labels with a larger patch grid (16 patches)."""

    # Create an 8x8 image with 16 2x2 patches
    seg_array = np.zeros((8, 8), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            # Each 2x2 patch gets a unique label
            label = i * 4 + j
            seg_array[i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2] = label

    segmentation = Image.fromarray(seg_array)

    patch_labels = writers.pixel_to_patch_labels(
        segmentation, n_patches=16, patch_size=2
    )

    expected = torch.arange(16, dtype=torch.uint8)
    torch.testing.assert_close(patch_labels, expected)


def test_pixel_to_patch_labels_non_square():
    """Test pixel_to_patch_labels with non-square patch grids."""

    # 4x2 image -> 2x1 patches (patch_size=2)
    seg_array = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.uint8)
    segmentation = Image.fromarray(seg_array)

    patch_labels = writers.pixel_to_patch_labels(
        segmentation, n_patches=2, patch_size=2
    )

    expected = torch.tensor([0, 1], dtype=torch.uint8)
    torch.testing.assert_close(patch_labels, expected)

    # 2x4 image -> 1x2 patches (patch_size=2)
    seg_array = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8)
    segmentation = Image.fromarray(seg_array)

    patch_labels = writers.pixel_to_patch_labels(
        segmentation, n_patches=2, patch_size=2
    )

    expected = torch.tensor([0, 1], dtype=torch.uint8)
    torch.testing.assert_close(patch_labels, expected)
