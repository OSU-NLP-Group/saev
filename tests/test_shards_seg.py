import contextlib
import pathlib
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from saev.data import datasets
from saev.data.shards import pixel_to_patch_labels, worker_fn


@contextlib.contextmanager
def tmp_shards_root():
    """Create a temporary shard root directory."""
    # We cannot use the tmp_path fixture because of Hypothesis.
    # See https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck.function_scoped_fixture
    with tempfile.TemporaryDirectory() as tmp_path:
        shards_root = pathlib.Path(tmp_path) / "saev" / "shards"
        shards_root.mkdir(parents=True)
        yield shards_root


@pytest.fixture
def data_cfg():
    return datasets.FakeImgSeg(
        n_examples=4, content_tokens_per_example=16, n_classes=3, bg_label=0
    )


def test_labels_bin_is_generated(data_cfg):
    """
    Expect shards to generate a labels.bin file for segmentation datasets.

    Uses a tiny FakeImgSeg dataset (no real files) and a temporary shard root.
    """
    with tmp_shards_root() as shards_root:
        # Use fake-clip which properly handles the tiny test model. The tiny model uses 8x8 images with 2x2 patches = 4x4 = 16 patches
        shards_dir = worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-1],
            data=data_cfg,
            batch_size=2,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
            pixel_agg="prefer-fg",
        )

        labels_path = shards_dir / "labels.bin"

        # Check that labels.bin was created
        assert labels_path.is_file(), (
            "labels.bin should be created for FakeImgSeg dataset"
        )

        # Check that the file has content
        file_size = labels_path.stat().st_size
        expected_size = data_cfg.n_examples * 16  # 4 * 16 = 64 bytes
        assert file_size == expected_size, (
            f"labels.bin size {file_size} != expected {expected_size}"
        )


def test_labels_bin_shape_and_dtype(data_cfg):
    """Test that labels.bin has the correct shape and dtype."""
    with tmp_shards_root() as shards_root:
        shards_dir = worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-2],
            data=data_cfg,
            batch_size=2,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
            pixel_agg="prefer-fg",
        )

        labels_path = shards_dir / "labels.bin"

        # Load the labels
        labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(
            data_cfg.n_examples, 16
        )

        assert labels.shape == (data_cfg.n_examples, 16)
        assert labels.dtype == np.uint8


def test_labels_bin_value_range(data_cfg):
    """Test that labels.bin contains valid class indices."""
    n_classes = 5
    with tmp_shards_root() as shards_root:
        shards_dir = worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-2],
            data=data_cfg,
            batch_size=4,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
            pixel_agg="prefer-fg",
        )

        labels_path = shards_dir / "labels.bin"

        # Load the labels
        labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(
            data_cfg.n_examples, 16
        )

        # Check that all values are in valid range [0, n_classes)
        assert labels.min() >= 0
        assert labels.max() < n_classes


def test_labels_bin_with_cls_token(data_cfg):
    """Test that labels.bin works correctly even when CLS token is used."""
    with tmp_shards_root() as shards_root:
        shards_dir = worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,  # This is image patches, not including CLS
            cls_token=True,  # Enable CLS token
            d_model=128,
            layers=[-2],
            data=data_cfg,
            batch_size=2,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
            pixel_agg="prefer-fg",
        )

        labels_path = shards_dir / "labels.bin"

        # Load the labels - should still be (n_examples, n_content_tokens_per_example)
        labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(
            data_cfg.n_examples, 16
        )

        # Labels should not include CLS token, so shape should be (2, 16)
        assert labels.shape == (data_cfg.n_examples, 16)


def test_labels_bin_multi_shard(data_cfg):
    """Test that labels.bin works correctly with multiple shards."""
    with tmp_shards_root() as shards_root:
        shards_dir = worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-2, -1],  # Multiple layers (tiny model only has 2 layers)
            data=data_cfg,
            batch_size=10,
            n_workers=0,
            max_tokens_per_shard=80,  # Force multiple shards
            shards_root=shards_root,
            device="cpu",
            pixel_agg="prefer-fg",
        )

        labels_path = shards_dir / "labels.bin"

        # Load the labels
        labels = np.memmap(labels_path, mode="r", dtype=np.uint8).reshape(
            data_cfg.n_examples, 16
        )

        # Should have all 10 images worth of labels
        assert labels.shape == (data_cfg.n_examples, 16)
        assert labels.min() >= 0
        assert labels.max() < 4


def test_no_labels_bin_for_non_seg_dataset():
    """Test that labels.bin is NOT created for non-segmentation datasets."""
    with tmp_shards_root() as shards_root:
        data_cfg = datasets.FakeImg(n_examples=2)  # Regular FakeImg, not FakeImgSeg
        shards_dir = worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-2],
            data=data_cfg,
            batch_size=2,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
        )

        labels_path = shards_dir / "labels.bin"

        # Check that labels.bin was NOT created
        assert not labels_path.exists(), (
            "labels.bin should not be created for non-segmentation datasets"
        )


def test_pixel_to_patch_labels_majority():
    """Test pixel_to_patch_labels with majority transformation."""

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

    patch_labels = pixel_to_patch_labels(
        segmentation, n_patches=4, patch_size=2, pixel_agg="majority"
    )

    expected = torch.tensor([0, 1, 2, 3], dtype=torch.uint8)
    torch.testing.assert_close(patch_labels, expected)


def test_pixel_to_patch_labels_prefer_fg():
    """Test pixel_to_patch_labels with prefer-fg transformation."""

    # Create a 4x4 image with 4 2x2 patches
    # Use 0 as background
    # Patch layout:
    # [0, 0, 1, 1]
    # [0, 2, 1, 1]  -> Patch 0 has [0,0,0,2], prefer-fg=2 (ignores bg=0)
    # [0, 0, 3, 3]  -> Patch 1 has [1,1,1,1], prefer-fg=1
    # [0, 0, 3, 3]  -> Patch 2 has [0,0,0,0], prefer-fg=0 (all bg)
    #                  Patch 3 has [3,3,3,3], prefer-fg=3
    seg_array = np.array(
        [[0, 0, 1, 1], [0, 2, 1, 1], [0, 0, 3, 3], [0, 0, 3, 3]], dtype=np.uint8
    )
    segmentation = Image.fromarray(seg_array)

    patch_labels = pixel_to_patch_labels(
        segmentation,
        n_patches=4,
        patch_size=2,
        pixel_agg="prefer-fg",
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

    patch_labels = pixel_to_patch_labels(segmentation, n_patches=4, patch_size=2)

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

    patch_labels = pixel_to_patch_labels(segmentation, n_patches=4, patch_size=2)

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

    patch_labels = pixel_to_patch_labels(segmentation, n_patches=16, patch_size=2)

    expected = torch.arange(16, dtype=torch.uint8)
    torch.testing.assert_close(patch_labels, expected)


def test_pixel_to_patch_labels_non_square():
    """Test pixel_to_patch_labels with non-square patch grids."""

    # 4x2 image -> 2x1 patches (patch_size=2)
    seg_array = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.uint8)
    segmentation = Image.fromarray(seg_array)

    patch_labels = pixel_to_patch_labels(segmentation, n_patches=2, patch_size=2)

    expected = torch.tensor([0, 1], dtype=torch.uint8)
    torch.testing.assert_close(patch_labels, expected)

    # 2x4 image -> 1x2 patches (patch_size=2)
    seg_array = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8)
    segmentation = Image.fromarray(seg_array)

    patch_labels = pixel_to_patch_labels(segmentation, n_patches=2, patch_size=2)

    expected = torch.tensor([0, 1], dtype=torch.uint8)
    torch.testing.assert_close(patch_labels, expected)
