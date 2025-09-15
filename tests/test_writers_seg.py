import os

import numpy as np

import saev.data.datasets as dsets
import saev.data.writers as writers


def test_labels_bin_is_generated(tmp_path):
    """
    Expect writers to generate a labels.bin file for segmentation datasets.

    Uses a tiny FakeSeg dataset (no real files) and a temporary shard root.
    """
    cfg = writers.Config(
        data=dsets.FakeSeg(n_imgs=3, n_patches_per_img=8, n_classes=3, bg_label=0),
        dump_to=str(tmp_path),
        n_patches_per_img=8,
        vit_layers=[-2],
        d_vit=32,
        cls_token=False,
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
    )

    acts_dir = writers.get_acts_dir(cfg)
    labels_fpath = os.path.join(acts_dir, "labels.bin")

    # Expected behavior (to be implemented): labels.bin exists alongside shards for Seg datasets
    assert os.path.exists(labels_fpath)


def test_labels_bin_shape_and_dtype(tmp_path):
    """Verify labels.bin has correct shape (n_imgs, n_patches_per_img) and dtype uint8."""
    n_imgs = 5
    n_patches = 16
    cfg = writers.Config(
        data=dsets.FakeSeg(n_imgs=n_imgs, n_patches_per_img=n_patches, n_classes=10, bg_label=0),
        dump_to=str(tmp_path),
        n_patches_per_img=n_patches,
        vit_layers=[-2],
        d_vit=32,
        cls_token=False,
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
    )

    acts_dir = writers.get_acts_dir(cfg)
    labels_fpath = os.path.join(acts_dir, "labels.bin")

    # Load and check shape/dtype
    labels = np.memmap(labels_fpath, dtype=np.uint8, mode="r", shape=(n_imgs, n_patches))
    assert labels.shape == (n_imgs, n_patches)
    assert labels.dtype == np.uint8


def test_labels_bin_value_range(tmp_path):
    """Verify all label values are within [0, n_classes)."""
    n_classes = 7
    cfg = writers.Config(
        data=dsets.FakeSeg(n_imgs=10, n_patches_per_img=8, n_classes=n_classes, bg_label=0),
        dump_to=str(tmp_path),
        n_patches_per_img=8,
        vit_layers=[-2],
        d_vit=32,
        cls_token=False,
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
    )

    acts_dir = writers.get_acts_dir(cfg)
    labels_fpath = os.path.join(acts_dir, "labels.bin")

    labels = np.memmap(labels_fpath, dtype=np.uint8, mode="r", shape=(10, 8))
    assert labels.min() >= 0
    assert labels.max() < n_classes


def test_labels_bin_with_cls_token(tmp_path):
    """Test labels.bin generation when cls_token=True (CLS token should be excluded)."""
    cfg = writers.Config(
        data=dsets.FakeSeg(n_imgs=4, n_patches_per_img=8, n_classes=5, bg_label=0),
        dump_to=str(tmp_path),
        n_patches_per_img=8,
        vit_layers=[-2],
        d_vit=32,
        cls_token=True,  # CLS token present
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
    )

    acts_dir = writers.get_acts_dir(cfg)
    labels_fpath = os.path.join(acts_dir, "labels.bin")

    # Should still be (n_imgs, n_patches_per_img), excluding CLS
    labels = np.memmap(labels_fpath, dtype=np.uint8, mode="r", shape=(4, 8))
    assert labels.shape == (4, 8)  # Not (4, 9) even though tokens include CLS


def test_labels_bin_multi_shard(tmp_path):
    """Test labels.bin with enough images to span multiple shards."""
    cfg = writers.Config(
        data=dsets.FakeSeg(n_imgs=100, n_patches_per_img=196, n_classes=150, bg_label=0),
        dump_to=str(tmp_path),
        n_patches_per_img=196,
        vit_layers=[-2, -1],
        d_vit=768,
        cls_token=False,
        vit_batch_size=8,
        n_workers=0,
        device="cpu",
        max_patches_per_shard=10000,  # Force multiple shards
    )

    acts_dir = writers.get_acts_dir(cfg)
    labels_fpath = os.path.join(acts_dir, "labels.bin")

    labels = np.memmap(labels_fpath, dtype=np.uint8, mode="r", shape=(100, 196))
    assert labels.shape == (100, 196)
    assert os.path.getsize(labels_fpath) == 100 * 196  # uint8 = 1 byte per value


def test_no_labels_bin_for_non_seg_dataset(tmp_path):
    """Verify labels.bin is NOT generated for non-segmentation datasets."""
    cfg = writers.Config(
        data=dsets.Fake(n_imgs=5),  # Regular Fake dataset, not FakeSeg
        dump_to=str(tmp_path),
        n_patches_per_img=16,
        vit_layers=[-2],
        d_vit=32,
        cls_token=False,
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
    )

    acts_dir = writers.get_acts_dir(cfg)
    labels_fpath = os.path.join(acts_dir, "labels.bin")

    # Should NOT exist for non-seg datasets
    assert not os.path.exists(labels_fpath)
