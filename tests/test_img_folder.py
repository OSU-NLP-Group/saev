"""Tests for ImgFolder and ImgFolderDataset."""

import pathlib

import pytest
from PIL import Image

from saev.data.datasets import ImgFolder, get_dataset


# ImgFolder integration tests (require --imgfolder argument)


def test_imgfolder_dataset_loads(imgfolder_root: pathlib.Path):
    """Test that ImgFolder dataset can be loaded."""
    cfg = ImgFolder(root=imgfolder_root)
    ds = get_dataset(cfg)

    assert len(ds) > 0, "Dataset should have examples"


@pytest.mark.timeout(3600)
def test_imgfolder_all_images_loadable(imgfolder_root: pathlib.Path):
    """Test that every image in the dataset can be loaded and converted to RGB."""
    cfg = ImgFolder(root=imgfolder_root)
    ds = get_dataset(cfg)

    errors = []
    for i in range(len(ds)):
        try:
            sample = ds[i]
            assert "data" in sample
            img = sample["data"]
            assert isinstance(img, Image.Image)
            # Force full load to catch truncated images
            img.load()
            img.convert("RGB")
        except Exception as e:
            # Get file path from dataset
            path, _ = ds.samples[i]
            errors.append((i, path, e))

    assert not errors, f"Failed to load {len(errors)} images:\n" + "\n".join(
        f"  {i} ({path}): {e}" for i, path, e in errors[:20]
    )


def test_imgfolder_sample_structure(imgfolder_root: pathlib.Path):
    """Test that samples have the expected structure."""
    cfg = ImgFolder(root=imgfolder_root)
    ds = get_dataset(cfg)

    sample = ds[0]

    assert "data" in sample
    assert "target" in sample
    assert "label" in sample
    assert "index" in sample
    assert isinstance(sample["data"], Image.Image)
    assert isinstance(sample["target"], int)
    assert isinstance(sample["label"], str)
    assert sample["index"] == 0
