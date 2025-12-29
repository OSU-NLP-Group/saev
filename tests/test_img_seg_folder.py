"""Tests for ImgSegFolder and ImgSegFolderDataset."""

import pathlib

import numpy as np
import pytest
from PIL import Image

from saev.data.datasets import ImgSegFolder, ImgSegFolderDataset


@pytest.fixture
def img_seg_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a minimal ImgSegFolder directory structure."""
    root = tmp_path / "seg_dataset"

    # Create directory structure
    (root / "images" / "training").mkdir(parents=True)
    (root / "images" / "validation").mkdir(parents=True)
    (root / "annotations" / "training").mkdir(parents=True)
    (root / "annotations" / "validation").mkdir(parents=True)

    # Create some dummy images and masks for training split
    n_train = 5
    img_size = (64, 64)
    for i in range(n_train):
        img = Image.new("RGB", img_size, color=(i * 50, i * 30, i * 20))
        img.save(root / "images" / "training" / f"img_{i:04d}.jpg")

        mask = Image.new("L", img_size, color=i % 3)
        mask.save(root / "annotations" / "training" / f"img_{i:04d}.png")

    # Create some dummy images and masks for validation split
    n_val = 3
    for i in range(n_val):
        img = Image.new("RGB", img_size, color=(100 + i * 20, 50, 50))
        img.save(root / "images" / "validation" / f"val_{i:04d}.jpg")

        mask = Image.new("L", img_size, color=(i + 1) % 3)
        mask.save(root / "annotations" / "validation" / f"val_{i:04d}.png")

    # Create labels CSV
    scenes = ["outdoor", "indoor", "nature"]
    with open(root / "labels.csv", "w") as fd:
        fd.write("stem,scene\n")
        for i in range(n_train):
            fd.write(f"img_{i:04d},{scenes[i % len(scenes)]}\n")
        for i in range(n_val):
            fd.write(f"val_{i:04d},{scenes[(i + 1) % len(scenes)]}\n")

    return root


@pytest.fixture
def multi_label_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create an ImgSegFolder with multiple label columns (e.g., species, habitat, diet)."""
    root = tmp_path / "multi_label_dataset"

    (root / "images" / "training").mkdir(parents=True)
    (root / "annotations" / "training").mkdir(parents=True)

    n_train = 4
    img_size = (64, 64)
    for i in range(n_train):
        img = Image.new("RGB", img_size, color=(i * 50, i * 30, i * 20))
        img.save(root / "images" / "training" / f"fish_{i:04d}.jpg")

        mask = Image.new("L", img_size, color=i % 3)
        mask.save(root / "annotations" / "training" / f"fish_{i:04d}.png")

    species = ["salmon", "trout", "bass", "carp"]
    habitats = ["freshwater", "saltwater", "freshwater", "freshwater"]
    diets = ["carnivore", "omnivore", "carnivore", "herbivore"]

    with open(root / "labels.csv", "w") as fd:
        fd.write("stem,species,habitat,diet\n")
        for i in range(n_train):
            fd.write(f"fish_{i:04d},{species[i]},{habitats[i]},{diets[i]}\n")

    return root


# Config tests


def test_config_default_values():
    cfg = ImgSegFolder()
    assert cfg.split == "training"
    assert cfg.labels_csv == "labels.csv"
    assert cfg.bg_label == 0


def test_config_n_examples_empty_dir(tmp_path: pathlib.Path):
    cfg = ImgSegFolder(root=tmp_path / "nonexistent")
    assert cfg.n_examples == 0


def test_config_n_examples_counts_images(img_seg_root: pathlib.Path):
    cfg_train = ImgSegFolder(root=img_seg_root, split="training")
    cfg_val = ImgSegFolder(root=img_seg_root, split="validation")

    assert cfg_train.n_examples == 5
    assert cfg_val.n_examples == 3


# Dataset tests


def test_dataset_init_smoke(img_seg_root: pathlib.Path):
    cfg = ImgSegFolder(root=img_seg_root, split="training")
    ds = ImgSegFolderDataset(cfg)
    assert len(ds) == 5


def test_dataset_init_validation_split(img_seg_root: pathlib.Path):
    cfg = ImgSegFolder(root=img_seg_root, split="validation")
    ds = ImgSegFolderDataset(cfg)
    assert len(ds) == 3


def test_dataset_getitem_returns_expected_keys(img_seg_root: pathlib.Path):
    cfg = ImgSegFolder(root=img_seg_root, split="training")
    ds = ImgSegFolderDataset(cfg)

    sample = ds[0]

    assert "data" in sample
    assert "labels" in sample
    assert "targets" in sample
    assert "index" in sample
    assert isinstance(sample["labels"], dict)
    assert isinstance(sample["targets"], dict)
    assert "scene" in sample["labels"]
    assert "scene" in sample["targets"]


def test_dataset_getitem_data_is_pil(img_seg_root: pathlib.Path):
    cfg = ImgSegFolder(root=img_seg_root, split="training")
    ds = ImgSegFolderDataset(cfg)

    sample = ds[0]
    assert isinstance(sample["data"], Image.Image)


def test_dataset_img_transform_applied(img_seg_root: pathlib.Path):
    cfg = ImgSegFolder(root=img_seg_root, split="training")

    transform_called = [False]

    def mock_transform(img):
        transform_called[0] = True
        return np.array(img)

    ds = ImgSegFolderDataset(cfg, img_transform=mock_transform)
    sample = ds[0]

    assert transform_called[0]
    assert isinstance(sample["data"], np.ndarray)


def test_dataset_mask_transform_applied(img_seg_root: pathlib.Path):
    cfg = ImgSegFolder(root=img_seg_root, split="training")

    transform_called = [False]

    def mock_mask_transform(mask):
        transform_called[0] = True
        return np.array(mask)

    ds = ImgSegFolderDataset(cfg, mask_transform=mock_mask_transform)
    sample = ds[0]

    assert transform_called[0]
    assert "patch_labels" in sample
    assert isinstance(sample["patch_labels"], np.ndarray)


def test_dataset_sample_transform_applied(img_seg_root: pathlib.Path):
    cfg = ImgSegFolder(root=img_seg_root, split="training")

    def add_custom_field(sample):
        sample["custom"] = "added"
        return sample

    ds = ImgSegFolderDataset(cfg, sample_transform=add_custom_field)
    sample = ds[0]

    assert sample["custom"] == "added"


def test_dataset_labels_are_consistent(img_seg_root: pathlib.Path):
    cfg = ImgSegFolder(root=img_seg_root, split="training")
    ds = ImgSegFolderDataset(cfg)

    label_to_target: dict[str, dict[str, int]] = {}
    for i in range(len(ds)):
        sample = ds[i]
        for col in sample["labels"]:
            label = sample["labels"][col]
            target = sample["targets"][col]

            if col not in label_to_target:
                label_to_target[col] = {}

            if label in label_to_target[col]:
                assert label_to_target[col][label] == target
            else:
                label_to_target[col][label] = target


def test_dataset_missing_images_dir_raises(tmp_path: pathlib.Path):
    root = tmp_path / "bad_dataset"
    (root / "annotations" / "training").mkdir(parents=True)

    cfg = ImgSegFolder(root=root, split="training")

    with pytest.raises(ValueError, match="Can't find path"):
        ImgSegFolderDataset(cfg)


def test_dataset_missing_annotations_dir_raises(tmp_path: pathlib.Path):
    root = tmp_path / "bad_dataset"
    (root / "images" / "training").mkdir(parents=True)

    cfg = ImgSegFolder(root=root, split="training")

    with pytest.raises(ValueError, match="Can't find path"):
        ImgSegFolderDataset(cfg)


# Multi-label tests


def test_multilabel_columns_loaded(multi_label_root: pathlib.Path):
    cfg = ImgSegFolder(root=multi_label_root, split="training")
    ds = ImgSegFolderDataset(cfg)

    sample = ds[0]

    assert "species" in sample["labels"]
    assert "habitat" in sample["labels"]
    assert "diet" in sample["labels"]

    assert sample["labels"]["species"] == "salmon"
    assert sample["labels"]["habitat"] == "freshwater"
    assert sample["labels"]["diet"] == "carnivore"


def test_multilabel_targets_computed_per_column(multi_label_root: pathlib.Path):
    cfg = ImgSegFolder(root=multi_label_root, split="training")
    ds = ImgSegFolderDataset(cfg)

    species_targets = {ds[i]["targets"]["species"] for i in range(len(ds))}
    habitat_targets = {ds[i]["targets"]["habitat"] for i in range(len(ds))}
    diet_targets = {ds[i]["targets"]["diet"] for i in range(len(ds))}

    assert species_targets == {0, 1, 2, 3}  # 4 unique species
    assert habitat_targets == {0, 1}  # 2 unique habitats
    assert diet_targets == {0, 1, 2}  # 3 unique diets


def test_multilabel_missing_labels_csv_raises(tmp_path: pathlib.Path):
    root = tmp_path / "no_labels"
    (root / "images" / "training").mkdir(parents=True)
    (root / "annotations" / "training").mkdir(parents=True)

    img = Image.new("RGB", (64, 64))
    img.save(root / "images" / "training" / "img.jpg")
    mask = Image.new("L", (64, 64))
    mask.save(root / "annotations" / "training" / "img.png")

    cfg = ImgSegFolder(root=root, split="training")

    with pytest.raises(FileNotFoundError):
        ImgSegFolderDataset(cfg)


# SegFolder integration tests (require --segfolder argument)


def test_segfolder_dataset_loads(segfolder_root: pathlib.Path):
    """Test that SegFolder dataset can be loaded."""
    cfg = ImgSegFolder(root=segfolder_root, split="training")
    ds = ImgSegFolderDataset(cfg)

    assert len(ds) > 0, "Dataset should have examples"


def test_segfolder_dataset_iteration(segfolder_root: pathlib.Path):
    """Test that we can iterate through SegFolder samples."""
    cfg = ImgSegFolder(root=segfolder_root, split="training")
    ds = ImgSegFolderDataset(cfg)

    n_samples = min(10, len(ds))
    for i in range(n_samples):
        sample = ds[i]

        assert "data" in sample
        assert "labels" in sample
        assert "targets" in sample
        assert "index" in sample
        assert isinstance(sample["data"], Image.Image)
        assert isinstance(sample["labels"], dict)
        assert isinstance(sample["targets"], dict)
        assert sample["index"] == i


def test_segfolder_validation_split(segfolder_root: pathlib.Path):
    """Test that validation split also works."""
    cfg = ImgSegFolder(root=segfolder_root, split="validation")
    ds = ImgSegFolderDataset(cfg)

    assert len(ds) > 0, "Validation split should have examples"

    sample = ds[0]
    assert "data" in sample
    assert "labels" in sample
