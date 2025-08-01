import dataclasses
import logging
import os
import typing
from collections.abc import Callable

import beartype
import torch
import torchvision.datasets
from PIL import Image

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Imagenet:
    """Configuration for HuggingFace Imagenet."""

    name: str = "ILSVRC/imagenet-1k"
    """Dataset name on HuggingFace. Don't need to change this.."""
    split: str = "train"
    """Dataset split. For the default ImageNet-1K dataset, can either be 'train', 'validation' or 'test'."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires loading the dataset. If you need to reference this number very often, cache it in a local variable."""
        import datasets

        dataset = datasets.load_dataset(
            self.name, split=self.split, trust_remote_code=True
        )
        return len(dataset)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ImageFolder:
    """Configuration for a generic image folder dataset."""

    root: str = os.path.join(".", "data", "split")
    """Where the class folders with images are stored."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires walking the directory structure. If you need to reference this number very often, cache it in a local variable."""
        n = 0
        for _, _, files in os.walk(self.root):
            n += len(files)
        return n


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Ade20k:
    """ """

    root: str = os.path.join(".", "data", "ade20k")
    """Where the class folders with images are stored."""
    split: typing.Literal["training", "validation"] = "training"
    """Data split."""

    @property
    def n_imgs(self) -> int:
        if self.split == "validation":
            return 2000
        else:
            return 20210


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Fake:
    n_imgs: int = 10


Config = Imagenet | ImageFolder | Ade20k | Fake


@beartype.beartype
def setup(cfg: Config):
    """
    Run dataset-specific setup. These setup functions can assume they are the only job running, but they should be idempotent; they should be safe (and ideally cheap) to run multiple times in a row.
    """
    if isinstance(cfg, Imagenet):
        setup_imagenet(cfg)
    elif isinstance(cfg, ImageFolder):
        setup_imagefolder(cfg)
    elif isinstance(cfg, Ade20k):
        setup_ade20k(cfg)
    elif isinstance(cfg, Fake):
        pass
    else:
        typing.assert_never(cfg.data)


@beartype.beartype
def setup_imagenet(cfg: Imagenet):
    pass


@beartype.beartype
def setup_imagefolder(cfg: ImageFolder):
    logger.info("No dataset-specific setup for ImageFolder.")


@beartype.beartype
def setup_ade20k(cfg: Ade20k):
    # url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    pass


@beartype.beartype
def get_dataset(cfg: Config, *, img_transform):
    """
    Gets the dataset for the current experiment; delegates construction to dataset-specific functions.

    Args:
        cfg: Experiment config.
        img_transform: Image transform to be applied to each image.

    Returns:
        A dataset that has dictionaries with `'image'`, `'index'`, `'target'`, and `'label'` keys containing examples.
    """
    if isinstance(cfg, Imagenet):
        return ImagenetDataset(cfg, img_transform=img_transform)
    elif isinstance(cfg, Ade20k):
        return Ade20kDataset(cfg, img_transform=img_transform)
    elif isinstance(cfg, ImageFolder):
        return ImageFolderDataset(cfg.root, transform=img_transform)
    elif isinstance(cfg, Fake):
        return FakeDataset(cfg, img_transform=img_transform)
    else:
        typing.assert_never(cfg)


@beartype.beartype
class ImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Imagenet, *, img_transform=None):
        import datasets

        self.hf_dataset = datasets.load_dataset(
            cfg.name, split=cfg.split, trust_remote_code=True
        )

        self.img_transform = img_transform
        self.labels = self.hf_dataset.info.features["label"].names

    def __getitem__(self, i):
        example = self.hf_dataset[i]
        example["index"] = i

        example["image"] = example["image"].convert("RGB")
        if self.img_transform:
            example["image"] = self.img_transform(example["image"])
        example["target"] = example.pop("label")
        example["label"] = self.labels[example["target"]]

        return example

    def __len__(self) -> int:
        return len(self.hf_dataset)


@beartype.beartype
class ImageFolderDataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int) -> dict[str, object]:
        """
        Args:
            index: Index

        Returns:
            dict with keys 'image', 'index', 'target' and 'label'.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            "image": sample,
            "target": target,
            "label": self.classes[target],
            "index": index,
        }


@beartype.beartype
class Ade20kDataset(torch.utils.data.Dataset):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Sample:
        img_path: str
        seg_path: str
        label: str
        target: int

    samples: list[Sample]

    def __init__(
        self,
        cfg: Ade20k,
        *,
        img_transform: Callable | None = None,
        seg_transform: Callable | None = lambda x: None,
    ):
        self.logger = logging.getLogger("ade20k")
        self.cfg = cfg
        self.img_dir = os.path.join(cfg.root, "images")
        self.seg_dir = os.path.join(cfg.root, "annotations")
        self.img_transform = img_transform
        self.seg_transform = seg_transform

        # Check that we have the right path.
        for subdir in ("images", "annotations"):
            if not os.path.isdir(os.path.join(cfg.root, subdir)):
                # Something is missing.
                if os.path.realpath(cfg.root).endswith(subdir):
                    self.logger.warning(
                        "The ADE20K root should contain 'images/' and 'annotations/' directories."
                    )
                raise ValueError(f"Can't find path '{os.path.join(cfg.root, subdir)}'.")

        _, split_mapping = torchvision.datasets.folder.find_classes(self.img_dir)
        split_lookup: dict[int, str] = {
            value: key for key, value in split_mapping.items()
        }
        self.loader = torchvision.datasets.folder.default_loader

        assert cfg.split in set(split_lookup.values())

        # Load all the image paths.
        imgs: list[str] = [
            path
            for path, s in torchvision.datasets.folder.make_dataset(
                self.img_dir,
                split_mapping,
                extensions=torchvision.datasets.folder.IMG_EXTENSIONS,
            )
            if split_lookup[s] == cfg.split
        ]

        segs: list[str] = [
            path
            for path, s in torchvision.datasets.folder.make_dataset(
                self.seg_dir,
                split_mapping,
                extensions=torchvision.datasets.folder.IMG_EXTENSIONS,
            )
            if split_lookup[s] == cfg.split
        ]

        # Load all the targets, classes and mappings
        with open(os.path.join(cfg.root, "sceneCategories.txt")) as fd:
            img_labels: list[str] = [line.split()[1] for line in fd.readlines()]

        label_set = sorted(set(img_labels))
        label_to_idx = {label: i for i, label in enumerate(label_set)}

        self.samples = [
            self.Sample(img_path, seg_path, label, label_to_idx[label])
            for img_path, seg_path, label in zip(imgs, segs, img_labels)
        ]

    def __getitem__(self, index: int) -> dict[str, object]:
        # Convert to dict.
        sample = dataclasses.asdict(self.samples[index])

        sample["image"] = self.loader(sample.pop("img_path"))
        if self.img_transform is not None:
            image = self.img_transform(sample.pop("image"))
            if image is not None:
                sample["image"] = image

        sample["segmentation"] = Image.open(sample.pop("seg_path")).convert("L")
        if self.seg_transform is not None:
            segmentation = self.seg_transform(sample.pop("segmentation"))
            if segmentation is not None:
                sample["segmentation"] = segmentation

        sample["index"] = index

        return sample

    def __len__(self) -> int:
        return len(self.samples)


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Fake, *, img_transform=None):
        self.n_imgs = cfg.n_imgs
        self.img_transform = img_transform

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, i):
        img = Image.new("RGB", (256, 256))
        return {
            "image": self.img_transform(img),
            "index": i,
            "target": 0,
            "label": "dummy",
        }
