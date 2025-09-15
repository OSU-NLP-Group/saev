import dataclasses
import glob
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
    """Where the class folders with images are stored. Can be a glob pattern to match multiple directories."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires walking the directory structure. If you need to reference this number very often, cache it in a local variable."""
        # Use the same image extensions as torchvision's ImageFolder
        img_extensions = tuple(torchvision.datasets.folder.IMG_EXTENSIONS)
        n = 0
        for root in glob.glob(self.root, recursive=True):
            for _, _, files in os.walk(root):
                # Only count files with valid image extensions
                n += sum(1 for f in files if f.lower().endswith(img_extensions))
        return n


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SegFolder:
    root: str = os.path.join(".", "data", "segdataset")
    """Where the class folders with images are stored."""
    split: typing.Literal["training", "validation"] = "training"
    """Data split."""
    img_label_fname: str = "sceneCategories.txt"
    """Image labels filename."""
    bg_label: int = 0
    """Background label."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly by counting image files in root/images/split."""
        # Use the same image extensions as torchvision's ImageFolder
        img_extensions = tuple(torchvision.datasets.folder.IMG_EXTENSIONS)

        # Look for images in root/images/split
        img_dir = os.path.join(self.root, "images", self.split)
        if not os.path.isdir(img_dir):
            return 0

        n = 0
        for _, _, files in os.walk(img_dir):
            # Only count files with valid image extensions
            n += sum(1 for f in files if f.lower().endswith(img_extensions))
        return n


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Fake:
    n_imgs: int = 10


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class FakeSeg:
    """Tiny synthetic segmentation dataset for tests.

    Generates dummy RGB images and per-image patch-level labels directly,
    avoiding any dependency on real segmentation masks or transforms.
    """

    n_imgs: int = 10
    """Number of images."""
    n_patches_per_img: int = 16
    """Number of patches per image (used to size the patch_labels)."""
    n_classes: int = 3
    """Number of segmentation classes."""
    bg_label: int = 0
    """Which class index is considered background."""


Config = Imagenet | ImageFolder | SegFolder | Fake | FakeSeg


@beartype.beartype
def get_dataset(cfg: Config, *, img_transform, sample_transform=None):
    """
    Gets the dataset for the current experiment; delegates construction to dataset-specific functions.

    Args:
        cfg: Experiment config.
        img_transform: Image transform to be applied to each image.
        sample_transform: Transform to be applied to each sample dict.
    Returns:
        A dataset that has dictionaries with `'image'`, `'index'`, `'target'`, and `'label'` keys containing examples.
    """
    # TODO: Can we reduce duplication? Or is it nice to see that there is no magic here?
    if isinstance(cfg, Imagenet):
        return ImagenetDataset(
            cfg, img_transform=img_transform, sample_transform=sample_transform
        )
    elif isinstance(cfg, SegFolder):
        return SegFolderDataset(
            cfg, img_transform=img_transform, sample_transform=sample_transform
        )
    elif isinstance(cfg, ImageFolder):
        ds = [
            ImageFolderDataset(
                root, transform=img_transform, sample_transform=sample_transform
            )
            for root in glob.glob(cfg.root, recursive=True)
        ]
        if len(ds) == 1:
            return ds[0]
        else:
            return torch.utils.data.ConcatDataset(ds)
    elif isinstance(cfg, Fake):
        return FakeDataset(
            cfg, img_transform=img_transform, sample_transform=sample_transform
        )
    elif isinstance(cfg, FakeSeg):
        return FakeSegDataset(
            cfg, img_transform=img_transform, sample_transform=sample_transform
        )
    else:
        typing.assert_never(cfg)


@beartype.beartype
class ImagenetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: Imagenet,
        *,
        img_transform=None,
        sample_transform: Callable | None = None,
    ):
        import datasets

        self.hf_dataset = datasets.load_dataset(
            cfg.name, split=cfg.split, trust_remote_code=True
        )

        self.img_transform = img_transform
        self.sample_transform = sample_transform
        self.labels = self.hf_dataset.info.features["label"].names

    def __getitem__(self, i):
        sample = self.hf_dataset[i]
        sample["index"] = i

        sample["image"] = sample["image"].convert("RGB")
        if self.img_transform:
            sample["image"] = self.img_transform(sample["image"])
        sample["target"] = sample.pop("label")
        sample["label"] = self.labels[sample["target"]]

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.hf_dataset)


@beartype.beartype
class ImageFolderDataset(torchvision.datasets.ImageFolder):
    def __init__(self, *args, sample_transform: Callable | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_transform = sample_transform

    def __getitem__(self, index: int) -> dict[str, object]:
        """
        Args:
            index: Index

        Returns:
            dict with keys 'image', 'index', 'target' and 'label'.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        sample = {
            "image": image,
            "target": target,
            "label": self.classes[target],
            "index": index,
        }

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample


def _stem(fpath: str) -> str:
    fname = os.path.basename(fpath)
    stem, ext = os.path.splitext(fname)
    return stem


@beartype.beartype
class SegFolderDataset(torch.utils.data.Dataset):
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
        cfg: SegFolder,
        *,
        img_transform: Callable | None = None,
        seg_transform: Callable | None = lambda x: None,
        sample_transform: Callable | None = None,
    ):
        self.logger = logging.getLogger("segfolder")
        self.cfg = cfg
        self.img_dir = os.path.join(cfg.root, "images")
        self.seg_dir = os.path.join(cfg.root, "annotations")

        self.img_transform = img_transform
        self.seg_transform = seg_transform
        self.sample_transform = sample_transform

        # Check that we have the right path.
        for subdir in ("images", "annotations"):
            if not os.path.isdir(os.path.join(cfg.root, subdir)):
                # Something is missing.
                if os.path.realpath(cfg.root).endswith(subdir):
                    self.logger.warning(
                        "The SegFolder root should contain 'images/' and 'annotations/' directories."
                    )
                raise ValueError(f"Can't find path '{os.path.join(cfg.root, subdir)}'.")

        _, split_mapping = torchvision.datasets.folder.find_classes(self.img_dir)
        split_lookup: dict[int, str] = {
            value: key for key, value in split_mapping.items()
        }
        self.loader = torchvision.datasets.folder.default_loader

        assert cfg.split in set(split_lookup.values())

        # Load all the image paths.
        base2seg: dict[str, str] = {
            _stem(seg_path): seg_path
            for seg_path, split in torchvision.datasets.folder.make_dataset(
                self.seg_dir,
                split_mapping,
                extensions=torchvision.datasets.folder.IMG_EXTENSIONS,
            )
            if split_lookup[split] == cfg.split
        }

        imgs: list[str] = [
            path
            for path, s in torchvision.datasets.folder.make_dataset(
                self.img_dir,
                split_mapping,
                extensions=torchvision.datasets.folder.IMG_EXTENSIONS,
            )
            if split_lookup[s] == cfg.split
        ]

        # Load all the targets, classes and mappings
        img_labels: dict[str, str] = {}
        with open(os.path.join(cfg.root, cfg.img_label_fname)) as fd:
            for line in fd.readlines():
                stem, _, label = line.rpartition(" ")
                img_labels[stem] = label

        label_set = sorted(set(img_labels.values()))
        label_to_idx = {label: i for i, label in enumerate(label_set)}

        self.samples = [
            self.Sample(
                img_path,
                base2seg[_stem(img_path)],
                img_labels[_stem(img_path)],
                label_to_idx[label],
            )
            for img_path in (imgs)
        ]

    def __getitem__(self, index: int) -> dict[str, object]:
        # Convert to dict.
        sample = dataclasses.asdict(self.samples[index])

        sample["image"] = self.loader(sample.pop("img_path"))
        if self.img_transform is not None:
            image = self.img_transform(sample.pop("image"))
            if image is not None:
                sample["image"] = image

        sample["segmentation"] = Image.open(sample.pop("seg_path"))
        if self.seg_transform is not None:
            segmentation = self.seg_transform(sample.pop("segmentation"))
            if segmentation is not None:
                sample["segmentation"] = segmentation

        sample["index"] = index

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.samples)


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Fake, *, img_transform=None, sample_transform=None):
        self.n_imgs = cfg.n_imgs
        self.img_transform = img_transform
        self.sample_transform = sample_transform

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, i):
        img = Image.new("RGB", (256, 256))
        if self.img_transform is not None:
            img = self.img_transform(img)

        sample = {"image": img, "index": i, "target": 0, "label": "dummy"}
        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample


@beartype.beartype
class FakeSegDataset(torch.utils.data.Dataset):
    """Synthetic segmentation dataset providing patch-level labels.

    Each example includes:
      - image: a dummy RGB PIL image
      - patch_labels: LongTensor[n_patches_per_img] with deterministic values
      - index, target, label
    """

    def __init__(
        self,
        cfg: FakeSeg,
        *,
        img_transform=None,
        sample_transform=None,
    ):
        self.cfg = cfg

        self.img_transform = img_transform
        self.sample_transform = sample_transform

    def __len__(self) -> int:
        return self.cfg.n_imgs

    def __getitem__(self, i: int) -> dict[str, object]:
        # Create a dummy RGB image; writers' label generation does not depend on pixels
        img = Image.new("RGB", (64, 64), color=(127, 127, 127))
        if self.img_transform is not None:
            img = self.img_transform(img)

        # Deterministic patch labels cycling over classes; ensure some background present
        import torch

        labels = torch.tensor(
            [
                (j + i) % max(1, self.cfg.n_classes)
                for j in range(self.cfg.n_patches_per_img)
            ],
            dtype=torch.long,
        )
        # Force a few background patches for variety
        if self.cfg.bg_label < self.cfg.n_classes:
            labels[: max(1, self.cfg.n_patches_per_img // 8)] = self.cfg.bg_label

        sample: dict[str, object] = {
            "image": img,
            "index": i,
            "target": 0,
            "label": "dummy",
            # Direct patch-level labels for tests; shape [n_patches]
            "patch_labels": labels,
        }

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample
