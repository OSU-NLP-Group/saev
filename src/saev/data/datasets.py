import abc
import dataclasses
import glob
import logging
import os
import pathlib
import typing as tp
from collections.abc import Callable

import beartype
import torch
import torchvision.datasets
from PIL import Image

logger = logging.getLogger(__name__)


@beartype.beartype
class DatasetConfig(abc.ABC):
    """Abstract base class for dataset configurations."""

    @property
    @abc.abstractmethod
    def n_examples(self) -> int:
        """Number of examples in the dataset."""

    @property
    @abc.abstractmethod
    def root(self) -> pathlib.Path:
        """Root directory path for the dataset."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Imagenet(DatasetConfig):
    """Configuration for HuggingFace Imagenet."""

    name: str = "ILSVRC/imagenet-1k"
    """Dataset name on HuggingFace. Don't need to change this.."""
    split: str = "train"
    """Dataset split. For the default ImageNet-1K dataset, can either be 'train', 'validation' or 'test'."""

    @property
    def n_examples(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires loading the dataset. If you need to reference this number very often, cache it in a local variable."""
        import datasets

        dataset = datasets.load_dataset(self.name, split=self.split)
        return len(dataset)

    @property
    def root(self) -> pathlib.Path:
        """Root directory path for the dataset."""
        return pathlib.Path(self.name)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Cifar10(DatasetConfig):
    """Configuration for HuggingFace CIFAR-10."""

    name: str = "uoft-cs/cifar10"
    """Dataset name on HuggingFace. Don't need to change this."""
    split: str = "train"
    """Dataset split. Can be 'train' or 'test'."""

    @property
    def n_examples(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires loading the dataset. If you need to reference this number very often, cache it in a local variable."""
        import datasets

        dataset = datasets.load_dataset(self.name, split=self.split)
        return len(dataset)

    @property
    def root(self) -> pathlib.Path:
        """Dummy path for the dataset."""
        return pathlib.Path(self.name)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ImgFolder(DatasetConfig):
    """Configuration for a generic image folder dataset that matches the structure used in PyTorch's [ImageFolder](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html).

    The datset must be laid out in:

    ```
    root/class1/image1.png
    root/class1/helloworld.jpg
    ...
    root/classN/123.jpeg
    root/classN/abc.webp
    ```

    If you don't have a class structure, you can add a dummy "all" folder instead of a class folder.
    """

    root: pathlib.Path = pathlib.Path("./data/split")
    """Where the class folders with images are stored. Can be a glob pattern to match multiple directories."""

    @property
    def n_examples(self) -> int:
        """Number of examples in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires walking the directory structure. If you need to reference this number very often, cache it in a local variable."""
        # Use the same image extensions as torchvision's ImageFolder
        img_extensions = tuple(torchvision.datasets.folder.IMG_EXTENSIONS)
        n = 0
        for root in self.root.parent.glob(self.root.name):
            for _, _, files in os.walk(root):
                # Only count files with valid image extensions
                n += sum(1 for f in files if f.lower().endswith(img_extensions))
        return n


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ImgSegFolder(DatasetConfig):
    root: pathlib.Path = pathlib.Path("./data/segdataset")
    """Where the class folders with images are stored."""
    split: tp.Literal["training", "validation"] = "training"
    """Data split."""
    img_label_fname: str = "sceneCategories.txt"
    """Image labels filename."""
    bg_label: int = 0
    """Background label."""

    @property
    def n_examples(self) -> int:
        """Number of examples in the dataset. Calculated on the fly by counting image files in root/images/split."""
        # Use the same image extensions as torchvision's ImageFolder
        img_extensions = tuple(torchvision.datasets.folder.IMG_EXTENSIONS)

        # Look for images in root/images/split
        img_dir = self.root / "images" / self.split
        if not img_dir.is_dir():
            return 0

        n = 0
        for _, _, files in os.walk(img_dir):
            # Only count files with valid image extensions
            n += sum(1 for f in files if f.lower().endswith(img_extensions))
        return n


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class FakeImg(DatasetConfig):
    n_examples: int = 10

    @property
    def root(self) -> pathlib.Path:
        """Root directory path for the dataset."""
        return pathlib.Path("fake")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class FakeImgSeg(DatasetConfig):
    """Tiny synthetic segmentation dataset for tests.

    Generates dummy RGB images and pixel-level segmentation masks, mimicking the behavior of real segmentation datasets like ImgSegFolder.
    """

    n_examples: int = 10
    """Number of examples."""
    content_tokens_per_example: int = 16
    """Number of content tokens per example."""
    n_classes: int = 3
    """Number of segmentation classes."""
    bg_label: int = 0
    """Which class index is considered background."""

    @property
    def root(self) -> pathlib.Path:
        """Root directory path for the dataset."""
        return pathlib.Path("fake-seg")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class BirdClef2025(DatasetConfig):
    """Configuration for BirdCLEF 2025 dataset, filtering to only bird species (Aves).

    See https://www.kaggle.com/competitions/birdclef-2025/data for more information.
    """

    root: pathlib.Path = pathlib.Path("data/birdclef-2025")
    """Root directory containing the BirdCLEF 2025 data."""
    split: tp.Literal["train_audio", "train_soundscapes", "test_soundscapes"] = (
        "train_audio"
    )
    """Which data split to use."""

    @property
    def n_examples(self) -> int:
        """Number of bird audio samples in the dataset."""
        import polars as pl

        taxonomy = pl.read_csv(self.root / "taxonomy.csv", infer_schema_length=None)
        taxonomy = taxonomy.with_columns(pl.col("primary_label").cast(pl.Utf8))
        birds = taxonomy.filter(pl.col("class_name") == "Aves")
        bird_labels = set(birds["primary_label"].to_list())

        if self.split == "train_audio":
            train = pl.read_csv(self.root / "train.csv", infer_schema_length=None)
            train = train.with_columns(pl.col("primary_label").cast(pl.Utf8))
            return len(train.filter(pl.col("primary_label").is_in(bird_labels)))
        elif self.split == "train_soundscapes":
            soundscapes_dpath = self.root / "train_soundscapes"
            return sum(1 for f in soundscapes_dpath.iterdir() if f.suffix == ".ogg")
        elif self.split == "test_soundscapes":
            soundscapes_dpath = self.root / "test_soundscapes"
            return sum(1 for f in soundscapes_dpath.iterdir() if f.suffix == ".ogg")
        else:
            tp.assert_never(self.split)


Config = (
    Imagenet | Cifar10 | ImgFolder | ImgSegFolder | FakeImg | FakeImgSeg | BirdClef2025
)


@beartype.beartype
def get_dataset(
    cfg: Config,
    *,
    data_transform: Callable,
    mask_transform: Callable | None = None,
    sample_transform: Callable | None = None,
):
    """
    Gets the dataset for the current experiment; delegates construction to dataset-specific functions.

    Args:
        cfg: Config for the dataset.
        data_tr: Transform to be applied to each 'data' key (typically the raw data).
        mask_tr: Transform to be applied to masks.
        dict_tr: Transform to be applied to the entire sample dict.
    Returns:
        A dataset that has dictionaries with `'data'`, `'index'`, `'target'`, and `'label'` keys containing examples.
    """
    # TODO: Can we reduce duplication? Or is it nice to see that there is no magic here?
    if isinstance(cfg, Imagenet):
        return ImagenetDataset(
            cfg, img_transform=data_transform, sample_transform=sample_transform
        )
    elif isinstance(cfg, Cifar10):
        return Cifar10Dataset(
            cfg, img_transform=data_transform, sample_transform=sample_transform
        )
    elif isinstance(cfg, ImgSegFolder):
        return ImgSegFolderDataset(
            cfg,
            img_transform=data_transform,
            mask_transform=mask_transform,
            sample_transform=sample_transform,
        )
    elif isinstance(cfg, ImgFolder):
        ds = [
            ImgFolderDataset(
                root, transform=data_transform, sample_transform=sample_transform
            )
            for root in glob.glob(str(cfg.root), recursive=True)
        ]
        if len(ds) == 1:
            return ds[0]
        else:
            return torch.utils.data.ConcatDataset(ds)
    elif isinstance(cfg, FakeImg):
        return FakeImgDataset(
            cfg, img_transform=data_transform, sample_transform=sample_transform
        )
    elif isinstance(cfg, FakeImgSeg):
        return FakeImgSegDataset(
            cfg,
            img_transform=data_transform,
            mask_transform=mask_transform,
            sample_transform=sample_transform,
        )
    elif isinstance(cfg, BirdClef2025):
        return BirdClef2025Dataset(
            cfg, audio_transform=data_transform, sample_transform=sample_transform
        )
    else:
        tp.assert_never(cfg)


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

        self.hf_dataset = datasets.load_dataset(cfg.name, split=cfg.split)

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
class Cifar10Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: Cifar10,
        *,
        img_transform=None,
        sample_transform: Callable | None = None,
    ):
        import datasets

        self.hf_dataset = datasets.load_dataset(cfg.name, split=cfg.split)

        self.img_transform = img_transform
        self.sample_transform = sample_transform
        self.labels = self.hf_dataset.info.features["label"].names

    def __getitem__(self, i):
        sample = self.hf_dataset[i]
        sample["index"] = i

        sample["image"] = sample.pop("img").convert("RGB")
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
class ImgFolderDataset(torchvision.datasets.ImageFolder):
    """A generic image folder dataset that matches the structure used in PyTorch's [ImageFolder](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html).

    The datset must be laid out in:

    ```
    root/class1/image1.png
    root/class1/helloworld.jpg
    ...
    root/classN/123.jpeg
    root/classN/abc.webp
    ```

    If you don't have a class structure, you can add a dummy "all" folder instead of a class folder.
    """

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
class ImgSegFolderDataset(torch.utils.data.Dataset):
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
        cfg: ImgSegFolder,
        *,
        img_transform: Callable | None = None,
        mask_transform: Callable | None = lambda x: None,
        sample_transform: Callable | None = None,
    ):
        self.logger = logging.getLogger("segfolder")
        self.cfg = cfg
        self.img_dir = os.path.join(cfg.root, "images")
        self.seg_dir = os.path.join(cfg.root, "annotations")

        self.img_transform = img_transform
        self.mask_transform = mask_transform
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

        segmentation = Image.open(sample.pop("seg_path"))

        # Apply segmentation transform to get patch labels
        if self.mask_transform is not None:
            patch_labels = self.mask_transform(segmentation)
            if patch_labels is not None:
                sample["patch_labels"] = patch_labels

        sample["index"] = index

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.samples)


class FakeImgDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: FakeImg, *, img_transform=None, sample_transform=None):
        self.n_examples = cfg.n_examples
        self.img_transform = img_transform
        self.sample_transform = sample_transform

    def __len__(self):
        return self.n_examples

    def __getitem__(self, i):
        img = Image.new("RGB", (256, 256))
        if self.img_transform is not None:
            img = self.img_transform(img)

        sample = {"image": img, "index": i, "target": 0, "label": "dummy"}
        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample


@beartype.beartype
class FakeImgSegDataset(torch.utils.data.Dataset):
    """Synthetic segmentation dataset providing pixel-level segmentation masks.

    Mimics ImgSegFolderDataset by providing:

    - image: a dummy RGB PIL image
    - segmentation: a PIL image with pixel-level class labels
    - index, target, label
    """

    def __init__(
        self,
        cfg: FakeImgSeg,
        *,
        img_transform=None,
        mask_transform=None,
        sample_transform=None,
    ):
        self.cfg = cfg
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.sample_transform = sample_transform

    def __len__(self) -> int:
        return self.cfg.n_examples

    def __getitem__(self, i: int) -> dict[str, object]:
        import numpy as np

        # Create a dummy RGB image
        img_size = 64  # Will be resized by transforms
        img = Image.new("RGB", (img_size, img_size), color=(127, 127, 127))

        # Create a deterministic segmentation mask with pixel-level labels
        # Use mode "L" for grayscale (0-255 values)
        seg_array = np.zeros((img_size, img_size), dtype=np.uint8)

        # Create a pattern that will result in different labels per patch
        # Assuming patches are created by dividing the image into a grid
        patch_grid_size = int(np.sqrt(self.cfg.content_tokens_per_example))
        patch_size = img_size // patch_grid_size

        for y in range(0, img_size, patch_size):
            for x in range(0, img_size, patch_size):
                patch_idx = (y // patch_size) * patch_grid_size + (x // patch_size)
                # Deterministic label based on patch index and image index
                label = (patch_idx + i) % self.cfg.n_classes
                seg_array[y : y + patch_size, x : x + patch_size] = label

        # Set some patches to background
        if self.cfg.bg_label < self.cfg.n_classes:
            seg_array[:patch_size, :] = self.cfg.bg_label

        segmentation = Image.fromarray(seg_array)

        if self.img_transform is not None:
            img = self.img_transform(img)

        # Apply segmentation transform to get patch labels
        patch_labels = None
        if self.mask_transform is not None:
            patch_labels = self.mask_transform(segmentation)

        sample: dict[str, object] = {
            "image": img,
            "index": i,
            "target": 0,
            "label": "dummy",
        }

        # Add patch_labels if we have them
        if patch_labels is not None:
            sample["patch_labels"] = patch_labels

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample


@beartype.beartype
def is_img_seg_dataset(data_cfg: DatasetConfig) -> bool:
    """
    Check if a dataset configuration is for an image segmentation dataset.

    Args:
        data_cfg: Dataset configuration

    Returns:
        True if this is an image segmentation dataset that should have labels.bin
    """
    return isinstance(data_cfg, (FakeImgSeg, ImgSegFolder))


@beartype.beartype
class BirdClef2025Dataset(torch.utils.data.Dataset):
    """Dataset for BirdCLEF 2025 filtered to bird species only (class_name == 'Aves')."""

    def __init__(
        self,
        cfg: BirdClef2025,
        *,
        audio_transform=None,
        mask_transform=None,
        sample_transform=None,
    ):
        import polars as pl

        self.cfg = cfg
        self.audio_transform = audio_transform
        self.sample_transform = sample_transform

        # Load taxonomy and filter to birds only
        taxonomy = pl.read_csv(cfg.root / "taxonomy.csv", infer_schema_length=None)
        taxonomy = taxonomy.with_columns(pl.col("primary_label").cast(pl.Utf8))
        birds = taxonomy.filter(pl.col("class_name") == "Aves")
        bird_labels = set(birds["primary_label"].to_list())

        # Build label -> target mapping from bird species only
        sorted_labels = sorted(bird_labels)
        self.label_to_target = {label: i for i, label in enumerate(sorted_labels)}
        self.target_to_label = {i: label for label, i in self.label_to_target.items()}

        if cfg.split == "train_audio":
            train = pl.read_csv(cfg.root / "train.csv", infer_schema_length=None)
            train = train.with_columns(pl.col("primary_label").cast(pl.Utf8))
            train_birds = train.filter(pl.col("primary_label").is_in(bird_labels))
            self.samples = [
                {"label": row["primary_label"], "filename": row["filename"]}
                for row in train_birds.iter_rows(named=True)
            ]
        elif cfg.split == "train_soundscapes":
            soundscapes_dpath = cfg.root / "train_soundscapes"
            self.samples = [
                {"label": None, "filename": f.name}
                for f in sorted(soundscapes_dpath.iterdir())
                if f.suffix == ".ogg"
            ]
        elif cfg.split == "test_soundscapes":
            soundscapes_dpath = cfg.root / "test_soundscapes"
            self.samples = [
                {"label": None, "filename": f.name}
                for f in sorted(soundscapes_dpath.iterdir())
                if f.suffix == ".ogg"
            ]
        else:
            tp.assert_never(cfg.split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        import soundfile as sf

        sample = self.samples[index]
        label = sample["label"]
        target = self.label_to_target[label] if label is not None else None

        # Build audio path based on split
        audio_fpath = self.cfg.root / self.cfg.split / sample["filename"]
        audio, sample_rate = sf.read(audio_fpath)

        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        sample = {
            "index": index,
            "target": target,
            "label": label,
            "data": audio,
            "sample_rate": sample_rate,
        }
        if self.sample_transform is not None:
            sample = self.sample_transform(sample)
        return sample

    @property
    def n_classes(self) -> int:
        """Number of bird species."""
        return len(self.label_to_target)
