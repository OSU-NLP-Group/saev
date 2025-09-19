import abc
import dataclasses
import logging
import pathlib
import typing as tp

import beartype
import polars as pl
import torch.utils.data
import torchvision.datasets

import saev.data.datasets


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Butterflies:
    root: pathlib.Path = pathlib.Path("./data/butterflies")
    """Where you stored the segmentation dataset."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class FishVista:
    pass


Config = Butterflies | FishVista


@beartype.beartype
class MetadataDataset(abc.ABC, torch.utils.data.Dataset):
    """Abstract base class for datasets that provide metadata without loading images."""

    @abc.abstractmethod
    def get_metadata(self, index: int) -> dict:
        """
        Get metadata for image at index without loading the image.

        Returns dict with at minimum:
        - label: str
        - target: int
        - Any additional dataset-specific fields
        """
        pass


@beartype.beartype
def get_dataset(cfg: Config, *, img_tr, seg_tr, sample_tr=None):
    """
    Gets the dataset for the current experiment; delegates construction to dataset-specific functions.

    Args:
        cfg: Experiment config.
        img_transform: Image transform to be applied to each image.
        sample_transform: Transform to be applied to each sample dict.
    Returns:
        A dataset that has dictionaries with 'image', 'index', 'target' and 'label' keys (and maybe some extras), containing examples.
    """
    if isinstance(cfg, Butterflies):
        return saev.data.datasets.SegFolderDataset(
            cfg, img_tr=img_tr, sample_tr=sample_tr
        )
    elif isinstance(cfg, FishVista):
        return FishVistaDataset(cfg, img_tr=img_tr, sample_tr=sample_tr)
    else:
        tp.assert_never(cfg)


@beartype.beartype
class ButterfliesDataset(MetadataDataset):
    dead_cols = [
        "file_url",
        "zenodo_name",
        "zenodo_link",
        "Image_name",
        "X",
        "Sequence",
        "Sample_accession",
        "Collected_by",
        "Other_ID",
        "Dataset",
        "Date",
        "Store",
        "Brood",
        "Death_Date",
        "file_type",
        "record_number",
    ]

    def __init__(self, cfg: Butterflies, *, img_tr=None, seg_tr=None, sample_tr=None):
        self.logger = logging.getLogger("bfly-ds")
        self.cfg = cfg

        self.ds = saev.data.datasets.SegFolderDataset(cfg)

    def get_metadata(self, index: int) -> dict:
        meta_idx = self.meta_indices[index]
        return self.metadata.row(meta_idx, named=True)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> dict[str, object]:
        """
        Args:
            index: Index

        Returns:
            dict with keys 'image', 'index', 'target' and 'label'.
        """
        sample = self.ds[index]
        sample.update(**self.get_metadata(index))
        return sample


class FishVistaDataset(MetadataDataset):
    def __init__(self, cfg: FishVista, *, img_transform=None, sample_transform=None):
        pass

    def get_metadata(self, index: int) -> dict:
        raise NotImplementedError("FishVistaDataset.get_metadata not yet implemented")
