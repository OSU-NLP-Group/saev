import abc
import dataclasses
import logging
import os.path
import typing as tp

import beartype
import polars as pl
import torch.utils.data

import saev.data.datasets


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Butterflies:
    root: str = os.path.join("data", "butterflies")
    """Where you stored the segmentation dataset."""


Config = Butterflies


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
        raise NotImplementedError()


@beartype.beartype
def get_dataset(cfg: Config, *, img_tr, seg_tr, sample_tr=None) -> MetadataDataset:
    """
    Gets the dataset for the current experiment; delegates construction to dataset-specific functions.

    Args:
        cfg: Experiment config.
        img_tr: Image transform to be applied to each image.
        seg_tr: Transform to be applied to each segmentation.
        sample_tr: Transform to be applied to each sample dict.
    Returns:
        A dataset that has dictionaries with 'image', 'index', 'target' and 'label' keys (and maybe some extras), containing examples.
    """
    if isinstance(cfg, Butterflies):
        return ButterfliesDataset(
            cfg, img_tr=img_tr, seg_tr=seg_tr, sample_tr=sample_tr
        )
    else:
        tp.assert_never(cfg)


@beartype.beartype
class ButterfliesDataset(MetadataDataset):
    dead_cols = [
        "file_url",
        "zenodo_name",
        "zenodo_link",
        "X",
        "Sequence",
        "Sample_accession",
        "Collected_by",
        "Other_ID",
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
        self.seg_cfg = saev.data.datasets.SegFolder(
            root=cfg.root,
            split="training",
            img_label_fname="image_labels.txt",
            bg_label=0,
        )

        self.ds = saev.data.datasets.SegFolderDataset(self.seg_cfg)

        self.metadata = (
            pl
            .read_csv(
                os.path.join(self.cfg.root, "Heliconius_img_master.csv"),
                infer_schema_length=None,
            )
            .drop(self.dead_cols, strict=True)
            .drop((self.ds[0].keys()), strict=False)
        )

        # Build index -> metadata row mapping using filenames
        # Create a dict from Image_name to row index in metadata
        image_name_to_meta_idx = {
            row["Image_name"]: i
            for i, row in enumerate(self.metadata.iter_rows(named=True))
        }

        # Map dataset index to metadata index using ds.samples
        self.index_to_meta = []
        for sample in self.ds.samples:
            # Extract just the filename from the full path
            img_fname = os.path.basename(sample.img_path)
            meta_idx = image_name_to_meta_idx.get(img_fname)
            if meta_idx is None:
                raise ValueError(f"No metadata found for image: {img_fname}")
            self.index_to_meta.append(meta_idx)

    def get_metadata(self, index: int) -> dict:
        meta_idx = self.index_to_meta[index]
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
