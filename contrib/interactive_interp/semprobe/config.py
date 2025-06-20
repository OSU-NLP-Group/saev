import dataclasses
import os.path

import beartype

import saev.config


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Score:
    sae_ckpt: str = os.path.join(".", "checkpoints", "abcdefg", "sae.pt")
    """Path to SAE checkpoint"""

    batch_size: int = 2048
    """Batch size for SAE inference."""
    n_workers: int = 32
    """Number of dataloader workers."""

    threshold: float = 0.0
    """Threshold for feature activation."""
    top_k: int = 5
    """Number of top features to manually analyze."""

    imgs: saev.config.ImageFolderDataset = dataclasses.field(
        default_factory=saev.config.ImageFolderDataset
    )
    """Where curated examples are stored"""
    acts: saev.config.DataLoad = dataclasses.field(default_factory=saev.config.DataLoad)
    """SAE activations for the curated examples."""

    dump_to: str = os.path.join(".", "logs", "contrib", "semprobe")
    """Where to save results/visualizations."""

    include_latents: list[int] = dataclasses.field(default_factory=list)
    """Latents to manually include."""

    device: str = "cuda"
    """Hardware device."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Negatives:
    dump_to: str = os.path.join(".", "data", "contrib", "semprobe", "test")
    """Where to save negative samples."""
    imgs: saev.config.DatasetConfig = dataclasses.field(
        default_factory=saev.config.ImagenetDataset
    )
    """Where to sample images from."""
    classes: list[str] = dataclasses.field(default_factory=lambda: ["brazil", "cool"])
    """Which classes to randomly sample."""
    n_imgs: int = 20
    """Number of negative images."""
    skip: list[str] = dataclasses.field(default_factory=lambda: [])
    """Which images to skip."""

    seed: int = 42
    """Random seed."""
