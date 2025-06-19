"""
All configs for all saev jobs.

## Import Times

This module should be very fast to import so that `python main.py --help` is fast.
This means that the top-level imports should not include big packages like numpy, torch, etc.
For example, `TreeOfLife.n_imgs` imports numpy when it's needed, rather than importing it at the top level.

Also contains code for expanding configs with lists into lists of configs (grid search).
Might be expanded in the future to support pseudo-random sampling from distributions to support random hyperparameter search, as in [this file](https://github.com/samuelstevens/sax/blob/main/sax/sweep.py).
"""

import dataclasses
import os
import typing

import beartype


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Visuals:
    """Configuration for generating visuals from trained SAEs."""

    ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    # data: DataLoad = dataclasses.field(default_factory=DataLoad)
    """Data configuration."""
    # images: DatasetConfig = dataclasses.field(default_factory=ImagenetDataset)
    """Which images to use."""
    top_k: int = 128
    """How many images per SAE feature to store."""
    n_workers: int = 16
    """Number of dataloader workers."""
    topk_batch_size: int = 1024 * 16
    """Number of examples to apply top-k op to."""
    sae_batch_size: int = 1024 * 16
    """Batch size for SAE inference."""
    epsilon: float = 1e-9
    """Value to add to avoid log(0)."""
    sort_by: typing.Literal["cls", "img", "patch"] = "patch"
    """How to find the top k images. 'cls' picks images where the SAE latents of the ViT's [CLS] token are maximized without any patch highligting. 'img' picks images that maximize the sum of an SAE latent over all patches in the image, highlighting the patches. 'patch' pickes images that maximize an SAE latent over all patches (not summed), highlighting the patches and only showing unique images."""
    device: str = "cuda"
    """Which accelerator to use."""
    dump_to: str = os.path.join(".", "data")
    """Where to save data."""
    log_freq_range: tuple[float, float] = (-6.0, -2.0)
    """Log10 frequency range for which to save images."""
    log_value_range: tuple[float, float] = (-1.0, 1.0)
    """Log10 frequency range for which to save images."""
    include_latents: list[int] = dataclasses.field(default_factory=list)
    """Latents to always include, no matter what."""
    n_distributions: int = 25
    """Number of features to save distributions for."""
    percentile: int = 99
    """Percentile to estimate for outlier detection."""
    n_latents: int = 400
    """Maximum number of latents to save images for."""
    seed: int = 42
    """Random seed."""

    @property
    def root(self) -> str:
        return os.path.join(self.dump_to, f"sort_by_{self.sort_by}")

    @property
    def top_values_fpath(self) -> str:
        return os.path.join(self.root, "top_values.pt")

    @property
    def top_img_i_fpath(self) -> str:
        return os.path.join(self.root, "top_img_i.pt")

    @property
    def top_patch_i_fpath(self) -> str:
        return os.path.join(self.root, "top_patch_i.pt")

    @property
    def mean_values_fpath(self) -> str:
        return os.path.join(self.root, "mean_values.pt")

    @property
    def sparsity_fpath(self) -> str:
        return os.path.join(self.root, "sparsity.pt")

    @property
    def distributions_fpath(self) -> str:
        return os.path.join(self.root, "distributions.pt")

    @property
    def percentiles_fpath(self) -> str:
        return os.path.join(self.root, f"percentiles_p{self.percentile}.pt")
