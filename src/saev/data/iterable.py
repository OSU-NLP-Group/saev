import collections.abc
import dataclasses
import os.path
import typing

import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from . import writers


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    shard_root: str = os.path.join(".", "shards")
    """Directory with .bin shards and a metadata.json file."""
    patches: typing.Literal["cls", "patches", "meanpool"] = "patches"
    """Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'patches' indicates it will return all patches. 'meanpool' returns the mean of all image patches."""
    layer: int | typing.Literal["all", "meanpool"] = -2
    """Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer, and ``"meanpool"`` averages activations across layers."""
    clamp: float = 1e5
    """Maximum value for activations; activations will be clamped to within [-clamp, clamp]`."""
    batch_size: int = 1024 * 16
    """Batch size."""
    n_threads: int = 4
    """Number of dataloading threads."""
    seed: int = 17
    """Random seed."""


@beartype.beartype
class DataLoader:
    """
    High-throughput streaming loader.
    """

    @jaxtyped(typechecker=beartype.beartype)
    class Example(typing.TypedDict):
        """Individual example."""

        act: Float[Tensor, "batch d_vit"]
        image_i: Int[Tensor, " batch"]
        patch_i: Int[Tensor, " batch"]

    def __init__(self, cfg):
        self.cfg = cfg
        if not os.path.isdir(self.cfg.shard_root):
            raise RuntimeError(f"Activations are not saved at '{self.cfg.shard_root}'.")

        metadata_fpath = os.path.join(self.cfg.shard_root, "metadata.json")
        self.metadata = writers.Metadata.load(metadata_fpath)

    def __iter__(self) -> collections.abc.Iterable[Example]:
        """Yields batches."""
        for _ in range(len(self)):
            yield self.Example(act=None, image_i=0, patch_i=0)

    def __len__(self) -> int:
        """
        Dataset length depends on `patches` and `layer`.
        """
        match (self.cfg.patches, self.cfg.layer):
            case ("cls", "all"):
                # Return a CLS token from a random image and random layer.
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("cls", int()):
                # Return a CLS token from a random image and fixed layer.
                return self.metadata.n_imgs
            case ("cls", "meanpool"):
                # Return a CLS token from a random image and meanpool over all layers.
                return self.metadata.n_imgs
            case ("meanpool", "all"):
                # Return the meanpool of all patches from a random image and random layer.
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("meanpool", int()):
                # Return the meanpool of all patches from a random image and fixed layer.
                return self.metadata.n_imgs
            case ("meanpool", "meanpool"):
                # Return the meanpool of all patches from a random image and meanpool over all layers.
                return self.metadata.n_imgs
            case ("patches", int()):
                # Return a patch from a random image, fixed layer, and random patch.
                return self.metadata.n_imgs * (self.metadata.n_patches_per_img)
            case ("patches", "meanpool"):
                # Return a patch from a random image, meanpooled over all layers, and a random patch.
                return self.metadata.n_imgs * (self.metadata.n_patches_per_img)
            case ("patches", "all"):
                # Return a patch from a random image, random layer and random patch.
                return (
                    self.metadata.n_imgs
                    * len(self.metadata.layers)
                    * self.metadata.n_patches_per_img
                )
            case _:
                typing.assert_never((self.cfg.patches, self.cfg.layer))
