import collections.abc
import dataclasses
import os.path
import typing

import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor


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

    def __init__(self, cfg): ...

    def __iter__(self) -> collections.abc.Iterable[Example]:
        """Yields batches shaped:
        {
            "act":      Float[Tensor, "B d_vit"],   # fp32, contiguous
            "image_i":  LongTensor[B],              # image index  [0 … n_imgs-1]
            "patch_i":  LongTensor[B],              # patch index  [0 … n_patches-1] or -1 for CLS/mean-pool
        }
        """
        yield self.Example(act=None, image_i=0, patch_i=0)

    def __len__(self) -> int: ...
