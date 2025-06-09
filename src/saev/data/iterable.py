import collections.abc
import dataclasses
import typing

import beartype
from jaxtyping import Float
from torch import Tensor


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    pass


@beartype.beartype
class DataLoader:
    """
    High-throughput streaming loader.

    Required constructor signature
    -------------------------------
    DataLoader(
        cfg: config.DataLoad,      # same cfg used by reference Dataset
        batch_size: int,
        *,                         # keyword-only
        workers: int = 8,          # threads or processes
        seed: int | None = None,   # for reproducible shuffling
    )

    Required public API
    -------------------
    __iter__(self) -> Iterator[dict[str, torch.Tensor | torch.LongTensor]]
    __len__(self) -> int   # optional but convenient; not used below
    """

    class Example(typing.TypedDict):
        """Individual example."""

        act: Float[Tensor, " d_vit"]
        image_i: int
        patch_i: int

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
