import dataclasses
import os.path
import typing

import beartype


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Activations:
    """
    Configuration for loading activation data from disk.
    """

    shard_root: str = os.path.join(".", "shards")
    """Directory with .bin shards and a metadata.json file."""
    patches: typing.Literal["cls", "image", "all"] = "patches"
    """Which kinds of patches to use."""
    layer: int | typing.Literal["all"] = "all"
    """Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer, and ``"meanpool"`` averages activations across layers."""
    clamp: float = 1e5
    """Maximum value for activations; activations will be clamped to within [-clamp, clamp]`."""
    n_random_samples: int = 2**19
    """Number of random samples used to calculate approximate dataset means at startup."""
    scale_mean: bool | str = True
    """Whether to subtract approximate dataset means from examples. If a string, manually load from the filepath."""
    scale_norm: bool | str = True
    """Whether to scale average dataset norm to sqrt(d_vit). If a string, manually load from the filepath."""
