import dataclasses

import beartype

from . import bird_mae, clip, dinov2, dinov3, fake_clip, models, siglip
from .indexed import Config as IndexedConfig
from .indexed import Dataset as IndexedDataset
from .ordered import Config as OrderedConfig
from .ordered import DataLoader as OrderedDataLoader
from .shards import Metadata, PixelAgg
from .shuffled import Config as ShuffledConfig
from .shuffled import DataLoader as ShuffledDataLoader

__all__ = [
    "IndexedConfig",
    "IndexedDataset",
    "OrderedDataLoader",
    "OrderedConfig",
    "ShuffledDataLoader",
    "ShuffledConfig",
    "Metadata",
    "PixelAgg",
    "make_ordered_config",
]

models.register_family(siglip.Vit)
models.register_family(clip.Vit)
models.register_family(dinov2.Vit)
models.register_family(dinov3.Vit)
models.register_family(fake_clip.Vit)
models.register_family(bird_mae.Transformer)


@beartype.beartype
def make_ordered_config(
    shuffled_cfg: ShuffledConfig, **overrides: object
) -> OrderedConfig:
    """Create an `OrderedConfig` from a `ShuffledConfig`, with optional overrides.

    Defaults come from `shuffled_cfg` for fields present in `OrderedConfig`, and `overrides` take precedence. Unknown override fields raise `TypeError` from the `OrderedConfig` constructor, mirroring `dataclasses.replace`.
    """
    params: dict[str, object] = {}
    for f in dataclasses.fields(OrderedConfig):
        name = f.name
        if hasattr(shuffled_cfg, name):
            params[name] = getattr(shuffled_cfg, name)
    params.update(overrides)
    return OrderedConfig(**params)
