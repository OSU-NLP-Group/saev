"""
.. include:: ./protocol.md

.. include:: ./performance.md
"""

import dataclasses

import beartype

from . import clip, dinov2, dinov3, models, siglip
from .indexed import Config as IndexedConfig
from .indexed import Dataset
from .ordered import Config as OrderedConfig
from .ordered import DataLoader as OrderedDataLoader
from .shuffled import Config as ShuffledConfig
from .shuffled import DataLoader as ShuffledDataLoader
from .writers import Metadata

__all__ = [
    "IndexedConfig",
    "Dataset",
    "OrderedDataLoader",
    "OrderedConfig",
    "ShuffledDataLoader",
    "ShuffledConfig",
    "Metadata",
    "make_ordered_config",
]

models.register_family(siglip.Vit)
models.register_family(clip.Vit)
models.register_family(dinov2.Vit)
models.register_family(dinov3.Vit)


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
