"""
.. include:: ./protocol.md

.. include:: ./performance.md
"""

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
]
