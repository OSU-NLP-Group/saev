"""
.. include:: ./protocol.md

.. include:: ./performance.md
"""

from .indexed import Config as IndexedConfig
from .indexed import Dataset
from .shuffled import Config as IterableConfig
from .shuffled import DataLoader
from .writers import Metadata

__all__ = ["Dataset", "DataLoader", "IndexedConfig", "IterableConfig", "Metadata"]
