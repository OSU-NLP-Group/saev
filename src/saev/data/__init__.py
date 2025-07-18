"""
.. include:: ./protocol.md

.. include:: ./performance.md
"""

from .indexed import Config as IndexedConfig
from .indexed import Dataset
from .iterable import Config as IterableConfig
from .iterable import DataLoader
from .writers import Metadata

__all__ = ["Dataset", "DataLoader", "IndexedConfig", "IterableConfig", "Metadata"]
