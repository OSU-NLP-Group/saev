import dataclasses
import logging
import os
import pathlib
import typing as tp

import beartype
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import shards

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for loading indexed activation data from disk

    Attributes:
        shards: Directory with .bin shards and a metadata.json file.
        tokens: Which kinds of tokens to use. 'special' indicates the special tokens token (if any). 'content' returns content tokens. 'all' returns both content and special tokens.
        layer: Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer.
        debug: Whether the dataloader process should log debug messages.
    """

    shards: pathlib.Path = pathlib.Path("$SAEV_SCRATCH/saev/shards/abcdefg")
    tokens: tp.Literal["special", "content", "all"] = "content"
    layer: int | tp.Literal["all"] = -2
    debug: bool = False


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    """
    Dataset of activations from disk.

    Attributes:
        cfg: Configuration set via CLI args.
        md: Activations metadata; automatically loaded from disk.
        layer_idx: Layer index into the shards if we are choosing a specific layer.
    """

    class Example(tp.TypedDict, total=False):
        """Individual example."""

        act: Float[Tensor, " d_model"]
        example_idx: int
        token_idx: int
        token_label: int

    cfg: Config
    md: shards.Metadata
    layer_idx: int
    index_map: shards.IndexMap

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not os.path.isdir(self.cfg.shards):
            raise RuntimeError(f"Activations are not saved at '{self.cfg.shards}'.")

        self.md = shards.Metadata.load(self.cfg.shards)

        # Validate shard files exist
        shard_info = shards.ShardInfo.load(self.cfg.shards)
        for shard in shard_info:
            shard_path = os.path.join(self.cfg.shards, shard.name)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Shard file not found: {shard_path}")

        # Check if labels.bin exists
        labels_path = os.path.join(self.cfg.shards, "labels.bin")
        self.labels_mmap = None
        if os.path.exists(labels_path):
            self.labels_mmap = np.memmap(
                labels_path,
                mode="r",
                dtype=np.uint8,
                shape=(self.md.n_examples, self.md.content_tokens_per_example),
            )

        self.index_map = shards.IndexMap(self.md, self.cfg.tokens, self.cfg.layer)

    def transform(
        self, act: Float[np.ndarray, " d_model"]
    ) -> Float[Tensor, " d_model"]:
        act = torch.from_numpy(act.copy())
        return act

    @property
    def d_model(self) -> int:
        """Dimension of the underlying vision transformer's embedding space."""
        return self.md.d_model

    def __getitem__(self, i: int) -> Example:
        # Add bounds checking
        idx = self.index_map.from_global(i)

        acts_fpath = self.cfg.shards / f"acts{idx.shard_idx:06}.bin"
        acts = np.memmap(
            acts_fpath, mode="c", dtype=np.float32, shape=self.md.shard_shape
        )

        # Get the activation
        act = acts[
            idx.example_idx_in_shard, idx.layer_idx_in_shard, idx.token_idx_in_shard
        ]

        result = self.Example(
            act=self.transform(act),
            example_idx=idx.example_idx,
            token_idx=idx.content_token_idx,
        )

        # Add patch label if available
        if self.labels_mmap is not None and idx.content_token_idx >= 0:
            result["token_label"] = int(
                self.labels_mmap[idx.example_idx, idx.content_token_idx]
            )

        return result

    def __len__(self) -> int:
        """
        Dataset length depends on `patches` and `layer`.
        """
        return len(self.index_map)
