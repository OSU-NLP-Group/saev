import dataclasses
import logging
import os
import pathlib
import typing

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
        patches: Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches.
        layer: Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer.
        seed: Random seed.
        debug: Whether the dataloader process should log debug messages.
    """

    shards: pathlib.Path = pathlib.Path("$SAEV_SCRATCH/saev/shards/abcdefg")
    patches: typing.Literal["cls", "image", "all"] = "image"
    layer: int | typing.Literal["all"] = -2
    seed: int = 17
    debug: bool = False


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    """
    Dataset of activations from disk.

    Attributes:
        cfg: Configuration set via CLI args.
        metadata: Activations metadata; automatically loaded from disk.
        layer_index: Layer index into the shards if we are choosing a specific layer.
    """

    class Example(typing.TypedDict, total=False):
        """Individual example."""

        act: Float[Tensor, " d_model"]
        ex_i: int
        patch_i: int
        patch_label: int

    cfg: Config
    metadata: shards.Metadata
    layer_index: int

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not os.path.isdir(self.cfg.shards):
            raise RuntimeError(f"Activations are not saved at '{self.cfg.shards}'.")

        self.metadata = shards.Metadata.load(self.cfg.shards)

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
                shape=(self.metadata.n_examples, self.metadata.patches_per_ex),
            )

        # Pick a really big number so that if you accidentally use this when you shouldn't, you get an out of bounds IndexError.
        self.layer_index = 1_000_000
        if isinstance(self.cfg.layer, int):
            err_msg = f"Non-exact matches for .layer field not supported; {self.cfg.layer} not in {self.metadata.layers}."
            assert self.cfg.layer in self.metadata.layers, err_msg
            self.layer_index = self.metadata.layers.index(self.cfg.layer)

    def transform(
        self, act: Float[np.ndarray, " d_model"]
    ) -> Float[Tensor, " d_model"]:
        act = torch.from_numpy(act.copy())
        return act

    @property
    def d_model(self) -> int:
        """Dimension of the underlying vision transformer's embedding space."""
        return self.metadata.d_model

    def __getitem__(self, i: int) -> Example:
        # Add bounds checking
        if i < 0 or i >= len(self):
            raise IndexError(
                f"Index {i} out of range for dataset of length {len(self)}"
            )

        match (self.cfg.patches, self.cfg.layer):
            case ("cls", int()):
                img_act = self.get_img_patches(i)
                # Select layer's cls token.
                act = img_act[self.layer_index, 0, :]
                result = self.Example(act=self.transform(act), ex_i=i, patch_i=-1)

                # Note: CLS tokens don't have patch labels since they're not image patches
                # patch_label is omitted for CLS tokens

                return result
            case ("image", int()):
                # Calculate which image and patch this index corresponds to
                ex_i = i // self.metadata.patches_per_ex
                patch_i = i % self.metadata.patches_per_ex

                # Calculate shard location
                ex_per_shard = (
                    self.metadata.max_patches_per_shard
                    // len(self.metadata.layers)
                    // (self.metadata.patches_per_ex + int(self.metadata.cls_token))
                )

                shard = ex_i // ex_per_shard
                img_pos_in_shard = ex_i % ex_per_shard

                acts_fpath = os.path.join(self.cfg.shards, f"acts{shard:06}.bin")
                shape = (
                    ex_per_shard,
                    len(self.metadata.layers),
                    self.metadata.patches_per_ex + int(self.metadata.cls_token),
                    self.metadata.d_model,
                )
                acts = np.memmap(acts_fpath, mode="c", dtype=np.float32, shape=shape)

                # Account for CLS token offset when accessing patches
                patch_idx_with_cls = patch_i + int(self.metadata.cls_token)

                # Get the activation
                act = acts[img_pos_in_shard, self.layer_index, patch_idx_with_cls]

                result = self.Example(
                    act=self.transform(act),
                    ex_i=ex_i,
                    patch_i=patch_i,
                )

                # Add patch label if available
                if self.labels_mmap is not None:
                    result["patch_label"] = int(self.labels_mmap[ex_i, patch_i])

                return result
            case _:
                print((self.cfg.patches, self.cfg.layer))
                typing.assert_never((self.cfg.patches, self.cfg.layer))

    def get_img_patches(
        self, i: int
    ) -> Float[np.ndarray, "n_layers all_patches d_model"]:
        ex_per_shard = (
            self.metadata.max_patches_per_shard
            // len(self.metadata.layers)
            // (self.metadata.patches_per_ex + int(self.metadata.cls_token))
        )
        shard = i // ex_per_shard
        pos = i % ex_per_shard
        acts_fpath = os.path.join(self.cfg.shards, f"acts{shard:06}.bin")
        shape = (
            ex_per_shard,
            len(self.metadata.layers),
            self.metadata.patches_per_ex + int(self.metadata.cls_token),
            self.metadata.d_model,
        )
        acts = np.memmap(acts_fpath, mode="c", dtype=np.float32, shape=shape)
        # Note that this is not yet copied!
        return acts[pos]

    def __len__(self) -> int:
        """
        Dataset length depends on `patches` and `layer`.
        """
        match (self.cfg.patches, self.cfg.layer):
            case ("cls", "all"):
                # Return a CLS token from a random image and random layer.
                return self.metadata.n_examples * len(self.metadata.layers)
            case ("cls", int()):
                # Return a CLS token from a random image and fixed layer.
                return self.metadata.n_examples
            case ("image", int()):
                # Return a patch from a random image, fixed layer, and random patch.
                return self.metadata.n_examples * (self.metadata.patches_per_ex)
            case ("image", "all"):
                # Return a patch from a random image, random layer and random patch.
                return (
                    self.metadata.n_examples
                    * len(self.metadata.layers)
                    * self.metadata.patches_per_ex
                )
            case _:
                typing.assert_never((self.cfg.patches, self.cfg.layer))
