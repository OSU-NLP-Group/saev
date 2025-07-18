import dataclasses
import logging
import os
import typing

import beartype
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import writers

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for loading indexed activation data from disk."""

    shard_root: str = os.path.join(".", "shards")
    """Directory with .bin shards and a metadata.json file."""
    patches: typing.Literal["cls", "image", "all"] = "image"
    """Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches."""
    layer: int | typing.Literal["all"] = -2
    """Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer."""
    seed: int = 17
    """Random seed."""
    debug: bool = False
    """Whether the dataloader process should log debug messages."""


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    """
    Dataset of activations from disk.
    """

    class Example(typing.TypedDict):
        """Individual example."""

        act: Float[Tensor, " d_vit"]
        image_i: int
        patch_i: int

    cfg: Config
    """Configuration; set via CLI args."""
    metadata: writers.Metadata
    """Activations metadata; automatically loaded from disk."""
    layer_index: int
    """Layer index into the shards if we are choosing a specific layer."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not os.path.isdir(self.cfg.shard_root):
            raise RuntimeError(f"Activations are not saved at '{self.cfg.shard_root}'.")

        self.metadata = writers.Metadata.load(self.cfg.shard_root)

        # Pick a really big number so that if you accidentally use this when you shouldn't, you get an out of bounds IndexError.
        self.layer_index = 1_000_000
        if isinstance(self.cfg.layer, int):
            err_msg = f"Non-exact matches for .layer field not supported; {self.cfg.layer} not in {self.metadata.layers}."
            assert self.cfg.layer in self.metadata.layers, err_msg
            self.layer_index = self.metadata.layers.index(self.cfg.layer)

    def transform(self, act: Float[np.ndarray, " d_vit"]) -> Float[Tensor, " d_vit"]:
        act = torch.from_numpy(act.copy())
        return act

    @property
    def d_vit(self) -> int:
        """Dimension of the underlying vision transformer's embedding space."""
        return self.metadata.d_vit

    def __getitem__(self, i: int) -> Example:
        match (self.cfg.patches, self.cfg.layer):
            case ("cls", int()):
                img_act = self.get_img_patches(i)
                # Select layer's cls token.
                act = img_act[self.layer_index, 0, :]
                return self.Example(act=self.transform(act), image_i=i, patch_i=-1)
            case ("image", int()):
                n_imgs_per_shard = (
                    self.metadata.max_patches_per_shard
                    // len(self.metadata.layers)
                    // (self.metadata.n_patches_per_img + int(self.metadata.cls_token))
                )
                n_patches_per_shard = n_imgs_per_shard * self.metadata.n_patches_per_img

                shard = i // n_patches_per_shard
                pos = i % n_patches_per_shard

                acts_fpath = os.path.join(self.cfg.shard_root, f"acts{shard:06}.bin")
                shape = (
                    n_imgs_per_shard,
                    len(self.metadata.layers),
                    self.metadata.n_patches_per_img + int(self.metadata.cls_token),
                    self.metadata.d_vit,
                )
                acts = np.memmap(acts_fpath, mode="c", dtype=np.float32, shape=shape)
                # Choose the layer and the non-CLS tokens.
                acts = acts[:, self.layer_index]

                # Choose a patch among n and the patches.
                act = acts[
                    pos // self.metadata.n_patches_per_img,
                    pos % self.metadata.n_patches_per_img,
                ]
                return self.Example(
                    act=self.transform(act),
                    # What image is this?
                    image_i=i // self.metadata.n_patches_per_img,
                    patch_i=i % self.metadata.n_patches_per_img,
                )
            case _:
                print((self.cfg.patches, self.cfg.layer))
                typing.assert_never((self.cfg.patches, self.cfg.layer))

    def get_img_patches(
        self, i: int
    ) -> Float[np.ndarray, "n_layers all_patches d_vit"]:
        n_imgs_per_shard = (
            self.metadata.max_patches_per_shard
            // len(self.metadata.layers)
            // (self.metadata.n_patches_per_img + 1)
        )
        shard = i // n_imgs_per_shard
        pos = i % n_imgs_per_shard
        acts_fpath = os.path.join(self.cfg.shard_root, f"acts{shard:06}.bin")
        shape = (
            n_imgs_per_shard,
            len(self.metadata.layers),
            self.metadata.n_patches_per_img + 1,
            self.metadata.d_vit,
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
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("cls", int()):
                # Return a CLS token from a random image and fixed layer.
                return self.metadata.n_imgs
            case ("image", int()):
                # Return a patch from a random image, fixed layer, and random patch.
                return self.metadata.n_imgs * (self.metadata.n_patches_per_img)
            case ("image", "all"):
                # Return a patch from a random image, random layer and random patch.
                return (
                    self.metadata.n_imgs
                    * len(self.metadata.layers)
                    * self.metadata.n_patches_per_img
                )
            case _:
                typing.assert_never((self.cfg.patches, self.cfg.layer))
