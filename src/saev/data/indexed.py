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
    """Configuration for loading indexed activation data from disk

    Attributes:
        shard_root: Directory with .bin shards and a metadata.json file.
        patches: Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches.
        layer: Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer.
        seed: Random seed.
        debug: Whether the dataloader process should log debug messages.
    """

    shard_root: str = os.path.join(".", "shards")
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

        act: Float[Tensor, " d_vit"]
        image_i: int
        patch_i: int
        patch_label: int

    cfg: Config
    metadata: writers.Metadata
    layer_index: int

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not os.path.isdir(self.cfg.shard_root):
            raise RuntimeError(f"Activations are not saved at '{self.cfg.shard_root}'.")

        self.metadata = writers.Metadata.load(self.cfg.shard_root)

        # Validate shard files exist
        shard_info = writers.ShardInfo.load(self.cfg.shard_root)
        for shard in shard_info:
            shard_path = os.path.join(self.cfg.shard_root, shard.name)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Shard file not found: {shard_path}")

        # Check if labels.bin exists
        labels_path = os.path.join(self.cfg.shard_root, "labels.bin")
        self.labels_mmap = None
        if os.path.exists(labels_path):
            self.labels_mmap = np.memmap(
                labels_path,
                mode="r",
                dtype=np.uint8,
                shape=(self.metadata.n_imgs, self.metadata.n_patches_per_img),
            )

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
                result = self.Example(act=self.transform(act), image_i=i, patch_i=-1)

                # Note: CLS tokens don't have patch labels since they're not image patches
                # patch_label is omitted for CLS tokens

                return result
            case ("image", int()):
                # Calculate which image and patch this index corresponds to
                image_i = i // self.metadata.n_patches_per_img
                patch_i = i % self.metadata.n_patches_per_img

                # Calculate shard location
                n_imgs_per_shard = (
                    self.metadata.max_patches_per_shard
                    // len(self.metadata.layers)
                    // (self.metadata.n_patches_per_img + int(self.metadata.cls_token))
                )

                shard = image_i // n_imgs_per_shard
                img_pos_in_shard = image_i % n_imgs_per_shard

                acts_fpath = os.path.join(self.cfg.shard_root, f"acts{shard:06}.bin")
                shape = (
                    n_imgs_per_shard,
                    len(self.metadata.layers),
                    self.metadata.n_patches_per_img + int(self.metadata.cls_token),
                    self.metadata.d_vit,
                )
                acts = np.memmap(acts_fpath, mode="c", dtype=np.float32, shape=shape)

                # Account for CLS token offset when accessing patches
                patch_idx_with_cls = patch_i + int(self.metadata.cls_token)

                # Get the activation
                act = acts[img_pos_in_shard, self.layer_index, patch_idx_with_cls]

                result = self.Example(
                    act=self.transform(act),
                    image_i=image_i,
                    patch_i=patch_i,
                )

                # Add patch label if available
                if self.labels_mmap is not None:
                    result["patch_label"] = int(self.labels_mmap[image_i, patch_i])

                return result
            case _:
                print((self.cfg.patches, self.cfg.layer))
                typing.assert_never((self.cfg.patches, self.cfg.layer))

    def get_img_patches(
        self, i: int
    ) -> Float[np.ndarray, "n_layers all_patches d_vit"]:
        n_imgs_per_shard = (
            self.metadata.max_patches_per_shard
            // len(self.metadata.layers)
            // (self.metadata.n_patches_per_img + int(self.metadata.cls_token))
        )
        shard = i // n_imgs_per_shard
        pos = i % n_imgs_per_shard
        acts_fpath = os.path.join(self.cfg.shard_root, f"acts{shard:06}.bin")
        shape = (
            n_imgs_per_shard,
            len(self.metadata.layers),
            self.metadata.n_patches_per_img + int(self.metadata.cls_token),
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
