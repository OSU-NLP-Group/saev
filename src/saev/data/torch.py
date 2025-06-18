import dataclasses
import logging
import math
import os
import typing

import beartype
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from saev import helpers

from . import writers

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for loading activation data from disk.
    """

    shard_root: str = os.path.join(".", "shards")
    """Directory with .bin shards and a metadata.json file."""
    patches: typing.Literal["cls", "patches", "all"] = "patches"
    """Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'patches' indicates it will return img patches. 'all' is both [CLS] and image patches."""
    layer: int | typing.Literal["all"] = -2
    """Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer."""
    clamp: float = 1e5
    """Maximum value for activations; activations will be clamped to within [-clamp, clamp]`."""
    n_random_samples: int = 2**19
    """Number of random samples used to calculate approximate dataset means at startup."""
    scale_mean: bool | str = True
    """Whether to subtract approximate dataset means from examples. If a string, manually load from the filepath."""
    scale_norm: bool | str = True
    """Whether to scale average dataset norm to sqrt(d_vit). If a string, manually load from the filepath."""


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
    scalar: float
    """Normalizing scalar such that ||x / scalar ||_2 ~= sqrt(d_vit)."""
    act_mean: Float[Tensor, " d_vit"]
    """Mean activation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not os.path.isdir(self.cfg.shard_root):
            raise RuntimeError(f"Activations are not saved at '{self.cfg.shard_root}'.")

        metadata_fpath = os.path.join(self.cfg.shard_root, "metadata.json")
        self.metadata = writers.Metadata.load(metadata_fpath)

        # Pick a really big number so that if you accidentally use this when you shouldn't, you get an out of bounds IndexError.
        self.layer_index = 1_000_000
        if isinstance(self.cfg.layer, int):
            err_msg = f"Non-exact matches for .layer field not supported; {self.cfg.layer} not in {self.metadata.layers}."
            assert self.cfg.layer in self.metadata.layers, err_msg
            self.layer_index = self.metadata.layers.index(self.cfg.layer)

        # Premptively set these values so that preprocess() doesn't freak out.
        self.scalar = 1.0
        self.act_mean = torch.zeros(self.d_vit)

        # If either of these are true, we must do this work.
        if self.cfg.scale_mean is True or self.cfg.scale_norm is True:
            # Load a random subset of samples to calculate the mean activation and mean L2 norm.
            perm = np.random.default_rng(seed=42).permutation(len(self))
            perm = perm[: cfg.n_random_samples]

            samples = [
                self[p.item()]
                for p in helpers.progress(
                    perm, every=25_000, desc="examples to calc means"
                )
            ]
            samples = torch.stack([sample["act"] for sample in samples])
            if samples.abs().max() > 1e3:
                raise ValueError(
                    f"You found an abnormally large activation {samples.abs().max().item():.5f} that will mess up your L2 mean."
                )

            # Activation mean
            if self.cfg.scale_mean:
                self.act_mean = samples.mean(axis=0)
                if (self.act_mean > 1e3).any():
                    raise ValueError(
                        "You found an abnormally large activation that is messing up your activation mean."
                    )

            # Norm
            if self.cfg.scale_norm:
                l2_mean = torch.linalg.norm(samples - self.act_mean, axis=1).mean()
                if l2_mean > 1e3:
                    raise ValueError(
                        "You found an abnormally large activation that is messing up your L2 mean."
                    )

                self.scalar = l2_mean / math.sqrt(self.d_vit)
        elif isinstance(self.cfg.scale_mean, str):
            # Load mean activations from disk
            self.act_mean = torch.load(self.cfg.scale_mean)
        elif isinstance(self.cfg.scale_norm, str):
            # Load scalar normalization from disk
            self.scalar = torch.load(self.cfg.scale_norm).item()

    def transform(self, act: Float[np.ndarray, " d_vit"]) -> Float[Tensor, " d_vit"]:
        """
        Apply a scalar normalization so the mean squared L2 norm is same as d_vit. This is from 'Scaling Monosemanticity':

        > As a preprocessing step we apply a scalar normalization to the model activations so their average squared L2 norm is the residual stream dimension

        So we divide by self.scalar which is the datasets (approximate) L2 mean before normalization divided by sqrt(d_vit).
        """
        act = torch.from_numpy(act.copy())
        act = act.clamp(-self.cfg.clamp, self.cfg.clamp)
        return (act - self.act_mean) / self.scalar

    @property
    def d_vit(self) -> int:
        """Dimension of the underlying vision transformer's embedding space."""
        return self.metadata.d_vit

    @jaxtyped(typechecker=beartype.beartype)
    def __getitem__(self, i: int) -> Example:
        match (self.cfg.patches, self.cfg.layer):
            case ("cls", int()):
                img_act = self.get_img_patches(i)
                # Select layer's cls token.
                act = img_act[self.layer_index, 0, :]
                return self.Example(act=self.transform(act), image_i=i, patch_i=-1)
            case ("cls", "meanpool"):
                img_act = self.get_img_patches(i)
                # Select cls tokens from across all layers
                cls_act = img_act[:, 0, :]
                # Meanpool over the layers
                act = cls_act.mean(axis=0)
                return self.Example(act=self.transform(act), image_i=i, patch_i=-1)
            case ("meanpool", int()):
                img_act = self.get_img_patches(i)
                # Select layer's patches.
                layer_act = img_act[self.layer_index, 1:, :]
                # Meanpool over the patches
                act = layer_act.mean(axis=0)
                return self.Example(act=self.transform(act), image_i=i, patch_i=-1)
            case ("meanpool", "meanpool"):
                img_act = self.get_img_patches(i)
                # Select all layer's patches.
                act = img_act[:, 1:, :]
                # Meanpool over the layers and patches
                act = act.mean(axis=(0, 1))
                return self.Example(act=self.transform(act), image_i=i, patch_i=-1)
            case ("patches", int()):
                n_imgs_per_shard = (
                    self.metadata.max_patches_per_shard
                    // len(self.metadata.layers)
                    // (self.metadata.n_patches_per_img + 1)
                )
                n_examples_per_shard = (
                    n_imgs_per_shard * self.metadata.n_patches_per_img
                )

                shard = i // n_examples_per_shard
                pos = i % n_examples_per_shard

                acts_fpath = os.path.join(self.cfg.shard_root, f"acts{shard:06}.bin")
                shape = (
                    n_imgs_per_shard,
                    len(self.metadata.layers),
                    self.metadata.n_patches_per_img + 1,
                    self.metadata.d_vit,
                )
                acts = np.memmap(acts_fpath, mode="c", dtype=np.float32, shape=shape)
                # Choose the layer and the non-CLS tokens.
                acts = acts[:, self.layer_index, 1:]

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
            case ("cls", "meanpool"):
                # Return a CLS token from a random image and meanpool over all layers.
                return self.metadata.n_imgs
            case ("meanpool", "all"):
                # Return the meanpool of all patches from a random image and random layer.
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("meanpool", int()):
                # Return the meanpool of all patches from a random image and fixed layer.
                return self.metadata.n_imgs
            case ("meanpool", "meanpool"):
                # Return the meanpool of all patches from a random image and meanpool over all layers.
                return self.metadata.n_imgs
            case ("patches", int()):
                # Return a patch from a random image, fixed layer, and random patch.
                return self.metadata.n_imgs * (self.metadata.n_patches_per_img)
            case ("patches", "meanpool"):
                # Return a patch from a random image, meanpooled over all layers, and a random patch.
                return self.metadata.n_imgs * (self.metadata.n_patches_per_img)
            case ("patches", "all"):
                # Return a patch from a random image, random layer and random patch.
                return (
                    self.metadata.n_imgs
                    * len(self.metadata.layers)
                    * self.metadata.n_patches_per_img
                )
            case _:
                typing.assert_never((self.cfg.patches, self.cfg.layer))
