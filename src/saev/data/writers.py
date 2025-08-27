# src/saev/data/writers.py
import dataclasses
import hashlib
import json
import logging
import math
import os
import typing
from collections.abc import Callable

import beartype
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from saev import helpers

from . import images

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for calculating and saving ViT activations."""

    data: images.Config = dataclasses.field(default_factory=images.Imagenet)
    """Which dataset to use."""
    dump_to: str = os.path.join(".", "shards")
    """Where to write shards."""
    vit_family: typing.Literal["clip", "siglip", "dinov2", "dinov3"] = "clip"
    """Which model family."""
    vit_ckpt: str = "ViT-L-14/openai"
    """Specific model checkpoint."""
    vit_batch_size: int = 1024
    """Batch size for ViT inference."""
    n_workers: int = 8
    """Number of dataloader workers."""
    d_vit: int = 1024
    """Dimension of the ViT activations (depends on model)."""
    vit_layers: list[int] = dataclasses.field(default_factory=lambda: [-2])
    """Which layers to save. By default, the second-to-last layer."""
    n_patches_per_img: int = 256
    """Number of ViT patches per image (depends on model)."""
    cls_token: bool = True
    """Whether the model has a [CLS] token."""
    max_patches_per_shard: int = 2_400_000
    """Maximum number of activations per shard; 2.4M is approximately 10GB for 1024-dimensional 4-byte activations."""

    ssl: bool = True
    """Whether to use SSL."""

    # Hardware
    device: str = "cuda"
    """Which device to use."""
    n_hours: float = 24.0
    """Slurm job length."""
    slurm_acct: str = ""
    """Slurm account string."""
    slurm_partition: str = ""
    """Slurm partition."""
    log_to: str = "./logs"
    """Where to log Slurm job stdout/stderr."""


@jaxtyped(typechecker=beartype.beartype)
class RecordedVisionTransformer(torch.nn.Module):
    _storage: Float[Tensor, "batch n_layers all_patches dim"] | None
    _i: int

    def __init__(
        self,
        vit: torch.nn.Module,
        n_patches_per_img: int,
        cls_token: bool,
        layers: list[int],
    ):
        super().__init__()

        self.vit = vit

        self.n_patches_per_img = n_patches_per_img
        self.cls_token = cls_token
        self.layers = layers

        self.patches = vit.get_patches(n_patches_per_img)

        self._storage = None
        self._i = 0

        self.logger = logging.getLogger(f"recorder({vit.name})")

        for i in self.layers:
            self.vit.get_residuals()[i].register_forward_hook(self.hook)

    def hook(
        self, module, args: tuple, output: Float[Tensor, "batch n_layers dim"]
    ) -> None:
        if self._storage is None:
            batch, _, dim = output.shape
            self._storage = self._empty_storage(batch, dim, output.device)

        if self._storage[:, self._i, 0, :].shape != output[:, 0, :].shape:
            batch, _, dim = output.shape

            old_batch, _, _, old_dim = self._storage.shape
            msg = "Output shape does not match storage shape: (batch) %d != %d or (dim) %d != %d"
            self.logger.warning(msg, old_batch, batch, old_dim, dim)

            self._storage = self._empty_storage(batch, dim, output.device)

        self._storage[:, self._i] = output[:, self.patches, :].detach()
        self._i += 1

    def _empty_storage(self, batch: int, dim: int, device: torch.device):
        n_patches_per_img = self.n_patches_per_img
        if self.cls_token:
            n_patches_per_img += 1

        return torch.zeros(
            (batch, len(self.layers), n_patches_per_img, dim), device=device
        )

    def reset(self):
        self._i = 0

    @property
    def activations(self) -> Float[Tensor, "batch n_layers all_patches dim"]:
        if self._storage is None:
            raise RuntimeError("First call forward()")
        return self._storage.cpu()

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> tuple[
        Float[Tensor, "batch patches dim"],
        Float[Tensor, "batch n_layers all_patches dim"],
    ]:
        self.reset()
        result = self.vit(batch)
        return result, self.activations


@beartype.beartype
class ShardWriter:
    """
    ShardWriter is a stateful object that handles sharded activation writing to disk.
    """

    root: str
    shape: tuple[int, int, int, int]
    shard: int
    acts_path: str
    acts: Float[np.ndarray, "n_imgs_per_shard n_layers all_patches d_vit"] | None
    filled: int

    def __init__(self, cfg: Config):
        self.logger = logging.getLogger("shard-writer")

        self.root = get_acts_dir(cfg)

        n_patches_per_img = cfg.n_patches_per_img
        if cfg.cls_token:
            n_patches_per_img += 1
        self.n_imgs_per_shard = (
            cfg.max_patches_per_shard // len(cfg.vit_layers) // n_patches_per_img
        )
        self.shape = (
            self.n_imgs_per_shard,
            len(cfg.vit_layers),
            n_patches_per_img,
            cfg.d_vit,
        )

        # builder for shard manifest
        self._shards: ShardInfo = ShardInfo()

        self.shard = -1
        self.acts = None
        self.next_shard()

    @jaxtyped(typechecker=beartype.beartype)
    def __setitem__(
        self, i: slice, val: Float[Tensor, "_ n_layers all_patches d_vit"]
    ) -> None:
        assert i.step is None
        a, b = i.start, i.stop
        assert len(val) == b - a

        offset = self.n_imgs_per_shard * self.shard

        if b >= offset + self.n_imgs_per_shard:
            # We have run out of space in this mmap'ed file. Let's fill it as much as we can.
            n_fit = offset + self.n_imgs_per_shard - a
            self.acts[a - offset : a - offset + n_fit] = val[:n_fit]
            self.filled = a - offset + n_fit

            self.next_shard()

            # Recursively call __setitem__ in case we need *another* shard
            self[a + n_fit : b] = val[n_fit:]
        else:
            msg = f"0 <= {a} - {offset} <= {offset} + {self.n_imgs_per_shard}"
            assert 0 <= a - offset <= offset + self.n_imgs_per_shard, msg
            msg = f"0 <= {b} - {offset} <= {offset} + {self.n_imgs_per_shard}"
            assert 0 <= b - offset <= offset + self.n_imgs_per_shard, msg
            self.acts[a - offset : b - offset] = val
            self.filled = b - offset

    def flush(self) -> None:
        if self.acts is not None:
            self.acts.flush()

            # record shard info
            self._shards.append(
                Shard(name=os.path.basename(self.acts_path), n_imgs=self.filled)
            )
            self._shards.dump(self.root)

        self.acts = None

    def next_shard(self) -> None:
        self.flush()

        self.shard += 1
        self._count = 0
        self.acts_path = os.path.join(self.root, f"acts{self.shard:06}.bin")
        self.acts = np.memmap(
            self.acts_path, mode="w+", dtype=np.float32, shape=self.shape
        )
        self.filled = 0

        self.logger.info("Opened shard '%s'.", self.acts_path)


@beartype.beartype
def get_acts_dir(cfg: Config) -> str:
    """
    Return the activations directory based on the relevant values of a config.
    Also saves a metadata.json file to that directory for human reference.

    Args:
        cfg: Config for experiment.

    Returns:
        Directory to where activations should be dumped/loaded from.
    """
    metadata = Metadata.from_cfg(cfg)

    acts_dir = os.path.join(cfg.dump_to, metadata.hash)
    os.makedirs(acts_dir, exist_ok=True)

    metadata.dump(acts_dir)

    return acts_dir


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Metadata:
    vit_family: typing.Literal["clip", "siglip", "dinov2", "dinov3"]
    vit_ckpt: str
    layers: tuple[int, ...]
    n_patches_per_img: int
    cls_token: bool
    d_vit: int
    n_imgs: int
    max_patches_per_shard: int
    data: dict[str, object]
    dtype: typing.Literal["float32"] = "float32"
    protocol: typing.Literal["1.0.0"] = "1.0.0"

    def __post_init__(self):
        # Check that at least one image per shard can fit.
        assert self.n_imgs_per_shard >= 1, (
            "At least one image per shard must fit; increase max_patches_per_shard."
        )

    @classmethod
    def from_cfg(cls, cfg: Config) -> "Metadata":
        return cls(
            cfg.vit_family,
            cfg.vit_ckpt,
            tuple(cfg.vit_layers),
            cfg.n_patches_per_img,
            cfg.cls_token,
            cfg.d_vit,
            cfg.data.n_imgs,
            cfg.max_patches_per_shard,
            {**dataclasses.asdict(cfg.data), "__class__": cfg.data.__class__.__name__},
        )

    @classmethod
    def load(cls, shard_root: str) -> "Metadata":
        with open(os.path.join(shard_root, "metadata.json")) as fd:
            dct = json.load(fd)
        dct["layers"] = tuple(dct.pop("layers"))
        return cls(**dct)

    def dump(self, shard_root: str):
        with open(os.path.join(shard_root, "metadata.json"), "w") as fd:
            json.dump(dataclasses.asdict(self), fd, indent=4)

    @property
    def hash(self) -> str:
        cfg_bytes = json.dumps(
            dataclasses.asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(cfg_bytes).hexdigest()

    @property
    def n_tokens_per_img(self) -> int:
        return self.n_patches_per_img + int(self.cls_token)

    @property
    def n_shards(self) -> int:
        return math.ceil(self.n_imgs / self.n_imgs_per_shard)

    @property
    def n_imgs_per_shard(self) -> int:
        """
        Calculate the number of images per shard based on the protocol.

        Returns:
            Number of images that fit in a shard.
        """
        n_tokens_per_img = self.n_patches_per_img + (1 if self.cls_token else 0)
        return self.max_patches_per_shard // (n_tokens_per_img * len(self.layers))

    @property
    def shard_shape(self) -> tuple[int, int, int, int]:
        return (
            self.n_imgs_per_shard,
            len(self.layers),
            self.n_tokens_per_img,
            self.d_vit,
        )


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Shard:
    """
    A single shard entry in shards.json, recording the filename and number of images.
    """

    name: str
    n_imgs: int


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ShardInfo:
    """
    A read-only container for shard metadata as recorded in shards.json.
    """

    shards: list[Shard] = dataclasses.field(default_factory=list)

    @classmethod
    def load(cls, shard_path: str) -> "ShardInfo":
        with open(os.path.join(shard_path, "shards.json")) as fd:
            data = json.load(fd)
        return cls([Shard(**entry) for entry in data])

    def dump(self, fpath: str) -> None:
        with open(os.path.join(fpath, "shards.json"), "w") as fd:
            json.dump([dataclasses.asdict(s) for s in self.shards], fd, indent=2)

    def append(self, shard: Shard):
        self.shards.append(shard)

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, i):
        return self.shards[i]

    def __iter__(self):
        yield from self.shards


@beartype.beartype
def get_dataloader(
    cfg: Config, *, img_transform: Callable | None = None
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for a default map-style dataset.

    Args:
        cfg: Config.
        img_transform: Image transform to be applied to each image.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches, `'index'` keys containing original dataset indices and `'label'` keys containing label batches.
    """
    dataset = images.get_dataset(cfg.data, img_transform=img_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.vit_batch_size,
        drop_last=False,
        num_workers=cfg.n_workers,
        persistent_workers=cfg.n_workers > 0,
        shuffle=False,
        pin_memory=False,
    )
    return dataloader


@beartype.beartype
def worker_fn(cfg: Config):
    """
    Args:
        cfg: Config for activations.
    """
    from . import models

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("worker_fn")

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA device available, using CPU.")
        cfg = dataclasses.replace(cfg, device="cpu")

    vit = models.make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)
    vit = RecordedVisionTransformer(
        vit, cfg.n_patches_per_img, cfg.cls_token, cfg.vit_layers
    )
    img_transform = models.make_img_transform(cfg.vit_family, cfg.vit_ckpt)
    dataloader = get_dataloader(cfg, img_transform=img_transform)

    writer = ShardWriter(cfg)

    n_batches = cfg.data.n_imgs // cfg.vit_batch_size + 1
    logger.info("Dumping %d batches of %d examples.", n_batches, cfg.vit_batch_size)

    vit = vit.to(cfg.device)
    # vit = torch.compile(vit)

    i = 0
    # Calculate and write ViT activations.
    with torch.inference_mode():
        for batch in helpers.progress(dataloader, total=n_batches):
            imgs = batch.pop("image").to(cfg.device)
            # cache has shape [batch size, n layers, n patches + 1, d vit]
            out, cache = vit(imgs)
            del out

            writer[i : i + len(cache)] = cache
            i += len(cache)

    writer.flush()


@beartype.beartype
class IndexLookup:
    """
    Index <-> shard helper.

    `map()`      – turn a global dataset index into precise physical offsets.
    `length()`   – dataset size for a particular (patches, layer) view.

    Parameters
    ----------
    metadata : Metadata
        Pre-computed dataset statistics (images, patches, layers, shard size).
    patches: 'cls' | 'image' | 'all'
    layer: int | 'all'
    """

    def __init__(
        self,
        metadata: Metadata,
        patches: typing.Literal["cls", "image", "all"],
        layer: int | typing.Literal["all"],
    ):
        if not metadata.cls_token and patches == "cls":
            raise ValueError("Cannot return [CLS] token if one isn't present.")

        self.metadata = metadata
        self.patches = patches

        if isinstance(layer, int) and layer not in metadata.layers:
            raise ValueError(f"Layer {layer} not in {metadata.layers}.")
        self.layer = layer
        self.layer_to_idx = {layer: i for i, layer in enumerate(metadata.layers)}

    def map_global(self, i: int) -> tuple[int, tuple[int, int, int]]:
        """
        Return
        -------
        (
            shard_i,
            index in shard (img_i_in_shard, layer_i, token_i)
        )
        """
        n = self.length()

        if i < 0 or i >= n:
            raise IndexError(f"{i=} out of range [0, {n})")

        match (self.patches, self.layer):
            case ("cls", int()):
                # For CLS token with specific layer, i is the image index
                img_i = i
                shard_i, img_i_in_shard = self.map_img(img_i)
                # CLS token is at position 0
                return shard_i, (img_i_in_shard, self.layer_to_idx[self.layer], 0)
            case ("image", int()):
                # For image patches with specific layer, i is (img_idx * n_patches_per_img + patch_idx)
                img_i = i // self.metadata.n_patches_per_img
                token_i = i % self.metadata.n_patches_per_img

                shard_i, img_i_in_shard = self.map_img(img_i)
                return shard_i, (img_i_in_shard, self.layer_to_idx[self.layer], token_i)
            case ("image", "all"):
                raise NotImplementedError()
            case ("all", int()):
                n_tokens_per_img = self.metadata.n_patches_per_img + (
                    1 if self.metadata.cls_token else 0
                )
                img_i = i // n_tokens_per_img
                token_i = i % n_tokens_per_img
                shard_i, img_i_in_shard = self.map_img(img_i)
                return shard_i, (img_i_in_shard, self.layer_to_idx[self.layer], token_i)
            case ("all", "all"):
                # For all tokens (CLS + patches) with all layers
                # Calculate total tokens per image across all layers
                n_tokens_per_img = self.metadata.n_patches_per_img + (
                    1 if self.metadata.cls_token else 0
                )
                total_tokens_per_img = n_tokens_per_img * len(self.metadata.layers)

                # Calculate which image and which token within that image
                img_i = i // total_tokens_per_img
                remainder = i % total_tokens_per_img

                # Calculate which layer and which token within that layer
                layer_idx = remainder // n_tokens_per_img
                token_i = remainder % n_tokens_per_img

                # Map to physical location
                shard_i, img_i_in_shard = self.map_img(img_i)
                return shard_i, (img_i_in_shard, layer_idx, token_i)

            case _:
                typing.assert_never((self.patches, self.layer))

    def map_img(self, img_i: int) -> tuple[int, int]:
        """
        Return
        -------
        (shard_i, img_i_in_shard)
        """
        if img_i < 0 or img_i >= self.metadata.n_imgs:
            raise IndexError(f"{img_i=} out of range [0, {self.metadata.n_imgs})")

        # Calculate which shard contains this image
        shard_i = img_i // self.metadata.n_imgs_per_shard
        img_i_in_shard = img_i % self.metadata.n_imgs_per_shard

        return shard_i, img_i_in_shard

    def length(self) -> int:
        match (self.patches, self.layer):
            case ("cls", "all"):
                # Return a CLS token from a random image and random layer.
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("cls", int()):
                # Return a CLS token from a random image and fixed layer.
                return self.metadata.n_imgs
            case ("image", int()):
                # Return a patch from a random image, fixed layer, and random patch.
                return self.metadata.n_imgs * (self.metadata.n_patches_per_img)
                return self.metadata.n_imgs * (self.metadata.n_patches_per_img)
            case ("image", "all"):
                # Return a patch from a random image, random layer and random patch.
                return (
                    self.metadata.n_imgs
                    * len(self.metadata.layers)
                    * self.metadata.n_patches_per_img
                )
            case ("all", int()):
                # Return a patch from a random image, specific layer and random patch.
                return self.metadata.n_imgs * (
                    self.metadata.n_patches_per_img + int(self.metadata.cls_token)
                )
            case ("all", "all"):
                # Return a patch from a random image, random layer and random patch.
                return (
                    self.metadata.n_imgs
                    * len(self.metadata.layers)
                    * (self.metadata.n_patches_per_img + int(self.metadata.cls_token))
                )
            case _:
                typing.assert_never((self.patches, self.layer))
