# src/saev/data/writers.py
import dataclasses
import hashlib
import json
import logging
import math
import os
import typing as tp
from collections.abc import Callable

import beartype
import einops
import numpy as np
import torch
from jaxtyping import Float, UInt8, jaxtyped
from PIL import Image
from torch import Tensor

from saev import helpers

from . import datasets

logger = logging.getLogger(__name__)


@jaxtyped(typechecker=beartype.beartype)
def pixel_to_patch_labels(
    seg: Image.Image,
    n_patches: int,
    patch_size: int,
    pixel_agg: tp.Literal["majority", "prefer-fg"] = "majority",
    bg_label: int = 0,
    max_classes: int = 256,
) -> UInt8[Tensor, " n_patches"]:
    """
    Convert pixel-level segmentation to patch-level labels using vectorized operations.

    Args:
        seg: Pixel-level segmentation mask as PIL Image
        n_patches: Total number of patches expected
        patch_size: Size of each patch in pixels
        pixel_agg: How to aggregate pixel labels into patch labels
        bg_label: Background label index
        max_classes: Maximum number of classes (for bincount)

    Returns:
        Patch labels as uint8 tensor of shape (n_patches,)
    """
    # Convert to torch tensor for vectorized operations
    seg_tensor = torch.from_numpy(np.array(seg, dtype=np.uint8))
    assert seg_tensor.ndim == 2

    h, w = seg_tensor.shape

    # Calculate patch grid dimensions
    patch_grid_h = h // patch_size
    patch_grid_w = w // patch_size
    assert patch_grid_w * patch_grid_h == n_patches, (
        f"Image size {w}x{h} with patch_size {patch_size} gives {patch_grid_w}x{patch_grid_h} = {patch_grid_w * patch_grid_h} patches, expected {n_patches}"
    )

    # Reshape into patches using einops: (n_patches, patch_size * patch_size)
    patches = einops.rearrange(
        seg_tensor,
        "(h p1) (w p2) -> (h w) (p1 p2)",
        p1=patch_size,
        p2=patch_size,
        h=patch_grid_h,
        w=patch_grid_w,
    )

    # Use vectorized bincount approach to get class counts for all patches at once
    # counts[i, c] = number of times class c appears in patch i
    offsets = torch.arange(n_patches, device=patches.device).unsqueeze(1) * max_classes
    flat = (patches + offsets).reshape(-1)
    counts = torch.bincount(flat, minlength=n_patches * max_classes).reshape(
        n_patches, max_classes
    )

    if pixel_agg == "majority":
        # Take the most common label in each patch
        patch_labels = counts.argmax(dim=1)
    elif pixel_agg == "prefer-fg":
        # Take the most common non-background label, or background if all background
        nonbg = counts.clone()
        nonbg[:, bg_label] = 0
        has_nonbg = nonbg.sum(dim=1) > 0
        nonbg_arg = nonbg.argmax(dim=1)
        bg_tensor = torch.full_like(nonbg_arg, bg_label)
        patch_labels = torch.where(has_nonbg, nonbg_arg, bg_tensor)
    else:
        tp.assert_never(pixel_agg)

    return patch_labels.to(torch.uint8)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for calculating and saving ViT activations."""

    data: datasets.Config = dataclasses.field(default_factory=datasets.Imagenet)
    """Which dataset to use."""
    dump_to: str = os.path.join(".", "shards")
    """Where to write shards."""
    vit_family: tp.Literal["clip", "siglip", "dinov2", "dinov3", "fake-clip"] = "clip"
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
    pixel_agg: tp.Literal["majority", "prefer-fg"] = "majority"
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

        # Select patches based on cls_token setting
        selected_output = output[:, self.patches, :]
        if (
            not self.cls_token
            and selected_output.shape[1] == self.n_patches_per_img + 1
        ):
            # Model has CLS token but we don't want to store it - skip first token
            selected_output = selected_output[:, 1:, :]

        self._storage[:, self._i] = selected_output.detach()
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
        self, batch: Float[Tensor, "batch 3 width height"], **kwargs
    ) -> tuple[
        Float[Tensor, "batch patches dim"],
        Float[Tensor, "batch n_layers all_patches dim"],
    ]:
        self.reset()
        result = self.vit(batch, **kwargs)
        return result, self.activations


@beartype.beartype
@jaxtyped(typechecker=beartype.beartype)
class LabelsWriter:
    """
    LabelsWriter handles writing patch-level segmentation labels to a single binary file.
    """

    labels: UInt8[np.ndarray, "n_imgs n_patches"] | None
    labels_path: str
    n_patches_per_img: int
    n_imgs: int
    current_idx: int
    has_written: bool

    def __init__(self, cfg: Config):
        self.logger = logging.getLogger("labels-writer")
        self.root = get_acts_dir(cfg)
        self.n_patches_per_img = cfg.n_patches_per_img
        self.n_imgs = cfg.data.n_imgs
        self.has_written = False
        self.current_idx = 0

        # Always create memory-mapped file for labels
        # If nothing is written, it will be deleted in flush()
        self.labels_path = os.path.join(self.root, "labels.bin")
        self.labels = np.memmap(
            self.labels_path,
            mode="w+",
            dtype=np.uint8,
            shape=(self.n_imgs, self.n_patches_per_img),
        )
        self.logger.info("Opened labels file '%s'.", self.labels_path)

    @beartype.beartype
    def write_batch(self, batch_labels: np.ndarray | Tensor, start_idx: int):
        """
        Write a batch of labels to the memory-mapped file.

        Args:
            batch_labels: Array of shape (batch_size, n_patches_per_img) with uint8 dtype
            start_idx: Starting index in the global labels array
        """
        # Convert to numpy if needed
        if isinstance(batch_labels, torch.Tensor):
            batch_labels = batch_labels.cpu().numpy()

        batch_size = len(batch_labels)
        assert start_idx + batch_size <= self.n_imgs
        assert batch_labels.shape == (batch_size, self.n_patches_per_img)
        assert batch_labels.dtype == np.uint8

        self.labels[start_idx : start_idx + batch_size] = batch_labels
        self.current_idx = start_idx + batch_size
        self.has_written = True

    def flush(self) -> None:
        """Flush the memory-mapped file to disk if anything was written."""
        if self.labels is not None and self.has_written:
            self.labels.flush()
            self.logger.info("Flushed labels to '%s'.", self.labels_path)


@beartype.beartype
def _is_segmentation_dataset(data_cfg: datasets.Config) -> bool:
    """
    Check if a dataset configuration is for a segmentation dataset.

    Args:
        data_cfg: Dataset configuration

    Returns:
        True if this is a segmentation dataset that should have labels.bin
    """
    # Check if it's FakeSeg or SegFolder
    return isinstance(data_cfg, (datasets.FakeSeg, datasets.SegFolder))


@jaxtyped(typechecker=beartype.beartype)
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
    labels_writer: LabelsWriter

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

        # Always initialize labels writer (it handles non-seg datasets internally)
        self.labels_writer = LabelsWriter(cfg)

        self.shard = -1
        self.acts = None
        self.next_shard()

    def write_batch(
        self,
        activations: Float[Tensor, "batch n_layers all_patches d_vit"],
        start_idx: int,
        patch_labels: UInt8[Tensor, "batch n_patches"] | None = None,
    ) -> None:
        """Write a batch of activations and optionally patch labels.

        Args:
            activations: Batch of activations to write.
            start_idx: Starting index for this batch.
            patch_labels: Optional patch labels for segmentation datasets.
        """
        batch_size = len(activations)
        end_idx = start_idx + batch_size

        # Write activations (handling sharding)
        offset = self.n_imgs_per_shard * self.shard

        if end_idx >= offset + self.n_imgs_per_shard:
            # We have run out of space in this mmap'ed file. Let's fill it as much as we can.
            n_fit = offset + self.n_imgs_per_shard - start_idx
            self.acts[start_idx - offset : start_idx - offset + n_fit] = activations[
                :n_fit
            ]
            self.filled = start_idx - offset + n_fit

            # Write labels for the portion that fits
            if patch_labels is not None:
                # Convert to numpy uint8 if needed
                if isinstance(patch_labels, torch.Tensor):
                    labels_to_write = (
                        patch_labels[:n_fit].cpu().numpy().astype(np.uint8)
                    )
                elif not isinstance(patch_labels, np.ndarray):
                    labels_to_write = np.array(patch_labels[:n_fit], dtype=np.uint8)
                else:
                    labels_to_write = patch_labels[:n_fit]

                self.labels_writer.write_batch(labels_to_write, start_idx)

            self.next_shard()

            # Recursively call write_batch for remaining data
            if n_fit < batch_size:
                self.write_batch(
                    activations[n_fit:],
                    start_idx + n_fit,
                    patch_labels[n_fit:] if patch_labels is not None else None,
                )
        else:
            msg = f"0 <= {start_idx} - {offset} <= {offset} + {self.n_imgs_per_shard}"
            assert 0 <= start_idx - offset <= offset + self.n_imgs_per_shard, msg
            msg = f"0 <= {end_idx} - {offset} <= {offset} + {self.n_imgs_per_shard}"
            assert 0 <= end_idx - offset <= offset + self.n_imgs_per_shard, msg
            self.acts[start_idx - offset : end_idx - offset] = activations
            self.filled = end_idx - offset

            # Write labels if provided
            if patch_labels is not None:
                # Convert to numpy uint8 if needed
                if isinstance(patch_labels, torch.Tensor):
                    patch_labels = patch_labels.cpu().numpy().astype(np.uint8)
                elif not isinstance(patch_labels, np.ndarray):
                    patch_labels = np.array(patch_labels, dtype=np.uint8)

                self.labels_writer.write_batch(patch_labels, start_idx)

    def flush(self) -> None:
        if self.acts is not None:
            self.acts.flush()

            # record shard info
            self._shards.append(
                Shard(name=os.path.basename(self.acts_path), n_imgs=self.filled)
            )
            self._shards.dump(self.root)

        self.acts = None

        # Flush labels to disk
        self.labels_writer.flush()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - handle cleanup."""
        self.flush()

        # Delete empty labels file if nothing was written
        if not self.labels_writer.has_written:
            if os.path.exists(self.labels_writer.labels_path):
                os.remove(self.labels_writer.labels_path)
                self.logger.info(
                    "Removed empty labels file '%s'.", self.labels_writer.labels_path
                )

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
    vit_family: tp.Literal["clip", "siglip", "dinov2", "dinov3", "fake-clip"]
    vit_ckpt: str
    layers: tuple[int, ...]
    n_patches_per_img: int
    cls_token: bool
    d_vit: int
    n_imgs: int
    max_patches_per_shard: int
    data: dict[str, object]
    pixel_agg: tp.Literal["majority", "prefer-fg", None] = None
    dtype: tp.Literal["float32"] = "float32"
    protocol: tp.Literal["1.0.0", "1.1"] = "1.1"

    def __post_init__(self):
        # Check that at least one image per shard can fit.
        assert self.n_imgs_per_shard >= 1, (
            "At least one image per shard must fit; increase max_patches_per_shard."
        )

    @classmethod
    def from_cfg(cls, cfg: Config) -> "Metadata":
        # Only include pixel_agg for segmentation datasets
        pixel_agg = None
        if _is_segmentation_dataset(cfg.data):
            pixel_agg = cfg.pixel_agg

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
            pixel_agg,
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
    cfg: Config,
    *,
    img_tr: Callable | None = None,
    seg_tr: Callable | None = None,
    sample_tr: Callable | None = None,
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for a default map-style dataset.

    Args:
        cfg: Config.
        img_tr: Image transform to be applied to each image.
        seg_tr: Segmentation transform to be applied to masks.
        sample_tr: Transform to be applied to sample dicts.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches, `'index'` keys containing original dataset indices and `'label'` keys containing label batches.
    """
    dataset = datasets.get_dataset(
        cfg.data, img_transform=img_tr, seg_transform=seg_tr, sample_transform=sample_tr
    )

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

    vit_cls = models.load_vit_cls(cfg.vit_family)
    vit_instance = vit_cls(cfg.vit_ckpt).to(cfg.device)
    vit = RecordedVisionTransformer(
        vit_instance, cfg.n_patches_per_img, cfg.cls_token, cfg.vit_layers
    )

    img_tr, sample_tr = vit_cls.make_transforms(cfg.vit_ckpt, cfg.n_patches_per_img)

    seg_tr = None
    if _is_segmentation_dataset(cfg.data):
        # For segmentation datasets, create a transform that converts pixels to patches
        # Use make_resize with NEAREST interpolation for segmentation masks
        seg_resize_tr = vit_cls.make_resize(
            cfg.vit_ckpt, cfg.n_patches_per_img, scale=1.0, resample=Image.NEAREST
        )

        def seg_to_patches(seg):
            """Transform that resizes segmentation and converts to patch labels."""

            # Convert to patch labels
            return pixel_to_patch_labels(
                seg_resize_tr(seg),
                cfg.n_patches_per_img,
                patch_size=vit_instance.patch_size,
                pixel_agg=cfg.pixel_agg,
                bg_label=cfg.data.bg_label,
            )

        seg_tr = seg_to_patches

    dataloader = get_dataloader(cfg, img_tr=img_tr, seg_tr=seg_tr, sample_tr=sample_tr)

    n_batches = math.ceil(cfg.data.n_imgs / cfg.vit_batch_size)
    logger.info("Dumping %d batches of %d examples.", n_batches, cfg.vit_batch_size)

    vit = vit.to(cfg.device)
    # vit = torch.compile(vit)

    # Use context manager for proper cleanup
    with ShardWriter(cfg) as writer:
        i = 0
        # Calculate and write ViT activations.
        with torch.inference_mode():
            for batch in helpers.progress(dataloader, total=n_batches):
                imgs = batch.get("image").to(cfg.device)
                grid = batch.get("grid")
                if grid is not None:
                    grid = grid.to(cfg.device)
                    out, cache = vit(imgs, grid=grid)
                else:
                    out, cache = vit(imgs)
                # cache has shape [batch size, n layers, n patches + 1, d vit]
                del out

                # Write activations and labels (if present) in one call
                patch_labels = batch.get("patch_labels")
                if patch_labels is not None:
                    logger.debug(
                        f"Found patch_labels in batch: shape={patch_labels.shape if hasattr(patch_labels, 'shape') else 'unknown'}"
                    )
                    # Ensure correct shape
                    assert patch_labels.shape == (len(cache), cfg.n_patches_per_img)
                else:
                    logger.debug(f"No patch_labels in batch. Keys: {batch.keys()}")

                writer.write_batch(cache, i, patch_labels=patch_labels)

                i += len(cache)


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
        patches: tp.Literal["cls", "image", "all"],
        layer: int | tp.Literal["all"],
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
            case ("cls", "all"):
                # For CLS token with all layers, i represents (img_idx * n_layers + layer_idx)
                n_layers = len(self.metadata.layers)
                img_i = i // n_layers
                layer_idx = i % n_layers
                shard_i, img_i_in_shard = self.map_img(img_i)
                # CLS token is at position 0
                return shard_i, (img_i_in_shard, layer_idx, 0)
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
                # For image patches with all layers
                # Total patches per image across all layers
                total_patches_per_img = self.metadata.n_patches_per_img * len(
                    self.metadata.layers
                )

                # Calculate which image and which patch within that image across all layers
                img_i = i // total_patches_per_img
                remainder = i % total_patches_per_img

                # Calculate which layer and which patch within that layer
                layer_idx = remainder // self.metadata.n_patches_per_img
                token_i = remainder % self.metadata.n_patches_per_img

                shard_i, img_i_in_shard = self.map_img(img_i)
                return shard_i, (img_i_in_shard, layer_idx, token_i)
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
                tp.assert_never((self.patches, self.layer))

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
                tp.assert_never((self.patches, self.layer))
