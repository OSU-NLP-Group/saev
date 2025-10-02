# src/saev/data/shards.py
"""
Library code for reading and writing sharded activations to disk.
"""

import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import typing as tp
from collections.abc import Callable, Sequence

import beartype
import einops
import numpy as np
import torch
from jaxtyping import Float, UInt8, jaxtyped
from PIL import Image
from torch import Tensor

from . import datasets

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True, kw_only=True)
class Metadata:
    family: tp.Literal["clip", "siglip", "dinov2", "dinov3", "fake-clip"]
    ckpt: str
    layers: tuple[int, ...]
    patches_per_ex: int
    cls_token: bool
    d_model: int
    n_ex: int
    patches_per_shard: int
    data: dict[str, object]
    pixel_agg: tp.Literal["majority", "prefer-fg", None] = None
    dtype: tp.Literal["float32"] = "float32"
    protocol: tp.Literal["1.0.0", "1.1"] = "1.1"

    def __post_init__(self):
        # Check that at least one image per shard can fit.
        msg = "At least one image per shard must fit; increase patches_per_shard."
        assert self.ex_per_shard >= 1, msg

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
    def n_tokens_per_ex(self) -> int:
        return self.patches_per_ex + int(self.cls_token)

    @property
    def n_shards(self) -> int:
        return math.ceil(self.n_ex / self.ex_per_shard)

    @property
    def ex_per_shard(self) -> int:
        """
        Calculate the number of images per shard based on the protocol.

        Returns:
            Number of images that fit in a shard.
        """
        n_tokens_per_ex = self.patches_per_ex + (1 if self.cls_token else 0)
        return self.patches_per_shard // (n_tokens_per_ex * len(self.layers))

    @property
    def shard_shape(self) -> tuple[int, int, int, int]:
        return (
            self.ex_per_shard,
            len(self.layers),
            self.n_tokens_per_ex,
            self.d_model,
        )


@jaxtyped(typechecker=beartype.beartype)
class RecordedTransformer(torch.nn.Module):
    _storage: Float[Tensor, "batch n_layers all_patches dim"] | None
    _i: int

    def __init__(
        self,
        model: torch.nn.Module,
        patches_per_ex: int,
        cls_token: bool,
        layers: Sequence[int],
    ):
        super().__init__()

        self.model = model

        self.patches_per_ex = patches_per_ex
        self.cls_token = cls_token
        self.layers = layers

        self.patches = model.get_patches(patches_per_ex)

        self._storage = None
        self._i = 0

        self.logger = logging.getLogger(f"recorder({model.name})")

        for i in self.layers:
            self.model.get_residuals()[i].register_forward_hook(self.hook)

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
        if not self.cls_token and selected_output.shape[1] == self.patches_per_ex + 1:
            # Model has CLS token but we don't want to store it - skip first token
            selected_output = selected_output[:, 1:, :]

        self._storage[:, self._i] = selected_output.detach()
        self._i += 1

    def _empty_storage(self, batch: int, dim: int, device: torch.device):
        patches_per_ex = self.patches_per_ex
        if self.cls_token:
            patches_per_ex += 1

        return torch.zeros(
            (batch, len(self.layers), patches_per_ex, dim), device=device
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
        result = self.model(batch, **kwargs)
        return result, self.activations


@beartype.beartype
@jaxtyped(typechecker=beartype.beartype)
class LabelsWriter:
    """
    LabelsWriter handles writing patch-level segmentation labels to a single binary file.
    """

    labels: UInt8[np.ndarray, "n_ex n_patches"] | None
    labels_path: str
    patches_per_ex: int
    n_ex: int
    current_idx: int
    has_written: bool

    def __init__(self, root: pathlib.Path, *, patches_per_ex: int, n_ex: int):
        self.logger = logging.getLogger("labels-writer")
        self.root = root
        self.patches_per_ex = patches_per_ex
        self.n_ex = n_ex
        self.has_written = False
        self.current_idx = 0

        # Always create memory-mapped file for labels
        # If nothing is written, it will be deleted in flush()
        self.labels_path = os.path.join(self.root, "labels.bin")
        self.labels = np.memmap(
            self.labels_path,
            mode="w+",
            dtype=np.uint8,
            shape=(self.n_ex, self.patches_per_ex),
        )
        self.logger.info("Opened labels file '%s'.", self.labels_path)

    @beartype.beartype
    def write_batch(self, batch_labels: np.ndarray | Tensor, start_idx: int):
        """
        Write a batch of labels to the memory-mapped file.

        Args:
            batch_labels: Array of shape (batch_size, patches_per_ex) with uint8 dtype
            start_idx: Starting index in the global labels array
        """
        # Convert to numpy if needed
        if isinstance(batch_labels, torch.Tensor):
            batch_labels = batch_labels.cpu().numpy()

        batch_size = len(batch_labels)
        assert start_idx + batch_size <= self.n_ex
        assert batch_labels.shape == (batch_size, self.patches_per_ex)
        assert batch_labels.dtype == np.uint8

        self.labels[start_idx : start_idx + batch_size] = batch_labels
        self.current_idx = start_idx + batch_size
        self.has_written = True

    def flush(self) -> None:
        """Flush the memory-mapped file to disk if anything was written."""
        if self.labels is not None and self.has_written:
            self.labels.flush()
            self.logger.info("Flushed labels to '%s'.", self.labels_path)


@jaxtyped(typechecker=beartype.beartype)
class ShardWriter:
    """
    ShardWriter is a stateful object that handles sharded activation writing to disk.
    """

    root: str
    shape: tuple[int, int, int, int]
    shard: int
    acts_path: str
    acts: Float[np.ndarray, "ex_per_shard n_layers all_patches d_model"] | None
    filled: int
    labels_writer: LabelsWriter

    def __init__(self, root: pathlib.Path, md: Metadata):
        self.logger = logging.getLogger("shard-writer")

        self.root = root
        patches_per_ex = md.patches_per_ex
        if md.cls_token:
            patches_per_ex += 1

        self.ex_per_shard = md.patches_per_shard // len(md.layers) // patches_per_ex
        self.shape = (self.ex_per_shard, len(md.layers), patches_per_ex, md.d_model)

        # builder for shard manifest
        self._shards: ShardInfo = ShardInfo()

        # Always initialize labels writer (it handles non-seg datasets internally)
        self.labels_writer = LabelsWriter(
            root, patches_per_ex=patches_per_ex, n_ex=md.n_ex
        )

        self.shard = -1
        self.acts = None
        self.next_shard()

    def write_batch(
        self,
        activations: Float[Tensor, "batch n_layers all_patches d_model"],
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
        offset = self.ex_per_shard * self.shard

        if end_idx >= offset + self.ex_per_shard:
            # We have run out of space in this mmap'ed file. Let's fill it as much as we can.
            n_fit = offset + self.ex_per_shard - start_idx
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
            msg = f"0 <= {start_idx} - {offset} <= {offset} + {self.ex_per_shard}"
            assert 0 <= start_idx - offset <= offset + self.ex_per_shard, msg
            msg = f"0 <= {end_idx} - {offset} <= {offset} + {self.ex_per_shard}"
            assert 0 <= end_idx - offset <= offset + self.ex_per_shard, msg
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
                Shard(name=os.path.basename(self.acts_path), n_ex=self.filled)
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
@dataclasses.dataclass(frozen=True)
class Shard:
    """
    A single shard entry in shards.json, recording the filename and number of images.
    """

    name: str
    n_ex: int


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
def worker_fn(
    *,
    family: str,
    ckpt: str,
    patches_per_ex: int,
    cls_token: bool,
    d_model: int,
    layers: list[int],
    data: datasets.Config,
    batch_size: int,
    n_workers: int,
    patches_per_shard: int,
    pixel_agg: tp.Literal["majority", "prefer-fg", None],
    shards_root: pathlib.Path,
    device: str,
):
    """
    Args:
        family: Transformer family (dinov2, dinov3, clip, etc).
        ckpt: Transformer ckpt (hf-hub:imageomics/bioclip2, etc).
        patches_per_ex: Number of patches per example.
        cls_token: Whether the transformer has a [CLS] token.
        d_model: Hidden dimension of transformer.
        layers: The layers to record activations for.
        data: Config for the particular (image) dataset to load.
        batch_size: Batch size for the dataset.
        n_workers: Number of workers for loading examples fromm the dataset.
        patches_per_shard: Number of patches per disk shard.
        pixel_agg: Optional method for aggregating segmentation label pixels.
        shards_root: Where to save shards. Should end with 'shards'. See [disk-layout.md](../../developers/disk-layout.md); this is $SAEV_SCRATCH/saev/shards.
        device: Device for doing the computation.
    """
    from saev import helpers
    from saev.data import models

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

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA device available, using CPU.")
        device = "cpu"

    md = Metadata(
        family=family,
        ckpt=ckpt,
        layers=tuple(layers),
        patches_per_ex=patches_per_ex,
        cls_token=cls_token,
        d_model=d_model,
        n_ex=data.n_ex,
        patches_per_shard=patches_per_shard,
        data={**dataclasses.asdict(data), "__class__": data.__class__.__name__},
        pixel_agg=pixel_agg if datasets.is_seg_dataset(data) else None,
    )
    model_cls = models.load_model_cls(family)
    model_instance = model_cls(ckpt).to(device)
    model = RecordedTransformer(model_instance, patches_per_ex, cls_token, layers)

    img_tr, sample_tr = model_cls.make_transforms(ckpt, patches_per_ex)

    seg_tr = None
    if datasets.is_seg_dataset(data):
        # For segmentation datasets, create a transform that converts pixels to patches
        # Use make_resize with NEAREST interpolation for segmentation masks
        from PIL import Image

        seg_resize_tr = model_cls.make_resize(
            ckpt, patches_per_ex, scale=1.0, resample=Image.NEAREST
        )

        def seg_to_patches(seg):
            """Transform that resizes segmentation and converts to patch labels."""

            # Convert to patch labels
            return pixel_to_patch_labels(
                seg_resize_tr(seg),
                patches_per_ex,
                patch_size=model_instance.patch_size,
                pixel_agg=pixel_agg,
                bg_label=data.bg_label,
            )

        seg_tr = seg_to_patches

    dataloader = get_dataloader(
        data,
        batch_size=batch_size,
        n_workers=n_workers,
        img_tr=img_tr,
        seg_tr=seg_tr,
        sample_tr=sample_tr,
    )

    n_batches = math.ceil(data.n_ex / batch_size)
    logger.info("Dumping %d batches of %d examples.", n_batches, batch_size)

    model = model.to(device)

    (shards_root / md.hash).mkdir(exist_ok=True)
    md.dump(shards_root / md.hash)

    # Use context manager for proper cleanup
    with ShardWriter(shards_root / md.hash, md) as writer:
        i = 0
        # Calculate and write transformer activations.
        with torch.inference_mode():
            for batch in helpers.progress(dataloader, total=n_batches):
                imgs = batch.get("image").to(device)
                grid = batch.get("grid")
                if grid is not None:
                    grid = grid.to(device)
                    out, cache = model(imgs, grid=grid)
                else:
                    out, cache = model(imgs)
                # cache has shape [batch size, n layers, n patches + 1, d model]
                del out

                # Write activations and labels (if present) in one call
                patch_labels = batch.get("patch_labels")
                if patch_labels is not None:
                    logger.debug(
                        "Found patch_labels in batch: shape=%s",
                        patch_labels.shape
                        if hasattr(patch_labels, "shape")
                        else "unknown",
                    )
                    # Ensure correct shape
                    assert patch_labels.shape == (len(cache), patches_per_ex)
                else:
                    logger.debug(f"No patch_labels in batch. Keys: {batch.keys()}")

                writer.write_batch(cache, i, patch_labels=patch_labels)

                i += len(cache)


@beartype.beartype
def get_dataloader(
    data: datasets.Config,
    *,
    batch_size: int,
    n_workers: int,
    img_tr: Callable | None = None,
    seg_tr: Callable | None = None,
    sample_tr: Callable | None = None,
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for a default map-style dataset.

    Args:
        data: Config for the dataset.
        batch_size: Batch size.
        n_workers: Number of dataloader workers.
        img_tr: Image transform to be applied to each image.
        seg_tr: Segmentation transform to be applied to masks.
        sample_tr: Transform to be applied to sample dicts.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches, `'index'` keys containing original dataset indices and `'label'` keys containing label batches.
    """
    dataset = datasets.get_dataset(
        data, img_transform=img_tr, seg_transform=seg_tr, sample_transform=sample_tr
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=n_workers,
        persistent_workers=n_workers > 0,
        shuffle=False,
        pin_memory=False,
    )
    return dataloader


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
