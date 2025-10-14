# src/saev/data/shards.py
"""
Library code for reading and writing sharded activations to disk.
"""

import base64
import dataclasses
import enum
import hashlib
import json
import logging
import math
import os
import pathlib
import pickle
import typing as tp
from collections.abc import Callable, Sequence

import beartype
import einops
import numpy as np
import orjson
import torch
from jaxtyping import Float, UInt8, jaxtyped
from PIL import Image
from torch import Tensor

from .. import disk, helpers
from . import datasets

logger = logging.getLogger(__name__)


class PixelAgg(enum.Enum):
    """How to aggregate pixel-level segmentation labels to token-level labels (only for image segmentation datasets)."""

    MAJORITY = "majority"
    PREFER_FG = "prefer-fg"


@beartype.beartype
@dataclasses.dataclass(frozen=True, kw_only=True)
class Metadata:
    """
    Metadata for a sharded set of transformer activations.

    Args:
        family: The transformer family.
        ckpt: The transformer checkpoint.
        layers: Which layers were saved.
        content_tokens_per_example: The number of content tokens per example.
        cls_token: Whether the transformer has a [CLS] token as well.
        d_model: Model hidden dimension.
        n_examples: Number of examples.
        max_tokens_per_shard: The maximum number of tokens per shard.
        data: base64-encoded string of pickle.dumps(dataset).
        dataset: Absolute path to the root directory of the original dataset.
        pixel_agg: (only for image segmentation datasets) how the pixel-level segmentation labels were aggregated to token-level labels.
        dtype: How activations are stored.
        protocol: Protocol version.
    """

    family: tp.Literal["clip", "siglip", "dinov2", "dinov3", "fake-clip"]
    ckpt: str
    layers: tuple[int, ...]
    content_tokens_per_example: int
    cls_token: bool
    d_model: int
    n_examples: int
    max_tokens_per_shard: int
    data: str
    dataset: pathlib.Path
    pixel_agg: PixelAgg = PixelAgg.MAJORITY
    dtype: tp.Literal["float32"] = "float32"
    protocol: tp.Literal["1.0.0", "1.1", "2.1"] = "2.1"

    def __post_init__(self):
        # Check that at least one image per shard can fit.
        msg = "At least one example per shard must fit; increase max_tokens_per_shard."
        assert self.examples_per_shard >= 1, msg

        try:
            helpers.dumps(self.data)
        except TypeError as err:
            raise TypeError("self.data has an unhashable object") from err

    @classmethod
    def load(cls, shards_dir: pathlib.Path) -> "Metadata":
        """
        Loads a Metadata object from a metadata.json file in shards_dir.

        Args:
            shards_dir: Path to $SAEV_SCRATCH/saev/shards/<hash> as described in [disk-layout.md](../../developers/disk-layout.md).
        """
        assert disk.is_shards_dir(shards_dir)

        with open(shards_dir / "metadata.json") as fd:
            dct = json.load(fd)
        dct["layers"] = tuple(dct.pop("layers"))
        dct["dataset"] = pathlib.Path(dct["dataset"])
        dct["pixel_agg"] = PixelAgg(dct["pixel_agg"])
        return cls(**dct)

    def dump(self, shards_root: pathlib.Path):
        """
        Dumps a Metadata object to a metadata.json file in shards_root / hash.

        Args:
            shards_root: Path to $SAEV_SCRATCH/saev/shards as described in [disk-layout.md](../../developers/disk-layout.md).
        """
        assert disk.is_shards_root(shards_root)
        (shards_root / self.hash).mkdir(exist_ok=True)
        with open(shards_root / self.hash / "metadata.json", "wb") as fd:
            helpers.dump(self, fd, option=orjson.OPT_INDENT_2)

    @property
    def hash(self) -> str:
        """
        First 8 bytes of a SHA256 hash of the metadata configuration.

        Returns:
            Hexadecimal hash string uniquely identifying this configuration.
        """
        cfg_bytes = helpers.dumps(self, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(cfg_bytes).hexdigest()[:8]

    @property
    def tokens_per_example(self) -> int:
        """
        Total number of tokens per example including [CLS] token if present.

        Returns:
            Number of tokens plus one if [CLS] token is included.
        """
        return self.content_tokens_per_example + int(self.cls_token)

    @property
    def n_shards(self) -> int:
        """
        Total number of shards needed to store all examples.

        Returns:
            Number of shards required.
        """
        return math.ceil(self.n_examples / self.examples_per_shard)

    @property
    def examples_per_shard(self) -> int:
        """
        The number of examples per shard based on the protocol.

        Returns:
            Number of examples that fit in a shard.
        """
        return self.max_tokens_per_shard // (self.tokens_per_example * len(self.layers))

    @property
    def shard_shape(self) -> tuple[int, int, int, int]:
        """
        Shape of each shard file.

        Returns:
            Tuple of (examples_per_shard, n_layers, tokens_per_example, d_model).
        """
        return (
            self.examples_per_shard,
            len(self.layers),
            self.tokens_per_example,
            self.d_model,
        )

    def make_data_cfg(self) -> datasets.DatasetConfig:
        cfg = pickle.loads(base64.b64decode(self.data.encode("utf8")))
        assert isinstance(cfg, datasets.DatasetConfig)
        return cfg


@jaxtyped(typechecker=beartype.beartype)
class RecordedTransformer(torch.nn.Module):
    """
    A wrapper around a transformer model that records intermediate layer activations during forward passes.

    Args:
        model: The transformer model to wrap.
        content_tokens_per_example: Number of content tokens per example.
        cls_token: Whether to record the [CLS] token in addition to content tokens.
        layers: Which transformer layers to record activations from.

    Attributes:
        model: The wrapped transformer model.
        content_tokens_per_example: Number of content tokens per example.
        cls_token: Whether the [CLS] token is included in recorded activations.
        layers: Tuple of layer indices being recorded.
        token_i: Token indices to extract from model outputs.
        logger: Logger instance for this recorder.
    """

    model: torch.nn.Module
    content_tokens_per_example: int
    cls_token: bool
    layers: Sequence[int]
    token_i: slice

    _storage: Float[Tensor, "batch n_layers tokens_per_example dim"] | None
    _i: int

    def __init__(
        self,
        model: torch.nn.Module,
        content_tokens_per_example: int,
        cls_token: bool,
        layers: Sequence[int],
    ):
        super().__init__()

        self.model = model

        self.content_tokens_per_example = content_tokens_per_example
        self.cls_token = cls_token
        self.layers = layers

        self.token_i = model.get_token_i(content_tokens_per_example)

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

        # Select tokens based on cls_token setting
        selected_output = output[:, self.token_i, :]
        if (
            not self.cls_token
            and selected_output.shape[1] == self.content_tokens_per_example + 1
        ):
            # Model has CLS token but we don't want to store it - skip first token
            selected_output = selected_output[:, 1:, :]

        # Verify we're storing the right number of tokens (patches + cls if enabled)
        assert selected_output.shape[1] == self.tokens_per_example, (
            f"Shape mismatch: got {selected_output.shape[1]} tokens, expected {self.tokens_per_example} (content_tokens_per_example={self.content_tokens_per_example}, cls_token={self.cls_token})"
        )

        self._storage[:, self._i] = selected_output.detach()
        self._i += 1

    def _empty_storage(self, batch: int, dim: int, device: torch.device):
        return torch.zeros(
            (batch, len(self.layers), self.tokens_per_example, dim), device=device
        )

    def reset(self):
        self._i = 0

    @property
    def tokens_per_example(self) -> int:
        return self.content_tokens_per_example + int(self.cls_token)

    @property
    def activations(self) -> Float[Tensor, "batch n_layers tokens_per_example dim"]:
        if self._storage is None:
            raise RuntimeError("First call forward()")
        return self._storage.cpu()

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"], **kwargs
    ) -> tuple[
        Float[Tensor, "batch tokens dim"],
        Float[Tensor, "batch n_layers tokens_per_example dim"],
    ]:
        self.reset()
        result = self.model(batch, **kwargs)
        return result, self.activations


@beartype.beartype
@jaxtyped(typechecker=beartype.beartype)
class LabelsWriter:
    """
    LabelsWriter handles writing patch-level segmentation labels to a single binary file.

    Args:
        shards_dir: The shard directory; $SAEV_SCRATCH/saev/shards/<hash>
        md: The Metadata object.

    Attributes:
        labels: The integer patch labels.
        labels_path: Where the integer patch labels are stored.
        md: The dataset metadata.
        has_written: Whether we have written any data to `self.labels`.
    """

    labels: UInt8[np.ndarray, "n_examples n_patches"]
    labels_path: pathlib.Path
    md: Metadata
    has_written: bool

    def __init__(self, shards_dir: pathlib.Path, md: Metadata):
        assert disk.is_shards_dir(shards_dir)
        self.logger = logging.getLogger("labels-writer")
        self.md = md
        self.has_written = False

        # Always create memory-mapped file for labels
        # If nothing is written, it will be deleted in flush()
        self.labels_path = shards_dir / "labels.bin"
        self.labels = np.memmap(
            self.labels_path,
            mode="w+",
            dtype=np.uint8,
            shape=(self.md.n_examples, self.md.content_tokens_per_example),
        )
        self.logger.info("Opened labels file '%s'.", self.labels_path)

    @beartype.beartype
    def write_batch(self, batch_labels: np.ndarray | Tensor, start_idx: int):
        """
        Write a batch of labels to the memory-mapped file.

        Args:
            batch_labels: Array of shape (batch_size, content_tokens_per_example) with uint8 dtype
            start_idx: Starting index in the global labels array
        """
        # Convert to numpy if needed
        if isinstance(batch_labels, torch.Tensor):
            batch_labels = batch_labels.cpu().numpy()

        batch_size = len(batch_labels)
        assert start_idx + batch_size <= self.md.n_examples
        assert batch_labels.shape == (batch_size, self.md.content_tokens_per_example)
        assert batch_labels.dtype == np.uint8

        self.labels[start_idx : start_idx + batch_size] = batch_labels
        self.has_written = True

    def flush(self) -> None:
        """Flush the memory-mapped file to disk if anything was written."""
        if self.has_written:
            self.labels.flush()
            self.logger.info("Flushed labels to '%s'.", self.labels_path)


@jaxtyped(typechecker=beartype.beartype)
class ShardWriter:
    """
    ShardWriter is a stateful object that handles sharded activation writing to disk.

    Args:
        shards_root: The $SAEV_SCRATCH/saev/shards path.
        md: The Metadata object for these shards.

    Attributes:
        shards: The  $SAEV_SCRATCH/saev/shards/<hash>.
        shard:
        acts_path:
        acts:
        filled:
        labels_writer: The LabelsWriter writer.
    """

    shards: pathlib.Path
    shard: int
    acts_path: pathlib.Path
    acts: Float[np.ndarray, "examples_per_shard n_layers all_patches d_model"] | None
    filled: int
    labels_writer: LabelsWriter

    def __init__(self, shards_root: pathlib.Path, md: Metadata):
        assert disk.is_shards_root(shards_root)
        self.md = md

        self.logger = logging.getLogger("shard-writer")

        self.shards_dir = shards_root / md.hash
        self.shards_dir.mkdir(exist_ok=True)

        # builder for shard manifest
        self._shards: ShardInfo = ShardInfo()

        # Always initialize labels writer (it handles non-seg datasets internally)
        self.labels_writer = LabelsWriter(self.shards_dir, md)

        self.shard = -1
        self.acts = None
        self.next_shard()

    def write_batch(
        self,
        activations: Float[Tensor, "batch n_layers all_patches d_model"],
        start_idx: int,
        patch_labels: UInt8[Tensor, "batch n_patches"] | None = None,
    ) -> None:
        """Write a batch of activations and (optionally) patch labels.

        Args:
            activations: Batch of activations to write.
            start_idx: Starting index for this batch.
            patch_labels: Optional patch labels for segmentation datasets.
        """
        batch_size = len(activations)
        end_idx = start_idx + batch_size

        # Write activations (handling sharding)
        offset = self.md.examples_per_shard * self.shard

        if end_idx >= offset + self.md.examples_per_shard:
            # We have run out of space in this mmap'ed file. Let's fill it as much as we can.
            n_fit = offset + self.md.examples_per_shard - start_idx
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
            msg = f"0 <= {start_idx} - {offset} <= {offset} + {self.md.examples_per_shard}"
            assert 0 <= start_idx - offset <= offset + self.md.examples_per_shard, msg
            msg = (
                f"0 <= {end_idx} - {offset} <= {offset} + {self.md.examples_per_shard}"
            )
            assert 0 <= end_idx - offset <= offset + self.md.examples_per_shard, msg
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
                Shard(name=os.path.basename(self.acts_path), n_examples=self.filled)
            )
            self._shards.dump(self.shards_dir)

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
        self.acts_path = self.shards_dir / f"acts{self.shard:06}.bin"
        self.acts = np.memmap(
            self.acts_path, mode="w+", dtype=np.float32, shape=self.md.shard_shape
        )
        self.filled = 0

        self.logger.info("Opened shard '%s'.", self.acts_path)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Shard:
    """
    A single shard entry in shards.json, recording the filename and number of examples.

    Attributes:
        name: The filename of the shard (e.g., "acts000000.bin").
        n_examples: Number of examples stored in this shard.
    """

    name: str
    n_examples: int


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ShardInfo:
    """
    A container for shard metadata as recorded in shards.json.

    Args:
        shards: A list of Shard objects.
    """

    shards: list[Shard] = dataclasses.field(default_factory=list)

    @classmethod
    def load(cls, shards_dir: pathlib.Path) -> "ShardInfo":
        assert disk.is_shards_dir(shards_dir)
        with open(shards_dir / "shards.json") as fd:
            data = json.load(fd)

        return cls([Shard(**entry) for entry in data])

    def dump(self, shards_dir: pathlib.Path) -> None:
        assert disk.is_shards_dir(shards_dir)
        with open(shards_dir / "shards.json", "wb") as fd:
            helpers.dump(self.shards, fd, option=orjson.OPT_INDENT_2)

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
    content_tokens_per_example: int,
    cls_token: bool,
    d_model: int,
    layers: list[int],
    data: datasets.Config,
    batch_size: int,
    n_workers: int,
    max_tokens_per_shard: int,
    shards_root: pathlib.Path,
    device: str,
    pixel_agg: PixelAgg = PixelAgg.MAJORITY,
) -> pathlib.Path:
    """
    Args:
        family: Transformer family (dinov2, dinov3, clip, etc).
        ckpt: Transformer ckpt (hf-hub:imageomics/bioclip2, etc).
        content_tokens_per_example: Number of content tokens per example.
        cls_token: Whether the transformer has a [CLS] token.
        d_model: Hidden dimension of transformer.
        layers: The layers to record activations for.
        data: Config for the particular (image) dataset to load.
        batch_size: Batch size for the dataset.
        n_workers: Number of workers for loading examples fromm the dataset.
        max_tokens_per_shard: Maximum number of tokens per disk shard.
        pixel_agg: Optional method for aggregating segmentation label pixels.
        shards_root: Where to save shards. Should end with 'shards'. See [disk-layout.md](../../developers/disk-layout.md); this is $SAEV_SCRATCH/saev/shards.
        device: Device for doing the computation.

    Returns:
        Path to the shards directory.
    """
    from saev import helpers
    from saev.data import models

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("worker_fn")

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA device available, using CPU.")
        device = "cpu"

    assert shards_root.name == "shards"

    model_cls = models.load_model_cls(family)
    model_instance = model_cls(ckpt).to(device)
    model = RecordedTransformer(
        model_instance, content_tokens_per_example, cls_token, layers
    )

    img_tr, sample_tr = model_cls.make_transforms(ckpt, content_tokens_per_example)

    seg_tr = None
    if datasets.is_img_seg_dataset(data):
        # For image segmentation datasets, create a transform that converts pixels to patches
        # Use make_resize with NEAREST interpolation for segmentation masks
        seg_resize_tr = model_cls.make_resize(
            ckpt, content_tokens_per_example, scale=1.0, resample=Image.NEAREST
        )

        def seg_to_patches(seg):
            """Transform that resizes segmentation and converts to patch labels."""

            # Convert to patch labels
            return pixel_to_patch_labels(
                seg_resize_tr(seg),
                content_tokens_per_example,
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

    n_batches = math.ceil(data.n_examples / batch_size)
    logger.info("Dumping %d batches of %d examples.", n_batches, batch_size)

    model = model.to(device)

    md = Metadata(
        family=family,
        ckpt=ckpt,
        layers=tuple(layers),
        content_tokens_per_example=content_tokens_per_example,
        cls_token=cls_token,
        d_model=d_model,
        n_examples=data.n_examples,
        max_tokens_per_shard=max_tokens_per_shard,
        data=base64.b64encode(pickle.dumps(data)).decode("utf8"),
        dataset=data.root,
        pixel_agg=pixel_agg,
    )
    md.dump(shards_root)

    # Use context manager for proper cleanup
    with ShardWriter(shards_root, md) as writer:
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
                    assert patch_labels.shape == (
                        len(cache),
                        content_tokens_per_example,
                    )
                else:
                    logger.debug(f"No patch_labels in batch. Keys: {batch.keys()}")

                writer.write_batch(cache, i, patch_labels=patch_labels)

                i += len(cache)

    return shards_root / md.hash


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
    pixel_agg: PixelAgg = PixelAgg.MAJORITY,
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

    if pixel_agg is PixelAgg.MAJORITY:
        # Take the most common label in each patch
        patch_labels = counts.argmax(dim=1)
    elif pixel_agg is PixelAgg.PREFER_FG:
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
@dataclasses.dataclass(frozen=True, kw_only=True)
class Index:
    """
    Attributes:
        idx: The index of the activation.
        example_idx: The index of the original example (image, audio clip etc).
        content_token_idx: The token's index within an example's content. -1 for all special tokens.
        shard_idx: The shard index.
        example_idx_in_shard: The example index along the examples axis in a shard.
        token_idx_in_shard: The token index along the tokens axis in a shard.
    """

    idx: int
    example_idx: int
    content_token_idx: int
    shard_idx: int
    example_idx_in_shard: int
    layer_idx_in_shard: int
    token_idx_in_shard: int


@beartype.beartype
class IndexMap:
    """
    Attributes:
        md: Metadata
        tokens: Which subset of tokens to load.
        layer: Which layer to load.
        layer_idx_lookup: The lookup from a transformer layer to the layer idx in the shard.
    """

    md: Metadata
    tokens: tp.Literal["special", "content", "all"]
    layer: int
    layer_idx_lookup: dict[int, int]

    def __init__(
        self,
        md: Metadata,
        tokens: tp.Literal["special", "content", "all"],
        layer: int | tp.Literal["all"],
    ):
        if tokens == "special":
            assert md.cls_token

        self.md = md
        self.tokens = tokens
        self.layer = layer

        if isinstance(layer, int):
            err_msg = f"No matche for layer; {layer} not in {md.layers}."
            assert layer in md.layers, err_msg

        self.layer_idx_lookup = {layer: i for i, layer in enumerate(md.layers)}

    def from_global(self, idx: int | np.int_) -> Index:
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        match (self.tokens, self.layer):
            case ("special", int()):
                # [CLS] tokens only right now
                example_idx = idx
                shard_idx = idx // self.md.examples_per_shard
                example_idx_in_shard = idx // self.md.examples_per_shard
                return Index(
                    idx=idx,
                    example_idx=example_idx,
                    content_token_idx=-1,
                    shard_idx=shard_idx,
                    example_idx_in_shard=example_idx_in_shard,
                    layer_idx_in_shard=self.layer_idx_lookup[self.layer],
                    token_idx_in_shard=0,
                )
            case ("content", int()):
                example_idx = idx // self.md.content_tokens_per_example
                content_token_idx = idx % self.md.content_tokens_per_example
                shard_idx = idx // (
                    self.md.examples_per_shard * self.md.content_tokens_per_example
                )
                example_idx_in_shard = (
                    idx
                    % (self.md.examples_per_shard * self.md.content_tokens_per_example)
                    // self.md.content_tokens_per_example
                )
                token_idx_in_shard = (
                    idx
                    % (self.md.examples_per_shard * self.md.content_tokens_per_example)
                    % self.md.content_tokens_per_example
                    + self.md.cls_token
                )
                return Index(
                    idx=idx,
                    example_idx=example_idx,
                    content_token_idx=content_token_idx,
                    shard_idx=shard_idx,
                    example_idx_in_shard=example_idx_in_shard,
                    layer_idx_in_shard=self.layer_idx_lookup[self.layer],
                    token_idx_in_shard=token_idx_in_shard,
                )

            case _:
                tp.assert_never((self.tokens, self.layer))

    def __len__(self) -> int:
        """
        Dataset length depends on `patches` and `layer`.
        """
        match (self.tokens, self.layer):
            case ("special", "all"):
                # Return a CLS token from a random example and random layer.
                return self.md.n_examples * len(self.md.layers)
            case ("special", int()):
                # Return a CLS token from a random example and fixed layer.
                return self.md.n_examples
            case ("content", int()):
                # Return a patch from a random example, fixed layer, and random patch.
                return self.md.n_examples * self.md.content_tokens_per_example
            case ("content", "all"):
                # Return a patch from a random example, random layer and random patch.
                return (
                    self.md.n_examples
                    * len(self.md.layers)
                    * self.md.content_tokens_per_example
                )
            case ("all", int()):
                # Return a token from a random example, fixed layer, and random token (including special).
                return self.md.n_examples * self.md.tokens_per_example
            case ("all", "all"):
                # Return a token from a random example, random layer and random token (including special).
                return (
                    self.md.n_examples
                    * len(self.md.layers)
                    * self.md.tokens_per_example
                )
            case _:
                tp.assert_never((self.cfg.tokens, self.cfg.layer))
