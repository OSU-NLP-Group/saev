# src/saev/data/ordered.py
"""
Ordered (sequential) dataloader for activation data.

This module provides a high-throughput dataloader that reads activation data from disk shards in sequential order, without shuffling. The implementation uses a single-threaded manager process to ensure data is delivered in the exact order it appears on disk.

Patch labels are provided if there is a labels.bin file on disk.

See the design decisions in src/saev/data/performance.md.

Usage:
    >>> cfg = Config(shards="./shards", layer=13, batch_size=4096)
    >>> dataloader = DataLoader(cfg)
    >>> for batch in dataloader:
    ...     activations = batch["act"]  # [batch_size, d_model]
    ...     image_indices = batch["image_i"]  # [batch_size]
    ...     patch_indices = batch["patch_i"]  # [batch_size]
    ...     patch_labels = batch["patch_labels"]  # [batch_size]
"""

import collections.abc
import dataclasses
import logging
import math
import os
import pathlib
import queue
import time
import traceback
import typing
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event

import beartype
import numpy as np
import torch
import torch.multiprocessing as mp
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from . import shards


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for loading ordered (non-shuffled) activation data from disk."""

    shards: pathlib.Path = pathlib.Path("$SAEV_SCRATCH/saev/shards/abcdefg")
    """Directory with .bin shards and a metadata.json file."""
    patches: typing.Literal["cls", "image", "all"] = "image"
    """Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches."""
    layer: int | typing.Literal["all"] = -2
    """Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer."""
    batch_size: int = 1024 * 16
    """Batch size."""
    batch_timeout_s: float = 30.0
    """How long to wait for at least one batch."""
    drop_last: bool = False
    """Whether to drop the last batch if it's smaller than the others."""
    buffer_size: int = 64
    """Number of batches to queue in the shared-memory ring buffer. Higher values add latency but improve resilience to brief stalls."""
    debug: bool = False
    """Whether the dataloader process should log debug messages."""
    log_every_s: float = 30.0
    """How frequently the dataloader process should log (debug) performance messages."""


@beartype.beartype
class ImageOutOfBoundsError(Exception):
    def __init__(self, metadata: shards.Metadata, i: int):
        self.metadata = metadata
        self.i = i

    @property
    def message(self) -> str:
        return f"Metadata says there are {self.metadata.n_imgs} images, but we found image {self.i}."


@beartype.beartype
def _manager_main(
    cfg: Config,
    metadata: shards.Metadata,
    batch_queue: Queue[dict[str, torch.Tensor]],
    stop_event: Event,
    err_queue: Queue[tuple[str, str]],
):
    """
    The main function for the data loader manager process.
    Reads data sequentially and pushes batches to the queue.
    """
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, force=True)
    logger = logging.getLogger("ordered.manager")
    logger.info(
        "Manager process started (debug=%s, logging=%s)",
        cfg.debug,
        logging.getLevelName(logger.getEffectiveLevel()),
    )

    # 0. PRE-CONDITIONS
    if cfg.patches != "image" or not isinstance(cfg.layer, int):
        raise NotImplementedError(
            "High-throughput loader only supports `image` and fixed `layer` mode for now."
        )

    assert cfg.layer in metadata.layers, f"Layer {cfg.layer} not in {metadata.layers}"

    try:
        # Load shard info to get actual distribution
        shard_info = shards.ShardInfo.load(cfg.shards)
        layer_i = metadata.layers.index(cfg.layer)

        # Check if labels.bin exists
        labels_path = os.path.join(cfg.shards, "labels.bin")
        labels_mmap = None
        if os.path.exists(labels_path):
            labels_mmap = np.memmap(
                labels_path,
                mode="r",
                dtype=np.uint8,
                shape=(metadata.n_imgs, metadata.n_patches_per_img),
            )
            logger.debug("Found labels.bin, will include patch labels in batches")

        # Calculate cumulative image offsets for each shard
        cumulative_imgs = [0]
        for shard in shard_info:
            cumulative_imgs.append(cumulative_imgs[-1] + shard.n_imgs)

        # Calculate cumulative sample offsets for each shard
        cumulative_samples = [0]
        for shard in shard_info:
            cumulative_samples.append(
                cumulative_samples[-1] + shard.n_imgs * metadata.n_patches_per_img
            )

        # Calculate total number of samples
        total_samples = metadata.n_imgs * metadata.n_patches_per_img

        logger.debug("Found %d samples.", total_samples)

        # Process batches in order
        current_idx = 0
        while current_idx < total_samples and not stop_event.is_set():
            batch_end_idx = min(current_idx + cfg.batch_size, total_samples)

            # Collect batch activations and metadata
            batch_acts = []
            batch_image_is = []
            batch_patch_is = []
            batch_patch_labels: list[int] | None = (
                [] if labels_mmap is not None else None
            )

            # Process samples in this batch range
            for idx in range(current_idx, batch_end_idx):
                # Calculate which image and patch this index corresponds to
                global_image_i = idx // metadata.n_patches_per_img
                patch_i = idx % metadata.n_patches_per_img

                # Find which shard contains this image
                shard_i = None
                for i in range(len(cumulative_imgs) - 1):
                    if cumulative_imgs[i] <= global_image_i < cumulative_imgs[i + 1]:
                        shard_i = i
                        break

                if shard_i is None:
                    continue

                # Get local image index within the shard
                local_image_i = global_image_i - cumulative_imgs[shard_i]

                if local_image_i >= shard_info[shard_i].n_imgs:
                    continue

                if global_image_i >= metadata.n_imgs:
                    err = ImageOutOfBoundsError(metadata, global_image_i)
                    logger.warning(err.message)
                    raise err

                # Load activation from the appropriate shard
                fname = f"acts{shard_i:06}.bin"
                acts_fpath = os.path.join(cfg.shards, fname)

                # Open mmap for this shard if needed
                mmap = np.memmap(
                    acts_fpath, mode="r", dtype=np.float32, shape=metadata.shard_shape
                )

                # Get the activation
                patch_idx_with_cls = patch_i + int(metadata.cls_token)
                act = torch.from_numpy(
                    mmap[local_image_i, layer_i, patch_idx_with_cls].copy()
                )

                batch_acts.append(act)
                batch_image_is.append(global_image_i)
                batch_patch_is.append(patch_i)

                # Add patch label if available
                if labels_mmap is not None and batch_patch_labels is not None:
                    label = labels_mmap[global_image_i, patch_i]
                    batch_patch_labels.append(label)

            # Send batch if we have data
            if batch_acts:
                batch = {
                    "act": torch.stack(batch_acts),
                    "image_i": torch.tensor(batch_image_is, dtype=torch.long),
                    "patch_i": torch.tensor(batch_patch_is, dtype=torch.long),
                }

                # Add labels if available
                if labels_mmap is not None:
                    batch["patch_labels"] = torch.tensor(
                        batch_patch_labels, dtype=torch.long
                    )

                batch_queue.put(batch)
                logger.debug(f"Sent batch with {len(batch_acts)} samples")

            current_idx = batch_end_idx

    except Exception:
        logger.exception("Fatal error in manager process")
        err_queue.put(("manager", traceback.format_exc()))
    finally:
        logger.info("Manager process finished.")

    logger.info("Manager process sleeping.")
    # Sleep a little longer, otherwise the tensors will be released and garbage collected, then we get a memory error in the parent process..
    time.sleep(60.0)
    logger.info("Manager process finished.")


@beartype.beartype
class DataLoader:
    """
    High-throughput streaming loader that reads data from disk shards in order (no shuffling).
    """

    @jaxtyped(typechecker=beartype.beartype)
    class ExampleBatch(typing.TypedDict, total=False):
        """Individual example."""

        act: Float[Tensor, "batch d_model"]
        image_i: Int[Tensor, " batch"]
        patch_i: Int[Tensor, " batch"]
        # Optional, only present if labels.bin exists
        patch_labels: Int[Tensor, " batch"]

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

        self.logger = logging.getLogger("ordered.DataLoader")
        self.ctx = mp.get_context()
        self.manager_proc = None
        self.batch_queue = None
        self.stop_event = None
        self._n_samples = self._calculate_n_samples()
        self.logger.info(
            "Initialized ordered.DataLoader with %d samples. (debug=%s)",
            self.n_samples,
            self.cfg.debug,
        )

    @property
    def n_batches(self) -> int:
        return len(self)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def batch_size(self) -> int:
        return self.cfg.batch_size

    @property
    def drop_last(self) -> int:
        return self.cfg.drop_last

    def _start_manager(self):
        # Always shutdown existing manager to ensure fresh start
        if self.manager_proc and self.manager_proc.is_alive():
            self.logger.info("Shutting down existing manager process.")
            self.shutdown()

        self.logger.info("Starting manager process.")

        # Create the batch queue
        self.batch_queue = self.ctx.Queue(maxsize=self.cfg.buffer_size)
        self.stop_event = self.ctx.Event()
        self.err_queue = self.ctx.Queue(maxsize=2)  # Manager + main process

        self.manager_proc = self.ctx.Process(
            target=_manager_main,
            args=(
                self.cfg,
                self.metadata,
                self.batch_queue,
                self.stop_event,
                self.err_queue,
            ),
            daemon=True,
        )
        self.manager_proc.start()

    def __iter__(self) -> collections.abc.Iterable[ExampleBatch]:
        """Yields batches in order."""
        self._start_manager()
        n = 0

        try:
            while n < self.n_samples:
                if not self.err_queue.empty():
                    who, tb = self.err_queue.get_nowait()
                    raise RuntimeError(f"{who} crashed:\n{tb}")

                try:
                    batch = self.batch_queue.get(timeout=self.cfg.batch_timeout_s)
                    actual_batch_size = batch["act"].shape[0]

                    # Handle drop_last
                    if (
                        self.cfg.drop_last
                        and actual_batch_size < self.cfg.batch_size
                        and n + actual_batch_size >= self.n_samples
                    ):
                        break

                    n += actual_batch_size
                    yield self.ExampleBatch(**batch)
                    continue
                except queue.Empty:
                    self.logger.info(
                        "Did not get a batch from manager process in %.1fs seconds.",
                        self.cfg.batch_timeout_s,
                    )
                except FileNotFoundError:
                    self.logger.info("Manager process (probably) closed.")
                    continue

                # If we don't continue, then we should check on the manager process.
                if not self.manager_proc.is_alive():
                    raise RuntimeError(
                        f"Manager process died unexpectedly after {n}/{self.n_samples} samples."
                    )

        finally:
            self.shutdown()

    def shutdown(self):
        if (
            hasattr(self, "stop_event")
            and self.stop_event
            and not self.stop_event.is_set()
        ):
            self.stop_event.set()

        if (
            hasattr(self, "manager_proc")
            and self.manager_proc
            and self.manager_proc.is_alive()
        ):
            self.manager_proc.join(timeout=5.0)
            if self.manager_proc.is_alive():
                self.logger.warning(
                    "Manager process did not shut down cleanly, killing."
                )
                self.manager_proc.kill()

        self.manager_proc = None
        self.batch_queue = None
        self.stop_event = None

    def __del__(self):
        self.shutdown()

    def _calculate_n_samples(self) -> int:
        """Helper to calculate total number of examples based on config."""
        match (self.cfg.patches, self.cfg.layer):
            case ("cls", "all"):
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("cls", int()):
                return self.metadata.n_imgs
            case ("image", int()):
                return self.metadata.n_imgs * self.metadata.n_patches_per_img
            case ("image", "all"):
                return (
                    self.metadata.n_imgs
                    * len(self.metadata.layers)
                    * self.metadata.n_patches_per_img
                )
            case _:
                typing.assert_never((self.cfg.patches, self.cfg.layer))

    def __len__(self) -> int:
        """Returns the number of batches in an epoch."""
        if self.cfg.drop_last:
            return self.n_samples // self.cfg.batch_size
        else:
            return math.ceil(self.n_samples / self.cfg.batch_size)
