# src/saev/data/iterable.py
import collections.abc
import dataclasses
import logging
import os
import queue
import random
import threading
import typing

import beartype
import numpy as np
import torch
import torch.multiprocessing as mp
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from . import utils, writers


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    shard_root: str = os.path.join(".", "shards")
    """Directory with .bin shards and a metadata.json file."""
    patches: typing.Literal["cls", "patches", "meanpool"] = "patches"
    """Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'patches' indicates it will return all patches. 'meanpool' returns the mean of all image patches."""
    layer: int | typing.Literal["all", "meanpool"] = -2
    """Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer, and ``"meanpool"`` averages activations across layers."""
    clamp: float = 1e5
    """Maximum value for activations; activations will be clamped to within [-clamp, clamp]`."""
    batch_size: int = 1024 * 16
    """Batch size."""
    n_threads: int = 4
    """Number of dataloading threads."""
    seed: int = 17
    """Random seed."""
    ring_buffer_slots: int = 8
    """Number of batches to queue in the shared-memory ring buffer. Higher values add latency but improve resilience to brief stalls."""


def _io_worker(
    worker_id: int,
    cfg: Config,
    metadata: writers.Metadata,
    layer_index: int,
    work_queue: mp.Queue,
    ready_queue: mp.Queue,
    stop_event: threading.Event,
):
    """
    Pulls work items from the queue, loads data, and pushes it to the ready queue.
    Work item is a tuple: (shard_idx, list_of_global_indices).
    """
    logger.info(f"I/O worker {worker_id} started.")
    # Calculate shard layout constants
    n_patches_per_img = metadata.n_patches_per_img
    n_layers = len(metadata.layers)
    n_imgs_per_shard = (
        metadata.n_patches_per_shard // n_layers // (n_patches_per_img + 1)
    )
    n_examples_per_shard = n_imgs_per_shard * n_patches_per_img
    shape = (
        n_imgs_per_shard,
        n_layers,
        n_patches_per_img + 1,
        metadata.d_vit,
    )

    while not stop_event.is_set():
        try:
            item = work_queue.get(timeout=0.1)
            if item is None:  # Poison pill
                break

            shard_idx, indices_to_load = item
            acts_fpath = os.path.join(cfg.shard_root, f"acts{shard_idx:06}.bin")
            mmap = np.memmap(acts_fpath, mode="r", dtype=np.float32, shape=shape)

            # Pre-slice the mmap to only the layer we need, skipping the CLS token
            layer_acts = mmap[:, layer_index, 1:, :]

            acts_chunk = np.zeros(
                (len(indices_to_load), metadata.d_vit), dtype=np.float32
            )
            img_i_chunk = np.zeros(len(indices_to_load), dtype=np.int64)
            patch_i_chunk = np.zeros(len(indices_to_load), dtype=np.int64)

            for i, global_idx in enumerate(indices_to_load):
                pos_in_shard = global_idx % n_examples_per_shard
                img_idx_in_shard = pos_in_shard // n_patches_per_img
                patch_idx_in_img = pos_in_shard % n_patches_per_img

                acts_chunk[i] = layer_acts[img_idx_in_shard, patch_idx_in_img, :]
                img_i_chunk[i] = global_idx // n_patches_per_img
                patch_i_chunk[i] = patch_idx_in_img

            ready_queue.put((acts_chunk, img_i_chunk, patch_i_chunk))

        except queue.Empty:
            continue
        except Exception:
            logger.exception(f"Error in worker {worker_id}")
            break
    logger.info(f"I/O worker {worker_id} finished.")


def _manager_main(
    cfg: Config,
    metadata: writers.Metadata,
    act_buffer: utils.RingBuffer,
    image_i_buffer: utils.RingBuffer,
    patch_i_buffer: utils.RingBuffer,
    stop_event: mp.Event,
    total_examples: int,
):
    """
    The main function for the data loader manager process.
    """

    logger = logging.getLogger("iterable.manager")

    try:
        if cfg.patches != "patches" or not isinstance(cfg.layer, int):
            raise NotImplementedError(
                "High-throughput loader only supports `patches` and fixed `layer` mode for now."
            )

        # 1. SETUP & GLOBAL SHUFFLE
        logger.info("Manager process started. Generating shuffled index...")
        rng = np.random.default_rng(cfg.seed)
        indices = rng.permutation(total_examples)

        # 2. GROUP INDICES BY SHARD
        n_patches_per_img = metadata.n_patches_per_img
        n_layers = len(metadata.layers)
        n_imgs_per_shard = (
            metadata.n_patches_per_shard // n_layers // (n_patches_per_img + 1)
        )
        n_examples_per_shard = n_imgs_per_shard * n_patches_per_img

        indices_by_shard = [[] for _ in range(metadata.n_shards)]
        for idx in indices:
            shard_idx = idx // n_examples_per_shard
            indices_by_shard[shard_idx].append(idx)
        logger.info(
            "Grouped %d indices into %d shards.", len(indices), metadata.n_shards
        )

        # 3. SETUP WORK QUEUE & I/O THREADS
        ctx = mp.get_context()
        work_queue = ctx.Queue()
        ready_queue = ctx.Queue(maxsize=cfg.n_threads * 2)

        work_items = list(enumerate(indices_by_shard))
        random.shuffle(work_items)  # Shuffle shard order to balance I/O at start

        for shard_idx, indices_for_shard in work_items:
            if indices_for_shard:
                # Further chunk the work to prevent one thread getting a huge list
                for i in range(0, len(indices_for_shard), 4096):
                    chunk = indices_for_shard[i : i + 4096]
                    work_queue.put((shard_idx, chunk))

        for _ in range(cfg.n_threads):
            work_queue.put(None)

        assert cfg.layer in metadata.layers, (
            f"Layer {cfg.layer} not in {metadata.layers}"
        )
        layer_index = metadata.layers.index(cfg.layer)

        threads = []
        thread_stop_event = threading.Event()
        for i in range(cfg.n_threads):
            thread = threading.Thread(
                target=_io_worker,
                args=(
                    i,
                    cfg,
                    metadata,
                    layer_index,
                    work_queue,
                    ready_queue,
                    thread_stop_event,
                ),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        logger.info("Launched %d I/O threads.", cfg.n_threads)

        # 4. BATCHING LOOP
        buffer_act, buffer_img, buffer_patch = [], [], []
        examples_batched = 0
        while examples_batched < total_examples:
            if stop_event.is_set():
                break
            try:
                (acts_chunk, img_i_chunk, patch_i_chunk) = ready_queue.get(timeout=1.0)
                buffer_act.append(acts_chunk)
                buffer_img.append(img_i_chunk)
                buffer_patch.append(patch_i_chunk)

                while sum(len(b) for b in buffer_act) >= cfg.batch_size:
                    # Concatenate chunks and slice off a batch
                    acts = np.concatenate(buffer_act)
                    imgs = np.concatenate(buffer_img)
                    pats = np.concatenate(buffer_patch)

                    batch_act = torch.from_numpy(acts[: cfg.batch_size])
                    batch_img = torch.from_numpy(imgs[: cfg.batch_size])
                    batch_patch = torch.from_numpy(pats[: cfg.batch_size])

                    # Put into shared memory ring buffers
                    act_buffer.put(batch_act.clamp(-cfg.clamp, cfg.clamp))
                    image_i_buffer.put(batch_img)
                    patch_i_buffer.put(batch_patch)

                    # Keep the remainder
                    buffer_act = (
                        [acts[cfg.batch_size :]] if len(acts) > cfg.batch_size else []
                    )
                    buffer_img = (
                        [imgs[cfg.batch_size :]] if len(imgs) > cfg.batch_size else []
                    )
                    buffer_patch = (
                        [pats[cfg.batch_size :]] if len(pats) > cfg.batch_size else []
                    )
                    examples_batched += cfg.batch_size

            except queue.Empty:
                # Check if workers are still alive
                if not any(t.is_alive() for t in threads):
                    logger.warning("All I/O threads exited prematurely.")
                    break
                continue

    except Exception:
        logger.exception("Fatal error in manager process")
    finally:
        # 5. CLEANUP
        logger.info("Manager process shutting down...")
        thread_stop_event.set()
        while not work_queue.empty():
            work_queue.get_nowait()
        for t in threads:
            t.join(timeout=2.0)
        logger.info("Manager process finished.")


@beartype.beartype
class DataLoader:
    """
    High-throughput streaming loader that deterministically shuffles data from disk shards.
    """

    @jaxtyped(typechecker=beartype.beartype)
    class Example(typing.TypedDict):
        """Individual example."""

        act: Float[Tensor, "batch d_vit"]
        image_i: Int[Tensor, " batch"]
        patch_i: Int[Tensor, " batch"]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not os.path.isdir(self.cfg.shard_root):
            raise RuntimeError(f"Activations are not saved at '{self.cfg.shard_root}'.")

        metadata_fpath = os.path.join(self.cfg.shard_root, "metadata.json")
        self.metadata = writers.Metadata.load(metadata_fpath)

        self.logger = logging.getLogger("iterable.DataLoader")
        self.ctx = mp.get_context()
        self.manager_proc = None
        self.act_buffer = None
        self.image_i_buffer = None
        self.patch_i_buffer = None
        self.stop_event = None
        self._total_examples = self._calculate_len()

    def _start_manager(self):
        if self.manager_proc and self.manager_proc.is_alive():
            return

        self.logger.info("Starting manager process...")
        d_vit = self.metadata.d_vit

        # Create the shared-memory ring buffers
        self.act_buffer = utils.RingBuffer(
            self.cfg.ring_buffer_slots, (self.cfg.batch_size, d_vit), torch.float32
        )
        self.image_i_buffer = utils.RingBuffer(
            self.cfg.ring_buffer_slots, (self.cfg.batch_size,), torch.int64
        )
        self.patch_i_buffer = utils.RingBuffer(
            self.cfg.ring_buffer_slots, (self.cfg.batch_size,), torch.int64
        )

        self.stop_event = self.ctx.Event()

        self.manager_proc = self.ctx.Process(
            target=_manager_main,
            args=(
                self.cfg,
                self.metadata,
                self.act_buffer,
                self.image_i_buffer,
                self.patch_i_buffer,
                self.stop_event,
                self._total_examples,
            ),
            daemon=True,
        )
        self.manager_proc.start()

    def __iter__(self) -> collections.abc.Iterable[Example]:
        """Yields batches."""
        self._start_manager()
        n_batches = (
            self._total_examples + self.cfg.batch_size - 1
        ) // self.cfg.batch_size
        try:
            for i in range(n_batches):
                if not self.manager_proc.is_alive():
                    raise RuntimeError(
                        f"Manager process died unexpectedly after {i}/{n_batches} batches."
                    )
                act = self.act_buffer.get()
                image_i = self.image_i_buffer.get()
                patch_i = self.patch_i_buffer.get()
                yield self.Example(act=act, image_i=image_i, patch_i=patch_i)
        finally:
            self.shutdown()

    def shutdown(self):
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()
        if self.manager_proc and self.manager_proc.is_alive():
            self.manager_proc.join(timeout=5.0)
            if self.manager_proc.is_alive():
                logger.warning("Manager process did not shut down cleanly, killing.")
                self.manager_proc.kill()

        # Free shared memory
        for buf in [self.act_buffer, self.image_i_buffer, self.patch_i_buffer]:
            if buf:
                buf.close()

        self.manager_proc = None
        self.act_buffer = self.image_i_buffer = self.patch_i_buffer = None
        self.stop_event = None

    def __del__(self):
        self.shutdown()

    def _calculate_len(self) -> int:
        """Helper to calculate total number of examples based on config."""
        match (self.cfg.patches, self.cfg.layer):
            case ("cls", "all"):
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("cls", int()):
                return self.metadata.n_imgs
            case ("cls", "meanpool"):
                return self.metadata.n_imgs
            case ("meanpool", "all"):
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("meanpool", int()):
                return self.metadata.n_imgs
            case ("meanpool", "meanpool"):
                return self.metadata.n_imgs
            case ("patches", int()):
                return self.metadata.n_imgs * self.metadata.n_patches_per_img
            case ("patches", "meanpool"):
                return self.metadata.n_imgs * self.metadata.n_patches_per_img
            case ("patches", "all"):
                return (
                    self.metadata.n_imgs
                    * len(self.metadata.layers)
                    * self.metadata.n_patches_per_img
                )
            case _:
                typing.assert_never((self.cfg.patches, self.cfg.layer))

    def __len__(self) -> int:
        """Returns the number of batches in an epoch."""
        return (self._total_examples + self.cfg.batch_size - 1) // self.cfg.batch_size
