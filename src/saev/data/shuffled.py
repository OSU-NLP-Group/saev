# src/saev/data/shuffled.py
import collections.abc
import dataclasses
import logging
import math
import os
import queue
import threading
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

from saev import helpers

from . import buffers, writers


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for loading shuffled activation data from disk."""

    shard_root: str = os.path.join(".", "shards")
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
    n_threads: int = 4
    """Number of dataloading threads."""
    buffer_size: int = 64
    """Number of batches to queue in the shared-memory ring buffer. Higher values add latency but improve resilience to brief stalls."""
    seed: int = 17
    """Random seed."""
    debug: bool = False
    """Whether the dataloader process should log debug messages."""


@beartype.beartype
class ImageOutOfBoundsError(Exception):
    def __init__(self, metadata: writers.Metadata, i: int):
        self.metadata = metadata
        self.i = i

    @property
    def message(self) -> str:
        return f"Metadata says there are {self.metadata.n_imgs} images, but we found image {self.i}."


@jaxtyped(typechecker=beartype.beartype)
def _io_worker(
    worker_id: int,
    cfg: Config,
    metadata: writers.Metadata,
    work_queue: queue.Queue[int | None],
    reservoir: buffers.ReservoirBuffer,
    stop_event: threading.Event,
    err_queue: Queue[tuple[str, str]],
):
    """
    Pulls work items from the queue, loads data, and pushes it to the ready queue.
    Work item is a tuple: (shard_idx, list_of_global_indices).

    See https://github.com/beartype/beartype/issues/397 for an explanation of why we use multiprocessing.queues.Queue for the type hint.
    """
    logger = logging.getLogger(f"shuffled.worker{worker_id}")
    logger.info(f"I/O worker {worker_id} started.")

    layer_i = metadata.layers.index(cfg.layer)
    shard_info = writers.ShardInfo.load(cfg.shard_root)

    # Pre-conditions
    assert cfg.patches == "image"
    assert isinstance(cfg.layer, int)

    while not stop_event.is_set():
        try:
            shard_i = work_queue.get(timeout=0.1)
            if shard_i is None:  # Poison pill
                logger.debug("Got 'None' from work_queue; exiting.")
                break

            fname = f"acts{shard_i:06}.bin"
            logger.info("Opening %s.", fname)

            img_i_offset = shard_i * metadata.n_imgs_per_shard

            acts_fpath = os.path.join(cfg.shard_root, fname)
            mmap = np.memmap(
                acts_fpath, mode="r", dtype=np.float32, shape=metadata.shard_shape
            )

            # Only iterate over the actual number of images in this shard
            for start, end in helpers.batched_idx(shard_info[shard_i].n_imgs, 64):
                for p in range(metadata.n_patches_per_img):
                    patch_i = p + int(metadata.cls_token)
                    acts = torch.from_numpy(mmap[start:end, layer_i, patch_i])

                    last_img_i = img_i_offset + (end - 1)
                    if last_img_i >= metadata.n_imgs:
                        err = ImageOutOfBoundsError(metadata, last_img_i)
                        logger.warning(err.message)
                        raise err

                    metas = [
                        {"image_i": img_i_offset + i, "patch_i": p}
                        for i in range(start, end)
                    ]

                    reservoir.put(acts, metas)
        except queue.Empty:
            # Wait 0.1 seconds for new data.
            time.sleep(0.1)
            continue
        except Exception:
            logger.exception("Error in worker.")
            err_queue.put((f"worker{worker_id}", traceback.format_exc()))
            break
    logger.info("Worker finished.")


@beartype.beartype
def _manager_main(
    cfg: Config,
    metadata: writers.Metadata,
    reservoir: buffers.ReservoirBuffer,
    stop_event: Event,
    err_queue: Queue[tuple[str, str]],
):
    """
    The main function for the data loader manager process.
    """
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger("shuffled.manager")
    logger.info("Manager process started.")

    # 0. PRE-CONDITIONS
    if cfg.patches != "image" or not isinstance(cfg.layer, int):
        raise NotImplementedError(
            "High-throughput loader only supports `image` and fixed `layer` mode for now."
        )

    assert cfg.layer in metadata.layers, f"Layer {cfg.layer} not in {metadata.layers}"

    # 1. GLOBAL SHUFFLE
    logger.info("Shuffling shards.")
    rng = np.random.default_rng(cfg.seed)
    work_items = rng.permutation(metadata.n_shards)
    logger.debug("First 10 shards: %s", work_items[:10])

    try:
        # 2. SETUP WORK QUEUE & I/O THREADS
        work_queue = queue.Queue()

        for shard_i in work_items:
            work_queue.put(shard_i)

        # Stop objects.
        for _ in range(cfg.n_threads):
            work_queue.put(None)

        threads = []
        thread_stop_event = threading.Event()
        for i in range(cfg.n_threads):
            args = (
                i,
                cfg,
                metadata,
                work_queue,
                reservoir,
                thread_stop_event,
                err_queue,
            )
            thread = threading.Thread(target=_io_worker, args=args, daemon=True)
            thread.start()
            threads.append(thread)
        logger.info("Launched %d I/O threads.", cfg.n_threads)

        # 4. WAIT
        while any(t.is_alive() for t in threads):
            time.sleep(1.0)

    except Exception:
        logger.exception("Fatal error in manager process")
        err_queue.put(("manager", traceback.format_exc()))
    finally:
        # 5. CLEANUP
        logger.info("Manager process shutting down...")
        thread_stop_event.set()
        while not work_queue.empty():
            work_queue.get_nowait()
        for t in threads:
            t.join(timeout=10.0)
        logger.info("Manager process finished.")


@beartype.beartype
class DataLoader:
    """
    High-throughput streaming loader that deterministically shuffles data from disk shards.
    """

    @jaxtyped(typechecker=beartype.beartype)
    class ExampleBatch(typing.TypedDict):
        """Individual example."""

        act: Float[Tensor, "batch d_vit"]
        image_i: Int[Tensor, " batch"]
        patch_i: Int[Tensor, " batch"]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not os.path.isdir(self.cfg.shard_root):
            raise RuntimeError(f"Activations are not saved at '{self.cfg.shard_root}'.")

        self.metadata = writers.Metadata.load(self.cfg.shard_root)

        self.logger = logging.getLogger("shuffled.DataLoader")
        self.ctx = mp.get_context()
        self.manager_proc = None
        self.reservoir = None
        self.stop_event = None
        self._n_samples = self._calculate_n_samples()

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
        if self.manager_proc and self.manager_proc.is_alive():
            return

        self.logger.info("Starting manager process.")

        # Create the shared-memory buffers
        self.reservoir = buffers.ReservoirBuffer(
            self.cfg.buffer_size * self.cfg.batch_size,
            (self.metadata.d_vit,),
            dtype=torch.float32,
            collate_fn=torch.utils.data.default_collate,
        )
        self.stop_event = self.ctx.Event()
        self.err_queue = self.ctx.Queue(maxsize=self.cfg.n_threads + 1)

        self.manager_proc = self.ctx.Process(
            target=_manager_main,
            args=(
                self.cfg,
                self.metadata,
                self.reservoir,
                self.stop_event,
                self.err_queue,
            ),
            daemon=True,
        )
        self.manager_proc.start()

    def __iter__(self) -> collections.abc.Iterable[ExampleBatch]:
        """Yields batches."""
        self._start_manager()
        n, b = 0, 0

        try:
            while n < self.n_samples:
                need = min(self.cfg.batch_size, self.n_samples - n)
                if not self.err_queue.empty():
                    who, tb = self.err_q.get_nowait()
                    raise RuntimeError(f"{who} crashed:\n{tb}")

                try:
                    act, meta = self.reservoir.get(
                        need, timeout=self.cfg.batch_timeout_s
                    )
                    n += need
                    b += 1
                    yield self.ExampleBatch(act=act, **meta)
                    continue
                except TimeoutError:
                    self.logger.info(
                        "Did not get a batch from %d worker threads in %.1fs seconds.",
                        self.cfg.n_threads,
                        self.cfg.batch_timeout_s,
                    )

                # If we don't continue, then we should check on the manager process.
                if not self.manager_proc.is_alive():
                    raise RuntimeError(
                        f"Manager process died unexpectedly after {b}/{len(self)} batches."
                    )

        finally:
            self.shutdown()

    def shutdown(self):
        if self.stop_event and not self.stop_event.is_set():
            self.stop_event.set()

        if self.manager_proc and self.manager_proc.is_alive():
            self.manager_proc.join(timeout=5.0)
            if self.manager_proc.is_alive():
                self.logger.warning(
                    "Manager process did not shut down cleanly, killing."
                )
                self.manager_proc.kill()

        if self.reservoir:
            self.reservoir.close()

        self.manager_proc = None
        self.reservoir = None
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
        return math.ceil(self.n_samples / self.cfg.batch_size)
