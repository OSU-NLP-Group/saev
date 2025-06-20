# src/saev/data/iterable.py
import collections.abc
import dataclasses
import logging
import os
import queue
import threading
import time
import typing
from multiprocessing.synchronize import Event

import beartype
import numpy as np
import torch
import torch.multiprocessing as mp
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from saev import helpers

from . import utils, writers


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    shard_root: str = os.path.join(".", "shards")
    """Directory with .bin shards and a metadata.json file."""
    patches: typing.Literal["cls", "image", "all"] = "image"
    """Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches."""
    layer: int | typing.Literal["all"] = -2
    """Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer."""
    clamp: float = 1e5
    """Maximum value for activations; activations will be clamped to within [-clamp, clamp]`."""
    batch_size: int = 1024 * 16
    """Batch size."""
    drop_last: bool = False
    """Whether to drop the last batch if it's smaller than the others."""
    n_threads: int = 4
    """Number of dataloading threads."""
    seed: int = 17
    """Random seed."""
    buffer_size: int = 64
    """Number of batches to queue in the shared-memory ring buffer. Higher values add latency but improve resilience to brief stalls."""


@jaxtyped(typechecker=beartype.beartype)
def _io_worker(
    worker_id: int,
    cfg: Config,
    metadata: writers.Metadata,
    work_queue: queue.Queue[int | None],
    resevoir: utils.ResevoirBuffer,
    stop_event: threading.Event,
):
    """
    Pulls work items from the queue, loads data, and pushes it to the ready queue.
    Work item is a tuple: (shard_idx, list_of_global_indices).

    See https://github.com/beartype/beartype/issues/397 for an explanation of why we use multiprocessing.queues.Queue for the type hint.
    """
    logger = logging.getLogger(f"iterable.worker{worker_id}")

    layer_i = metadata.layers.index(cfg.layer)

    logger.info(f"I/O worker {worker_id} started.")
    # Calculate shard layout constants

    # Pre-conditions
    assert cfg.patches == "image"
    assert isinstance(cfg.layer, int)

    while not stop_event.is_set():
        try:
            shard_i = work_queue.get(timeout=0.1)
            if shard_i is None:  # Poison pill
                break

            acts_fpath = os.path.join(cfg.shard_root, f"acts{shard_i:06}.bin")
            mmap = np.memmap(
                acts_fpath, mode="r", dtype=np.float32, shape=metadata.shard_shape
            )

            for start, end in helpers.batched_idx(metadata.n_imgs_per_shard, 64):
                for p in range(metadata.n_patches_per_img):
                    patch_i = p + int(metadata.cls_token)
                    acts = torch.from_numpy(mmap[start:end, layer_i, patch_i])
                    metas = [{"image_i": i, "patch_i": p} for i in range(start, end)]
                    resevoir.put(acts, metas)
        except queue.Empty:
            # Wait 0.1 seconds for new data.
            time.sleep(0.1)
            continue
        except Exception:
            logger.exception(f"Error in worker {worker_id}")
            break
    logger.info(f"I/O worker {worker_id} finished.")


@beartype.beartype
def _manager_main(
    cfg: Config,
    metadata: writers.Metadata,
    resevoir: utils.ResevoirBuffer,
    stop_event: Event,
):
    """
    The main function for the data loader manager process.
    """

    logger = logging.getLogger("iterable.manager")

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
            thread = threading.Thread(
                target=_io_worker,
                args=(i, cfg, metadata, work_queue, resevoir, thread_stop_event),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        logger.info("Launched %d I/O threads.", cfg.n_threads)

        # 4. WAIT
        while any(t.is_alive() for t in threads):
            time.sleep(1.0)

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
    class ExampleBatch(typing.TypedDict):
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
        self.resevoir = None
        self.stop_event = None
        self._total_examples = self._calculate_len()

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
        self.resevoir = utils.ResevoirBuffer(
            self.cfg.buffer_size * self.cfg.batch_size,
            (self.metadata.d_vit,),
            dtype=torch.float32,
            collate_fn=torch.utils.data.default_collate,
        )
        self.stop_event = self.ctx.Event()

        self.manager_proc = self.ctx.Process(
            target=_manager_main,
            args=(self.cfg, self.metadata, self.resevoir, self.stop_event),
            daemon=True,
        )
        self.manager_proc.start()

    def __iter__(self) -> collections.abc.Iterable[ExampleBatch]:
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
                # TODO: add a timeout (60s); if nothing happens, continue so that we check manager_proc.is_alive() again.
                act, metas = self.resevoir.get(self.cfg.batch_size)
                yield self.ExampleBatch(act=act, **metas)
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

        if self.resevoir:
            self.resevoir.close()

        self.manager_proc = None
        self.resevoir = None
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
        return (self._total_examples + self.cfg.batch_size - 1) // self.cfg.batch_size
