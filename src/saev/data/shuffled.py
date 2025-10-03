# src/saev/data/shuffled.py
# TODO: read https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
import collections.abc
import dataclasses
import logging
import math
import os
import pathlib
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

from . import buffers, shards


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for loading shuffled activation data from disk.

    Attributes:
        shards: Directory with .bin shards and a metadata.json file.
        patches: Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'image' indicates it will return image patches. 'all' returns all patches.
    """

    shards: pathlib.Path = pathlib.Path("$SAEV_SCRATCH/saev/shards/abcdefg")
    patches: typing.Literal["cls", "image", "all"] = "image"
    layer: int | typing.Literal["all"] = -2
    """Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer."""
    batch_size: int = 1024 * 16
    """Batch size."""
    drop_last: bool = False
    """Whether to drop the last batch if it's smaller than the others."""
    scale_norm: bool = False
    """Whether to scale norms to sqrt(D)."""
    # Patch filtering
    ignore_labels: list[int] = dataclasses.field(default_factory=list)
    """If provided, exclude patches with these label values. None means no filtering. Common use: ignore_labels=[0] to exclude background."""
    # Performance
    n_threads: int = 4
    """Number of dataloading threads."""
    buffer_size: int = 64
    """Number of batches to queue in the shared-memory ring buffer. Higher values add latency but improve resilience to brief stalls."""
    batch_timeout_s: float = 30.0
    """How long to wait for at least one batch."""
    # Diagnostics
    seed: int = 17
    """Random seed."""
    debug: bool = False
    """Whether the dataloader process should log debug messages."""
    log_every_s: float = 30.0
    """How frequently the dataloader process should log (debug) performance messages."""


@beartype.beartype
class ExampleOutOfBoundsError(Exception):
    def __init__(self, metadata: shards.Metadata, i: int):
        self.metadata = metadata
        self.i = i

    @property
    def message(self) -> str:
        return f"Metadata says there are {self.metadata.n_ex} examples, but we found example {self.i}."


@jaxtyped(typechecker=beartype.beartype)
def _io_worker(
    worker_id: int,
    cfg: Config,
    metadata: shards.Metadata,
    work_queue: queue.Queue[int | None],
    reservoir: buffers.ReservoirBuffer,
    stop_event: threading.Event,
    err_queue: Queue[tuple[str, str]],
    labels_mmap: np.memmap | None = None,
):
    """
    Pulls work items from the queue, loads data, and pushes it to the ready queue.
    Work item is a tuple: (shard_idx, list_of_global_indices).

    See https://github.com/beartype/beartype/issues/397 for an explanation of why we use multiprocessing.queues.Queue for the type hint.
    """
    logger = logging.getLogger(f"shuffled.worker{worker_id}")
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, force=True)
    logger = logging.getLogger("shuffled.manager")
    logger.info(
        "I/O worker %s started (debug=%s, logging=%s).",
        worker_id,
        cfg.debug,
        logging.getLevelName(logger.getEffectiveLevel()),
    )

    layer_i = metadata.layers.index(cfg.layer)
    shard_info = shards.ShardInfo.load(cfg.shards)

    # Pre-conditions
    assert cfg.patches == "image"
    assert isinstance(cfg.layer, int)

    # If we need to filter by labels, ensure we have the labels
    if cfg.ignore_labels and labels_mmap is None:
        raise ValueError("ignore_labels specified but no labels.bin found")

    bytes_sent = 0
    n_reads = 0
    t_last_report = time.time()

    chunk_size = min(1024, math.ceil(cfg.batch_size * cfg.buffer_size / cfg.n_threads))

    while not stop_event.is_set():
        try:
            shard_i = work_queue.get(timeout=0.1)
            if shard_i is None:  # Poison pill
                logger.debug("Got 'None' from work_queue; exiting.")
                break
            t1 = time.perf_counter()

            fname = f"acts{shard_i:06}.bin"
            logger.info("Opening %s.", fname)

            ex_i_offset = shard_i * metadata.ex_per_shard

            acts_fpath = os.path.join(cfg.shards, fname)
            mmap = np.memmap(
                acts_fpath, mode="r", dtype=np.float32, shape=metadata.shard_shape
            )
            t2 = time.perf_counter()

            # Only iterate over the actual number of examples in this shard
            for start, end in helpers.batched_idx(shard_info[shard_i].n_ex, chunk_size):
                for p in range(metadata.patches_per_ex):
                    patch_i = p + int(metadata.cls_token)

                    # If filtering by labels, check which samples to keep
                    if cfg.ignore_labels:
                        # Get the labels for this batch of examples and patch
                        ex_indices = np.arange(ex_i_offset + start, ex_i_offset + end)
                        patch_labels = labels_mmap[ex_indices, p]

                        # Find which samples to keep (NOT in ignore list)
                        mask = ~np.isin(patch_labels, cfg.ignore_labels)
                        valid_indices = np.where(mask)[0]

                        # Skip this batch if no samples match
                        if len(valid_indices) == 0:
                            continue

                        # Only load the matching activations
                        t0 = time.perf_counter()
                        acts = torch.from_numpy(
                            mmap[start + valid_indices, layer_i, patch_i]
                        )
                        t1 = time.perf_counter()

                        # Create metadata for valid samples only
                        meta = torch.full((len(valid_indices), 2), p, dtype=torch.int32)
                        meta[:, 0] = (
                            ex_i_offset + start + torch.from_numpy(valid_indices)
                        )
                    else:
                        # No filtering, load all
                        t0 = time.perf_counter()
                        acts = torch.from_numpy(mmap[start:end, layer_i, patch_i])
                        t1 = time.perf_counter()

                        meta = torch.full((end - start, 2), p, dtype=torch.int32)
                        meta[:, 0] = ex_i_offset + torch.arange(start, end)

                    last_ex_i = meta[:, 0].max().item()
                    if last_ex_i >= metadata.n_ex:
                        err = ExampleOutOfBoundsError(metadata, last_ex_i)
                        logger.warning(err.message)
                        raise err

                    fill_before = reservoir.fill()
                    reservoir.put(acts, meta)
                    t2 = time.perf_counter()
                    fill_after = reservoir.fill()

                    n_reads += 1
                    bytes_sent += (
                        acts.numel() * acts.element_size()
                        + meta.numel() * meta.element_size()
                    )

                    now = time.time()
                    if now - t_last_report >= cfg.log_every_s:
                        logger.debug(
                            "shard=%s mb_sent=%.1f read_ms=%.2f put_ms=%.2f fill-before=%.3f fill-after=%.3f",
                            shard_i,
                            bytes_sent / 1e6,
                            (t1 - t0) * 1e3,
                            (t2 - t1) * 1e3,
                            fill_before,
                            fill_after,
                        )
                        t_last_report = now
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
    metadata: shards.Metadata,
    reservoir: buffers.ReservoirBuffer,
    stop_event: Event,
    err_queue: Queue[tuple[str, str]],
    labels_mmap: np.memmap | None = None,
):
    """
    The main function for the data loader manager process.
    """
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, force=True)
    logger = logging.getLogger("shuffled.manager")
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

    # 1. GLOBAL SHUFFLE
    logger.info("Shuffling shards.")
    rng = np.random.default_rng(cfg.seed)
    work_items = rng.permutation(metadata.n_shards)
    logger.info("First 10 shards: %s", work_items[:10])

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
                labels_mmap,
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

        act: Float[Tensor, "batch d_model"]
        ex_i: Int[Tensor, " batch"]
        patch_i: Int[Tensor, " batch"]

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.manager_proc = None
        self.reservoir = None
        self.stop_event = None

        self.logger = logging.getLogger("shuffled.DataLoader")
        self.ctx = mp.get_context()

        if not os.path.isdir(self.cfg.shards):
            raise RuntimeError(f"Activations are not saved at '{self.cfg.shards}'.")

        if self.cfg.scale_norm:
            raise NotImplementedError("scale_norm not implemented.")

        self.metadata = shards.Metadata.load(self.cfg.shards)

        # Validate shard files exist
        shard_info = shards.ShardInfo.load(self.cfg.shards)
        for shard in shard_info:
            shard_path = os.path.join(self.cfg.shards, shard.name)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Shard file not found: {shard_path}")

        self._n_samples = self._calculate_n_samples()

        # Check if labels.bin exists for filtering
        self.labels_mmap = None
        if self.cfg.ignore_labels:
            labels_path = os.path.join(self.cfg.shards, "labels.bin")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(
                    f"ignore_labels filtering requested but labels.bin not found at {labels_path}"
                )
            # We'll create the memmap when starting the manager process

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

    @property
    def manager_pid(self) -> int:
        if not self.manager_proc or not self.manager_proc.is_alive():
            return -1

        return self.manager_proc.pid

    def _start_manager(self):
        if self.manager_proc and self.manager_proc.is_alive():
            return

        self.logger.info("Starting manager process.")

        # Create the shared-memory buffers
        self.reservoir = buffers.ReservoirBuffer(
            self.cfg.buffer_size * self.cfg.batch_size,
            (self.metadata.d_model,),
            dtype=torch.float32,
            meta_shape=(2,),
            meta_dtype=torch.int32,
            seed=self.cfg.seed,
            collate_fn=torch.utils.data.default_collate,
        )
        self.stop_event = self.ctx.Event()
        self.err_queue = self.ctx.Queue(maxsize=self.cfg.n_threads + 1)

        # Create labels memmap if needed
        labels_mmap = None
        if self.cfg.ignore_labels:
            labels_path = os.path.join(self.cfg.shards, "labels.bin")
            labels_mmap = np.memmap(
                labels_path,
                mode="r",
                dtype=np.uint8,
                shape=(self.metadata.n_ex, self.metadata.patch_per_ex),
            )

        self.manager_proc = self.ctx.Process(
            target=_manager_main,
            args=(
                self.cfg,
                self.metadata,
                self.reservoir,
                self.stop_event,
                self.err_queue,
                labels_mmap,
            ),
            daemon=True,
        )
        self.manager_proc.start()

    def __iter__(self) -> collections.abc.Iterator[ExampleBatch]:
        """Yields batches."""
        self._start_manager()
        n, b = 0, 0

        try:
            while n < self.n_samples:
                need = min(self.cfg.batch_size, self.n_samples - n)
                if not self.err_queue.empty():
                    who, tb = self.err_queue.get_nowait()
                    raise RuntimeError(f"{who} crashed:\n{tb}")

                try:
                    act, meta = self.reservoir.get(
                        need, timeout=self.cfg.batch_timeout_s
                    )
                    n += need
                    b += 1
                    ex_i, patch_i = meta.T
                    yield self.ExampleBatch(act=act, ex_i=ex_i, patch_i=patch_i)
                    continue
                except TimeoutError:
                    if self.cfg.ignore_labels:
                        self.logger.info(
                            "Did not get a batch from %d worker threads in %.1fs seconds. This can happen when filtering out many labels.",
                            self.cfg.n_threads,
                            self.cfg.batch_timeout_s,
                        )
                    else:
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

        if hasattr(self, "reservoir") and self.reservoir:
            self.reservoir.close()

        self.manager_proc = None
        self.reservoir = None
        self.stop_event = None

    def __del__(self):
        self.shutdown()

    def _calculate_n_samples(self) -> int:
        """Helper to calculate total number of examples based on config.

        When ignore_labels is specified, this counts the actual number of patches
        that remain after filtering out the ignored labels.
        """
        # First calculate the maximum possible samples
        max_samples = 0
        match (self.cfg.patches, self.cfg.layer):
            case ("cls", "all"):
                max_samples = self.metadata.n_ex * len(self.metadata.layers)
            case ("cls", int()):
                max_samples = self.metadata.n_ex
            case ("image", int()):
                max_samples = self.metadata.n_ex * self.metadata.patches_per_ex
            case ("image", "all"):
                max_samples = (
                    self.metadata.n_ex
                    * len(self.metadata.layers)
                    * self.metadata.patches_per_ex
                )
            case _:
                typing.assert_never((self.cfg.patches, self.cfg.layer))

        # If no filtering, return max samples
        if not self.cfg.ignore_labels:
            return max_samples

        # For patch filtering, count actual remaining patches
        # Note: This only works for "image" patches with fixed layer
        if self.cfg.patches != "image" or not isinstance(self.cfg.layer, int):
            raise NotImplementedError(
                "Patch label filtering only supports 'image' patches with fixed layer"
            )

        # Load labels and count remaining patches
        labels_path = os.path.join(self.cfg.shards, "labels.bin")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.bin not found at {labels_path}")

        # Memory-map the labels file
        labels = np.memmap(
            labels_path,
            mode="r",
            dtype=np.uint8,
            shape=(self.metadata.n_ex, self.metadata.n_patches_per_ex),
        )

        # Count patches that are NOT in the ignore list
        mask = ~np.isin(labels, self.cfg.ignore_labels)
        n_remaining = int(np.sum(mask))

        # Clean up the memmap
        del labels

        return n_remaining

    def __len__(self) -> int:
        """Returns the number of batches in an epoch."""
        return math.ceil(self.n_samples / self.cfg.batch_size)
