[media pointer="file-service://file-KYGX9W6Ub8pHytZbjZLqeY"]
Here's a design document for shuffled dataloaders with ViT activations:

# Performance

SAEs are mostly disk-bound.
Gemma Scope (Google SAE paper) aimed for 1 GB/s to keep their GPUS brrr'ing.
This is pretty hard even with sequential reads, much less random access.

I run all my experiments on [OSC](https://www.osc.edu/) and their scratch filesystem `/fs/scratch` has sequential read speeds of around 800 MB/s and random access speeds around 22 MB/s.

I got these numbers with:

```sh
fio --name=net --filename=/fs/scratch/PAS2136/samuelstevens/cache/saev/366017a10220b85014ae0a594276b25f6ea3d756b74d1d3218da1e34ffcf32e9/acts000000.bin --rw=read --bs=1M --direct=1 --iodepth=16 --runtime=30 --time_based
```

and

```sh
fio --name=net --filename=/fs/scratch/PAS2136/samuelstevens/cache/saev/366017a10220b85014ae0a594276b25f6ea3d756b74d1d3218da1e34ffcf32e9/acts000000.bin --rw=randread --bs=4K --direct=1 --iodepth=16 --runtime=30 --time_based
```

These two commands reported, respectively:

```
READ: bw=796MiB/s (835MB/s), 796MiB/s-796MiB/s (835MB/s-835MB/s), io=23.3GiB (25.0GB), run=30001-30001msec
```

and

```
READ: bw=22.9MiB/s (24.0MB/s), 22.9MiB/s-22.9MiB/s (24.0MB/s-24.0MB/s), io=687MiB (721MB), run=30001-30001msec
```

My naive pytorch-style dataset that uses multiple processes to feed a dataloader did purely random reads and was too slow.
It reports around 500 examples/s:

![Performance plot showing that naive random access dataloading maxes out around 500 examples/s.](assets/benchmarking/ee86c12134a89ea819b129bcce0d1abbda5143c4/plot.png)

I've implemented a dataloader that tries to do sequential reads rather than random reads in `saev/data/iterable.py`.
It's much faster (around 4.5K examples/s) on OSC.

![Performance plot showing that my first attempt at a sequential dataloader maxes out around 4500 examples/s.](assets/benchmarking/4e9b2faf065ffb21e635633a2ee485bd699b0941/plot.png)

I know that it should be even faster; the dataset of 128M examples is 2.9TB, my sequential disk read speed is 800 MB/s, so it should take ~1 hr.
For 128M examples at 4.5K examples/s, it should take 7.9 hours.
You can see this on a [wandb run here](https://wandb.ai/samuelstevens/saev/runs/okm4fv8j?nw=nwusersamuelstevens&panelDisplayName=Disk+Utilization+%28%25%29&panelSectionName=System) which reports 14.6% disk utilization.
Certainly that can be higher.

> *Not sure if this is the correct way to think about it, but: 100 / 14.6 = 6.8, close to 7.9 hours.*

## Ordered Dataloader Design

The `saev/data/ordered.py` module implements a high-throughput ordered dataloader that guarantees sequential data delivery.
This is useful for iterating through all patches in an image at once.

### Key Design Decisions

1. Single-threaded I/O in Manager Process
   
   Initially, the dataloader used multiple worker threads for parallel I/O, similar to PyTorch's DataLoader. However, this created a fundamental ordering problem: when multiple workers read batches in parallel, they complete at different times and deliver batches out of order.
   
   We switched to single-threaded I/O because:
   - Sequential reads from memory-mapped files are already highly optimized by the OS
   - The OS page cache provides excellent performance for sequential access patterns
   - Eliminating multi-threading removes all batch reordering complexity
   - The simpler design is more maintainable and debuggable

2. Process Separation with Ring Buffer
   
   The dataloader still uses a separate manager process connected via a multiprocessing Queue (acting as a ring buffer). This provides:
   - Overlap between I/O and computation
   - Configurable read-ahead via `buffer_size` parameter
   - Natural backpressure when computation is slower than I/O
   - Process isolation for better resource management

3. Shard-Aware Sequential Reading
   
   The dataloader correctly handles the actual distribution of data across shards by:
   - Reading `shards.json` to get the exact number of images per shard
   - Maintaining cumulative offsets for efficient index-to-shard mapping
   - Handling batches that span multiple shards without gaps or duplicates

### Performance Considerations

- Memory-mapped files: Using `np.memmap` allows efficient access to large files without loading them entirely into memory
- Sequential access pattern: The dataloader reads data in the exact order it's stored on disk, maximizing OS cache effectiveness
- Minimal data copying: Activations are copied only once from the memory-mapped file to PyTorch tensors
- Read-ahead buffering: The configurable buffer size allows tuning the trade-off between memory usage and I/O overlap

### Trade-offs

The single-threaded design trades potential parallel I/O throughput for:
- Guaranteed ordering
- Simplicity and maintainability  
- Elimination of synchronization overhead
- Predictable performance characteristics

In practice, the sequential read performance is sufficient for most use cases, especially when the computation (e.g., SAE forward pass) is the bottleneck rather than I/O.

---

And the implementation:

```
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
```

Here's my benchmarking script:

```
# tests/benchmark.py
"""
Slurm benchmark harness for saev data loaders.

Usage
-----
uv run python bench/run_bench.py --shard-path /fs/scratch/.../cache/saev/<hash>
"""

import dataclasses
import json
import logging
import pathlib
import subprocess
import sys
import time
import typing

import beartype
import submitit
import tyro

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Result:
    kind: typing.Literal["indexed", "ordered", "shuffled"]
    n_workers: int
    batch_size: int
    batches_per_s: float
    peak_rss_mb: float

    @property
    def patches_per_s(self) -> float:
        return self.batches_per_s * self.batch_size


@beartype.beartype
def get_git_commit() -> str:
    """Return current commit hash; abort if working tree is dirty."""
    dirty = subprocess.run(["git", "diff", "--quiet"]).returncode != 0
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode != 0
    if dirty or staged:
        sys.exit("Working tree dirty. Commit or stash changes first.")

    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def infinite(dataloader):
    """Creates an infinite iterator from a dataloader by creating a new iterator each time the previous one is exhausted.

    Args:
        dataloader: A PyTorch dataloader or similar iterable

    Yields:
        Batches from the dataloader, indefinitely
    """
    while True:
        # Create a fresh iterator from the dataloader
        it = iter(dataloader)
        for batch in it:
            yield batch


@beartype.beartype
def benchmark_fn(
    kind: typing.Literal["indexed", "shuffled", "ordered"],
    *,
    shard_root: str,
    layer: int,
    batch_size: int,
    n_workers: int,
    warmup_min: int,
    run_min: int,
) -> Result:
    import psutil
    import torch

    import saev.data.indexed
    import saev.data.ordered
    import saev.data.shuffled

    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("benchmark_fn")

    if kind == "indexed":
        cfg = saev.data.indexed.Config(
            shard_root=shard_root,
            patches="image",
            layer=layer,
        )
        ds = saev.data.indexed.Dataset(cfg)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=True,
            persistent_workers=True,
        )
    elif kind == "ordered":
        cfg = saev.data.ordered.Config(
            shard_root=shard_root,
            patches="image",
            layer=layer,
            batch_size=batch_size,
            buffer_size=128,
        )
        dl = saev.data.ordered.DataLoader(cfg)
    elif kind == "shuffled":
        cfg = saev.data.shuffled.Config(
            shard_root=shard_root,
            patches="image",
            layer=layer,
            batch_size=batch_size,
            n_threads=n_workers,
            buffer_size=128,
            seed=0,
        )
        dl = saev.data.shuffled.DataLoader(cfg)
    else:
        raise ValueError(kind)

    logger.info("kind: %s, cfg: %s", kind, cfg)

    # warm-up
    logger.info("Warming up for %d minutes.", warmup_min)
    start = time.perf_counter()
    tgt_end = start + warmup_min * 60
    it = infinite(dl)
    n_warmup = 0
    while time.perf_counter() < tgt_end:
        next(it)
        n_warmup += 1
        logger.info("warmup batches: %d", n_warmup)

    true_end = time.perf_counter()
    warmup_duration_m = (true_end - start) / 60
    logger.info("Warmed up (took %.1f minutes).", warmup_duration_m)

    # measured run
    n_batches = 0
    rss_max = 0
    proc = psutil.Process()
    start = time.perf_counter()
    tgt_end = start + run_min * 60
    while time.perf_counter() < tgt_end:
        next(it)
        n_batches += 1

        rss = proc.memory_info().rss + sum(
            c.memory_info().rss for c in proc.children(recursive=True)
        )
        rss_max = max(rss_max, rss)

        logger.info("batches: %d, peak GB: %.1f", n_batches, rss_max / 1e9)

    true_end = time.perf_counter()

    bps = n_batches / (true_end - start)
    return Result(
        kind=kind,
        n_workers=n_workers,
        batch_size=batch_size,
        batches_per_s=bps,
        peak_rss_mb=rss_max / 1e6,
    )


@beartype.beartype
def benchmark(
    shards: str,
    layer: int,
    run_min: int = 10,
    warmup_min: int = 5,
    margin_min: int = 45,
    out: pathlib.Path = pathlib.Path("logs", "benchmarking"),
    slurm_partition: str = "gpu",
    slurm_acct: str = "",
    n_iter: int = 8,
):
    import saev.data.writers
    import saev.helpers

    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("benchmark")

    commit = get_git_commit()

    out = out / commit
    out.mkdir(parents=True, exist_ok=True)
    ex = submitit.SlurmExecutor(str(out))
    ex.update_parameters(
        partition=slurm_partition,
        account=slurm_acct,
        gpus_per_node=0,
        ntasks_per_node=1,
        cpus_per_task=16,
        time=run_min + warmup_min + margin_min,
        stderr_to_stdout=True,
    )
    jobs = []
    with ex.batch():
        for kind in ["indexed", "ordered", "shuffled"]:
            for n_workers in [2, 4, 8, 16, 32]:
                for batch_size in [2, 4, 8, 16]:
                    for _ in range(n_iter):
                        job = ex.submit(
                            benchmark_fn,
                            kind,
                            shard_root=shards,
                            layer=layer,
                            n_workers=n_workers,
                            batch_size=batch_size * 1024,
                            warmup_min=warmup_min,
                            run_min=run_min,
                        )
                        jobs.append(job)

    logger.info("Submitted %d jobs.", len(jobs))
    results = []
    for j, job in enumerate(saev.helpers.progress(jobs)):
        try:
            results.append(job.result())
        except submitit.core.utils.UncompletedJobError:
            logger.warning("Job %d did not finish.", j)

    meta = saev.data.writers.Metadata.load(shards)
    payload = dict(
        meta=dataclasses.asdict(meta), results=[dataclasses.asdict(r) for r in results]
    )
    with open(out / "results.json", "w") as f:
        json.dump(payload, f, indent=4)


@beartype.beartype
def plot(results: pathlib.Path):
    import altair as alt
    import polars as pl

    alt.renderers.enable("png")

    df = (
        pl.read_json(results)
        .select("results")
        .explode("results")
        .unnest("results")
        .with_columns(patches_per_s=pl.col("batch_size") * pl.col("batches_per_s"))
    )

    title = str(results.parent)

    band = (
        alt.Chart(df)
        .mark_errorband(extent="stdev")  # mean +/- 1 stddev
        .encode(
            alt.X("n_workers", type="quantitative"),
            alt.Y("patches_per_s", type="quantitative"),
            alt.Color("batch_size", type="nominal"),
            alt.Shape("kind", type="nominal"),
            alt.Detail("kind", type="nominal"),
        )
    )

    line = (
        alt.Chart(df, title=title)
        .mark_line(point=True)
        .encode(
            alt.X("n_workers", type="quantitative"),
            alt.Y("patches_per_s", aggregate="mean", type="quantitative"),
            alt.Color("batch_size", type="nominal"),
            alt.Shape("kind", type="nominal"),
            alt.Detail("kind", type="nominal"),
        )
    )

    fpath = results.parent / "plot.png"

    (band + line).save(fpath, ppi=300)

    print(f"Saved to {fpath}", file=sys.stderr)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"benchmark": benchmark, "plot": plot})
```

And here are results (both as an image and as json):

```json
{
    "meta": {
        "vit_family": "siglip",
        "vit_ckpt": "hf-hub:timm/ViT-L-16-SigLIP2-256",
        "layers": [
            13,
            15,
            17,
            19,
            21,
            23
        ],
        "n_patches_per_img": 256,
        "cls_token": false,
        "d_vit": 1024,
        "n_imgs": 500000,
        "max_patches_per_shard": 500000,
        "data": {
            "__cls__": "ImageFolder",
            "root": "/fs/ess/PAS2136/foundation_model/inat21/raw/train_mini/"
        },
        "dtype": "float32",
        "protocol": "1.0.0"
    },
    "results": [
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.0953811802910761,
            "peak_rss_mb": 6327.717888
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.09544645007871033,
            "peak_rss_mb": 6322.270208
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.534078291985787,
            "peak_rss_mb": 861.667328
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.10480753909228946,
            "peak_rss_mb": 6319.8208
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.09552472183662788,
            "peak_rss_mb": 6330.466304
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.5338894692128782,
            "peak_rss_mb": 855.015424
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.53390034018454,
            "peak_rss_mb": 851.800064
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.7562939351060722,
            "peak_rss_mb": 900.431872
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.5338873973809095,
            "peak_rss_mb": 859.623424
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.7798860899434091,
            "peak_rss_mb": 860.8768
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.09497133108558298,
            "peak_rss_mb": 6326.628352
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.348386527868692,
            "peak_rss_mb": 983.216128
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.34837446244494535,
            "peak_rss_mb": 987.168768
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.7562953853574729,
            "peak_rss_mb": 885.73952
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.7562940960115203,
            "peak_rss_mb": 894.656512
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.7562961616435689,
            "peak_rss_mb": 885.448704
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.7795323333584896,
            "peak_rss_mb": 876.191744
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.7562960219441899,
            "peak_rss_mb": 877.338624
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.7795306651797387,
            "peak_rss_mb": 862.994432
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.7795331045481233,
            "peak_rss_mb": 851.136512
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.7795267691637414,
            "peak_rss_mb": 846.938112
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.7795285716028214,
            "peak_rss_mb": 862.683136
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.7795297983142306,
            "peak_rss_mb": 865.56672
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.3899684835149664,
            "peak_rss_mb": 890.105856
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.7795306524478142,
            "peak_rss_mb": 858.882048
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.3899681707253317,
            "peak_rss_mb": 895.049728
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.38996897248745127,
            "peak_rss_mb": 895.213568
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.38996752971220466,
            "peak_rss_mb": 892.145664
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.7783581777643751,
            "peak_rss_mb": 844.30848
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.3899697105402248,
            "peak_rss_mb": 881.33632
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.3899704869819891,
            "peak_rss_mb": 891.957248
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.7783618802211403,
            "peak_rss_mb": 863.379456
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.7783595238334137,
            "peak_rss_mb": 859.574272
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.19080381767088672,
            "peak_rss_mb": 6993.317888
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.7783599790761657,
            "peak_rss_mb": 853.6064
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.3931354143430613,
            "peak_rss_mb": 891.908096
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.7783593329322148,
            "peak_rss_mb": 853.34016
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.7783595868483324,
            "peak_rss_mb": 871.477248
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.38921317239341335,
            "peak_rss_mb": 895.16032
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.3931359062664094,
            "peak_rss_mb": 899.657728
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.39313719877645115,
            "peak_rss_mb": 894.029824
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.19654410666617167,
            "peak_rss_mb": 903.118848
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.39313715419419476,
            "peak_rss_mb": 890.458112
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.19198084924432854,
            "peak_rss_mb": 966.569984
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.19516505641657103,
            "peak_rss_mb": 908.816384
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.1965432620511531,
            "peak_rss_mb": 895.87712
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.19653908064638517,
            "peak_rss_mb": 897.011712
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.10084192626392803,
            "peak_rss_mb": 6316.70784
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.0972712147758687,
            "peak_rss_mb": 6322.180096
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.06540105848956287,
            "peak_rss_mb": 6348.10368
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.10595695071539415,
            "peak_rss_mb": 6313.795584
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.04637164106330478,
            "peak_rss_mb": 6342.193152
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.19516160103134536,
            "peak_rss_mb": 904.032256
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.19516551015585942,
            "peak_rss_mb": 898.224128
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.1951651099062157,
            "peak_rss_mb": 899.50208
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.1951634718806252,
            "peak_rss_mb": 906.420224
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.1951622329138314,
            "peak_rss_mb": 906.334208
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.09108880796262857,
            "peak_rss_mb": 848.494592
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.19516401080220516,
            "peak_rss_mb": 902.946816
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.0949050819124635,
            "peak_rss_mb": 6310.264832
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.09663530074616206,
            "peak_rss_mb": 979.206144
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.09108909390246002,
            "peak_rss_mb": 855.232512
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.0966364081569373,
            "peak_rss_mb": 972.45184
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.09663635152996004,
            "peak_rss_mb": 983.392256
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.09663503491690345,
            "peak_rss_mb": 977.395712
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.09663717513309675,
            "peak_rss_mb": 974.147584
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.06398424494520157,
            "peak_rss_mb": 6347.48928
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.09663641562087329,
            "peak_rss_mb": 981.123072
        },
        {
            "kind": "ordered",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.09663498378253377,
            "peak_rss_mb": 979.00544
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.19045661401580882,
            "peak_rss_mb": 6989.68064
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.04642115501400174,
            "peak_rss_mb": 6346.77248
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.475955758784489,
            "peak_rss_mb": 8354.6112
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.06509140719885247,
            "peak_rss_mb": 6345.207808
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.19203516125188638,
            "peak_rss_mb": 7010.97984
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.510238248788955,
            "peak_rss_mb": 8357.912576
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.09604589813716248,
            "peak_rss_mb": 7080.198144
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.5035405514406216,
            "peak_rss_mb": 8375.697408
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.24818514691965163,
            "peak_rss_mb": 7021.481984
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7461344484833534,
            "peak_rss_mb": 11082.969088
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7539621689745196,
            "peak_rss_mb": 11048.443904
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.5048554152658199,
            "peak_rss_mb": 8371.007488
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.4785683432739087,
            "peak_rss_mb": 8389.935104
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7348441229987999,
            "peak_rss_mb": 11040.940032
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 0.0934275933276077,
            "peak_rss_mb": 6322.561024
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7450419611650777,
            "peak_rss_mb": 11116.253184
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.1893429294449006,
            "peak_rss_mb": 7001.923584
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7563784281284326,
            "peak_rss_mb": 11038.4128
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.3718711137053235,
            "peak_rss_mb": 8398.135296
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.3716166620655913,
            "peak_rss_mb": 8383.053824
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7553819091913934,
            "peak_rss_mb": 11093.684224
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.06557216538841784,
            "peak_rss_mb": 6348.464128
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.3707173118378452,
            "peak_rss_mb": 8374.910976
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.24806098601644966,
            "peak_rss_mb": 6992.359424
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7548243299298864,
            "peak_rss_mb": 11087.171584
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7593449270852104,
            "peak_rss_mb": 11060.027392
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.06519665548842433,
            "peak_rss_mb": 6349.643776
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.06505321477755492,
            "peak_rss_mb": 6360.666112
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.5091964660003925,
            "peak_rss_mb": 8340.385792
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.1904758801382868,
            "peak_rss_mb": 6986.747904
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.06482265680413499,
            "peak_rss_mb": 6353.309696
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 0.7622224541582604,
            "peak_rss_mb": 16485.261312
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.09565275420687974,
            "peak_rss_mb": 7031.595008
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.09537670351994988,
            "peak_rss_mb": 7035.65824
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.09538768194411579,
            "peak_rss_mb": 7047.02464
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.09538731668002981,
            "peak_rss_mb": 7059.329024
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.23984798230897647,
            "peak_rss_mb": 8476.864512
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.2398091785009067,
            "peak_rss_mb": 8466.55488
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.2404834991496441,
            "peak_rss_mb": 8502.04672
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.09518439533023804,
            "peak_rss_mb": 7040.892928
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7594969042580083,
            "peak_rss_mb": 11123.613696
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.3723178805576083,
            "peak_rss_mb": 11314.52416
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.014221580598999847,
            "peak_rss_mb": 6488.813568
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.032018658515736376,
            "peak_rss_mb": 6395.24864
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.19030677930165996,
            "peak_rss_mb": 6989.266944
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.19167359253733646,
            "peak_rss_mb": 6986.952704
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.3704565758825186,
            "peak_rss_mb": 11277.074432
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 0.7603586791829875,
            "peak_rss_mb": 16407.605248
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 0.7596806887303806,
            "peak_rss_mb": 16467.47648
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7597067511750598,
            "peak_rss_mb": 11120.943104
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.37173139345828415,
            "peak_rss_mb": 8387.334144
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.0651686691849053,
            "peak_rss_mb": 6353.36704
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.09648621502422433,
            "peak_rss_mb": 8630.960128
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 0.19202464913994277,
            "peak_rss_mb": 6988.603392
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.19062424481559875,
            "peak_rss_mb": 8445.89056
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.09534980210578686,
            "peak_rss_mb": 7061.958656
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.19095877149679358,
            "peak_rss_mb": 8490.381312
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.03269693026216981,
            "peak_rss_mb": 6392.815616
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.09590668757890457,
            "peak_rss_mb": 8640.200704
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.028821335752421795,
            "peak_rss_mb": 6400.253952
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.09418560822009966,
            "peak_rss_mb": 8669.786112
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.23873914324506817,
            "peak_rss_mb": 8472.006656
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.09618196665746616,
            "peak_rss_mb": 7035.092992
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.19009153150486552,
            "peak_rss_mb": 8432.88576
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.03178889485196024,
            "peak_rss_mb": 6400.503808
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.15658593898959236,
            "peak_rss_mb": 11671.412736
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.19006985928020634,
            "peak_rss_mb": 8475.037696
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.032013249965625386,
            "peak_rss_mb": 6401.839104
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.03193681758697047,
            "peak_rss_mb": 6396.755968
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.18921447122005872,
            "peak_rss_mb": 8444.899328
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.2418958666023683,
            "peak_rss_mb": 8434.307072
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.09540675222762021,
            "peak_rss_mb": 8709.812224
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.09619869107756619,
            "peak_rss_mb": 7052.92288
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.02387488158266766,
            "peak_rss_mb": 6399.459328
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.09595212041377303,
            "peak_rss_mb": 8663.117824
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.09599801655693989,
            "peak_rss_mb": 8630.96832
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 0.5330757450675683,
            "peak_rss_mb": 16591.0528
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 0.5371509452477957,
            "peak_rss_mb": 16470.806528
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.048032847384393985,
            "peak_rss_mb": 7153.307648
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.03182520253944334,
            "peak_rss_mb": 6396.329984
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 0.5324272647376775,
            "peak_rss_mb": 16368.861184
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 0.5331239028576995,
            "peak_rss_mb": 16442.253312
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 0.5327069101091471,
            "peak_rss_mb": 16414.33088
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.31142436994205713,
            "peak_rss_mb": 11252.20352
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.310860372126242,
            "peak_rss_mb": 11210.838016
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.3099524419996161,
            "peak_rss_mb": 11286.003712
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.318236848873756,
            "peak_rss_mb": 11341.66016
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 0.5322688613699206,
            "peak_rss_mb": 16546.742272
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 0.5313891771205271,
            "peak_rss_mb": 16488.902656
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.09598607425880365,
            "peak_rss_mb": 8635.12576
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.3146158164534962,
            "peak_rss_mb": 11284.13184
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.04826856812394417,
            "peak_rss_mb": 7130.083328
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.023821213450787408,
            "peak_rss_mb": 6419.697664
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.309809477141962,
            "peak_rss_mb": 11328.372736
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.02386577664822001,
            "peak_rss_mb": 6395.564032
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.04832346441365929,
            "peak_rss_mb": 7146.754048
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.04829756911819874,
            "peak_rss_mb": 7181.02528
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.04792031908491128,
            "peak_rss_mb": 7149.674496
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.20964903662222892,
            "peak_rss_mb": 11668.406272
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.09585260796953703,
            "peak_rss_mb": 12442.13248
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.048293238996297026,
            "peak_rss_mb": 7146.491904
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.0444810663524527,
            "peak_rss_mb": 9009.696768
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.20891075366152054,
            "peak_rss_mb": 11703.1936
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.3162851699319989,
            "peak_rss_mb": 11209.1136
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.09605158141995976,
            "peak_rss_mb": 7063.38816
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.2089616379585731,
            "peak_rss_mb": 11678.273536
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.2102517672921467,
            "peak_rss_mb": 11681.611776
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.04710315075880686,
            "peak_rss_mb": 9008.898048
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.047972392590864936,
            "peak_rss_mb": 7156.588544
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.09690273828784528,
            "peak_rss_mb": 8685.551616
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.04800555400018991,
            "peak_rss_mb": 7139.627008
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.3172036993089612,
            "peak_rss_mb": 11357.114368
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.1565275645263255,
            "peak_rss_mb": 11628.945408
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.09587621804065195,
            "peak_rss_mb": 8639.71328
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.09575318400732588,
            "peak_rss_mb": 12428.210176
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.04789571141166288,
            "peak_rss_mb": 7176.64256
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.04810860514104229,
            "peak_rss_mb": 7166.656512
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.04699079962020681,
            "peak_rss_mb": 9058.42688
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.0955297940954115,
            "peak_rss_mb": 12427.804672
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.09654802750276938,
            "peak_rss_mb": 8719.40096
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.26651070247750486,
            "peak_rss_mb": 16826.626048
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.09515253449371938,
            "peak_rss_mb": 12390.957056
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.09557992399798002,
            "peak_rss_mb": 12352.438272
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.2663175139028396,
            "peak_rss_mb": 16703.975424
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.26508162588369255,
            "peak_rss_mb": 16844.005376
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.027963935539600686,
            "peak_rss_mb": 7346.5856
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.15991234405468657,
            "peak_rss_mb": 11619.295232
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.26562298050885713,
            "peak_rss_mb": 16721.977344
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.26528900824883966,
            "peak_rss_mb": 16867.917824
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.26435833953773114,
            "peak_rss_mb": 16924.196864
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.2648963603641453,
            "peak_rss_mb": 16814.358528
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.26450120478708505,
            "peak_rss_mb": 16808.513536
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.1643650957258768,
            "peak_rss_mb": 11581.939712
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.16000200796155797,
            "peak_rss_mb": 11584.802816
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.1615621623871749,
            "peak_rss_mb": 11632.852992
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.016071362398221944,
            "peak_rss_mb": 6472.302592
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.028006065543495853,
            "peak_rss_mb": 7313.57184
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.02555529270067886,
            "peak_rss_mb": 7374.11072
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.01594635578105761,
            "peak_rss_mb": 6472.384512
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.024910304969999818,
            "peak_rss_mb": 7333.376
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.026076741018090006,
            "peak_rss_mb": 7331.79904
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.027993410225075566,
            "peak_rss_mb": 7312.805888
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.015988100363137067,
            "peak_rss_mb": 6486.781952
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.01603886421252337,
            "peak_rss_mb": 6484.062208
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.04886489264466587,
            "peak_rss_mb": 9050.189824
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.04944316268300279,
            "peak_rss_mb": 9006.702592
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.02815373346925128,
            "peak_rss_mb": 7335.182336
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.024845341632517986,
            "peak_rss_mb": 7326.33088
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.10612699366246027,
            "peak_rss_mb": 12323.495936
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.2645307480284461,
            "peak_rss_mb": 16854.786048
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.015969210906201462,
            "peak_rss_mb": 6482.649088
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.263640297254429,
            "peak_rss_mb": 16812.376064
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.049079775385934596,
            "peak_rss_mb": 9035.28448
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.04983122157403828,
            "peak_rss_mb": 9016.29952
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.05052632727430921,
            "peak_rss_mb": 9002.921984
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.011933185829822689,
            "peak_rss_mb": 6470.41024
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.011908311394904742,
            "peak_rss_mb": 6477.43488
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 1.6218266010317042,
            "peak_rss_mb": 896.88064
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.011925072668410397,
            "peak_rss_mb": 6480.318464
        },
        {
            "kind": "indexed",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.011968410406118292,
            "peak_rss_mb": 6470.275072
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.13592705357896218,
            "peak_rss_mb": 17637.339136
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.1357690843424621,
            "peak_rss_mb": 17485.021184
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.13570919530957343,
            "peak_rss_mb": 17433.350144
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.13569051035022034,
            "peak_rss_mb": 17563.656192
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.06088683398194776,
            "peak_rss_mb": 12317.376512
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.06088672030578276,
            "peak_rss_mb": 12271.935488
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.13263008202235518,
            "peak_rss_mb": 18933.415936
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.02521011047542785,
            "peak_rss_mb": 7367.29088
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.060979392914212394,
            "peak_rss_mb": 12344.139776
        },
        {
            "kind": "indexed",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.061021085127139586,
            "peak_rss_mb": 12342.239232
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.05070300460031412,
            "peak_rss_mb": 8991.768576
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.13240979574645273,
            "peak_rss_mb": 17630.183424
        },
        {
            "kind": "indexed",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.025193074446943537,
            "peak_rss_mb": 7307.014144
        },
        {
            "kind": "indexed",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.050533757942202134,
            "peak_rss_mb": 9030.885376
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.13260693407649582,
            "peak_rss_mb": 18956.066816
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.13277786110897183,
            "peak_rss_mb": 17586.671616
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.13322525358954962,
            "peak_rss_mb": 18936.684544
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.13267082142081282,
            "peak_rss_mb": 17597.288448
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.13276435476381548,
            "peak_rss_mb": 18890.694656
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.13213828327536994,
            "peak_rss_mb": 17530.728448
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.13257715666598366,
            "peak_rss_mb": 17560.317952
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.13261539909206085,
            "peak_rss_mb": 17487.966208
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.09838265173729856,
            "peak_rss_mb": 18977.263616
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.09759871586953635,
            "peak_rss_mb": 18976.37888
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.07026983417723219,
            "peak_rss_mb": 18826.346496
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.09774634109045105,
            "peak_rss_mb": 18879.033344
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.09770757025397718,
            "peak_rss_mb": 18956.435456
        },
        {
            "kind": "indexed",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.09771329400647434,
            "peak_rss_mb": 18895.101952
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.740916620128332,
            "peak_rss_mb": 898.670592
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 1.6057302598494494,
            "peak_rss_mb": 895.279104
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 1.6338542744155737,
            "peak_rss_mb": 905.904128
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 1.5313854397360198,
            "peak_rss_mb": 899.252224
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 1.5889670852400102,
            "peak_rss_mb": 911.495168
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 2.1637572139907006,
            "peak_rss_mb": 873.930752
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.817735063252423,
            "peak_rss_mb": 903.294976
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.8157253798390938,
            "peak_rss_mb": 915.615744
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.7122880282212403,
            "peak_rss_mb": 913.53088
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 2.1639832377266526,
            "peak_rss_mb": 856.285184
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 2.1639076882832446,
            "peak_rss_mb": 864.407552
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.7254669517234931,
            "peak_rss_mb": 917.225472
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.6590882083619893,
            "peak_rss_mb": 909.807616
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.6590947094567413,
            "peak_rss_mb": 905.125888
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.725367956359926,
            "peak_rss_mb": 911.72864
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.7122879654414374,
            "peak_rss_mb": 908.9024
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 1.4818381163684968,
            "peak_rss_mb": 875.446272
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.7496734044377421,
            "peak_rss_mb": 904.68352
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.6551286976211604,
            "peak_rss_mb": 910.618624
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 2.952543784973979,
            "peak_rss_mb": 873.967616
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.7409211494604994,
            "peak_rss_mb": 904.327168
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.7409158011845448,
            "peak_rss_mb": 879.39072
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.7409140584913224,
            "peak_rss_mb": 904.527872
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7861019446061935,
            "peak_rss_mb": 887.980032
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 2.952595283517396,
            "peak_rss_mb": 870.334464
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.40422266779386173,
            "peak_rss_mb": 982.05696
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7861008628223066,
            "peak_rss_mb": 868.61824
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7860984494002871,
            "peak_rss_mb": 894.959616
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.38087212669213183,
            "peak_rss_mb": 912.498688
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7861040731929246,
            "peak_rss_mb": 867.364864
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7861080689251683,
            "peak_rss_mb": 894.128128
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7861052932539577,
            "peak_rss_mb": 859.70944
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 0.7860994227830118,
            "peak_rss_mb": 890.24512
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.3808901760633509,
            "peak_rss_mb": 914.194432
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.38087800562193613,
            "peak_rss_mb": 908.517376
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.2144886637500965,
            "peak_rss_mb": 981.655552
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.38084774136533667,
            "peak_rss_mb": 912.572416
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.8346472594122143,
            "peak_rss_mb": 908.009472
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.3821538384421622,
            "peak_rss_mb": 890.761216
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.382157433704907,
            "peak_rss_mb": 905.09312
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.38216417033740996,
            "peak_rss_mb": 894.38208
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.3821517189534014,
            "peak_rss_mb": 903.712768
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.7836515821529916,
            "peak_rss_mb": 864.19456
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.3821554299366786,
            "peak_rss_mb": 896.344064
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.3821534718665994,
            "peak_rss_mb": 902.844416
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.7836535195234829,
            "peak_rss_mb": 883.372032
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.7836495537856697,
            "peak_rss_mb": 857.665536
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.78365570128657,
            "peak_rss_mb": 862.195712
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.7836542298088605,
            "peak_rss_mb": 872.349696
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.7836500387172888,
            "peak_rss_mb": 866.2016
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.26327983596745735,
            "peak_rss_mb": 982.831104
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 0.7836517208266196,
            "peak_rss_mb": 887.967744
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.3519005623227805,
            "peak_rss_mb": 988.114944
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.26328730702410846,
            "peak_rss_mb": 996.532224
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.26328406604608384,
            "peak_rss_mb": 987.742208
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.3919303059710269,
            "peak_rss_mb": 886.841344
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.2632778738647881,
            "peak_rss_mb": 989.995008
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.3919304031516606,
            "peak_rss_mb": 893.497344
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.39193238849832457,
            "peak_rss_mb": 884.535296
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.39193157372832343,
            "peak_rss_mb": 886.337536
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.3919321271212878,
            "peak_rss_mb": 903.708672
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.39610036875821175,
            "peak_rss_mb": 895.766528
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.391934423563842,
            "peak_rss_mb": 884.256768
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.3919327704989492,
            "peak_rss_mb": 893.898752
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.42881542750590484,
            "peak_rss_mb": 895.356928
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.42882176528209526,
            "peak_rss_mb": 894.910464
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.3639633706697795,
            "peak_rss_mb": 916.168704
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.3960997333598225,
            "peak_rss_mb": 890.535936
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.36393775957389385,
            "peak_rss_mb": 909.08672
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.3639633123905859,
            "peak_rss_mb": 909.029376
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.19804346746634047,
            "peak_rss_mb": 905.658368
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.36396323498110267,
            "peak_rss_mb": 914.128896
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.36395257319223456,
            "peak_rss_mb": 915.931136
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.19804282231863524,
            "peak_rss_mb": 905.162752
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.19804266181547736,
            "peak_rss_mb": 898.92864
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.09600269596808683,
            "peak_rss_mb": 980.451328
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.1980398948172092,
            "peak_rss_mb": 902.92224
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.19804162854207089,
            "peak_rss_mb": 905.023488
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.21512764982690347,
            "peak_rss_mb": 978.706432
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.21438359109689767,
            "peak_rss_mb": 973.225984
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.19706758769450305,
            "peak_rss_mb": 906.674176
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.09600155734031789,
            "peak_rss_mb": 979.664896
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.09600199907206561,
            "peak_rss_mb": 967.553024
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.09600159344888598,
            "peak_rss_mb": 979.873792
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.09600110874418528,
            "peak_rss_mb": 979.316736
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.09600187986360438,
            "peak_rss_mb": 979.656704
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.1961626855187893,
            "peak_rss_mb": 909.520896
        },
        {
            "kind": "ordered",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.0960011811605203,
            "peak_rss_mb": 974.942208
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.09808147075451548,
            "peak_rss_mb": 986.763264
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.09808132254141662,
            "peak_rss_mb": 991.96928
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.19616153656719515,
            "peak_rss_mb": 910.061568
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.0980809329008634,
            "peak_rss_mb": 989.847552
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.09808276795745881,
            "peak_rss_mb": 986.783744
        },
        {
            "kind": "ordered",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.0980805396375596,
            "peak_rss_mb": 987.869184
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.09859331060884466,
            "peak_rss_mb": 981.491712
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.09859320297305278,
            "peak_rss_mb": 991.584256
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.09859314693328743,
            "peak_rss_mb": 983.629824
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.09859252957774428,
            "peak_rss_mb": 982.34368
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.09859278943042998,
            "peak_rss_mb": 992.079872
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.09859345036991171,
            "peak_rss_mb": 988.987392
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 2.563499474602479,
            "peak_rss_mb": 852.488192
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 2.9330787180949014,
            "peak_rss_mb": 870.027264
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 2.9329599648142923,
            "peak_rss_mb": 858.206208
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 3.1931809829138844,
            "peak_rss_mb": 854.020096
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 3.118368535020876,
            "peak_rss_mb": 877.805568
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.7827264918894434,
            "peak_rss_mb": 899.54304
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.6565638162173397,
            "peak_rss_mb": 905.494528
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.6565637680228394,
            "peak_rss_mb": 909.094912
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.752553686137746,
            "peak_rss_mb": 909.008896
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.2731774145567578,
            "peak_rss_mb": 989.724672
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 2.5635238092724464,
            "peak_rss_mb": 863.801344
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.0191020584591002,
            "peak_rss_mb": 861.601792
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 2.563516685264569,
            "peak_rss_mb": 853.282816
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.39581104318648896,
            "peak_rss_mb": 984.027136
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 1.4471905954985644,
            "peak_rss_mb": 903.561216
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 1.4921762071498763,
            "peak_rss_mb": 900.071424
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 1.4665550608849045,
            "peak_rss_mb": 903.217152
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 1.4946729318360978,
            "peak_rss_mb": 905.035776
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 1.7631818108127535,
            "peak_rss_mb": 2948.9152
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.019087403203109,
            "peak_rss_mb": 869.732352
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.0190518656226495,
            "peak_rss_mb": 871.043072
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 1.4961966817158119,
            "peak_rss_mb": 904.163328
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 1.4817120156677852,
            "peak_rss_mb": 906.694656
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.3779714319295647,
            "peak_rss_mb": 988.975104
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.5095354469519329,
            "peak_rss_mb": 904.704
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.509535033072754,
            "peak_rss_mb": 904.773632
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.5095364802016452,
            "peak_rss_mb": 880.828416
        },
        {
            "kind": "ordered",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.10325593522054137,
            "peak_rss_mb": 983.629824
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 1.7647830699586493,
            "peak_rss_mb": 2986.016768
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 1.4908593194910833,
            "peak_rss_mb": 3014.332416
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.27471790396341816,
            "peak_rss_mb": 992.948224
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 1.7710060369008405,
            "peak_rss_mb": 3010.363392
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 1.7517382017355814,
            "peak_rss_mb": 2972.7744
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 1.7761750494068713,
            "peak_rss_mb": 2937.38496
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 1.7669456297691766,
            "peak_rss_mb": 2986.143744
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 1.770252915905162,
            "peak_rss_mb": 2989.002752
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.2747224890934661,
            "peak_rss_mb": 985.35424
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.8834039270256024,
            "peak_rss_mb": 4098.793472
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.2747187661352537,
            "peak_rss_mb": 989.380608
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.27471718887641683,
            "peak_rss_mb": 989.827072
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.3931179210633277,
            "peak_rss_mb": 6347.07968
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.41412300130852286,
            "peak_rss_mb": 6303.531008
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 1.7431426165716524,
            "peak_rss_mb": 3006.369792
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.883780550861636,
            "peak_rss_mb": 4085.407744
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 2048,
            "batches_per_s": 1.7429312660168146,
            "peak_rss_mb": 3013.640192
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.883068098858446,
            "peak_rss_mb": 4105.35936
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.7885052502256804,
            "peak_rss_mb": 4040.679424
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.8839761623654371,
            "peak_rss_mb": 4078.780416
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.8860066612979295,
            "peak_rss_mb": 4084.273152
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.22261978157797538,
            "peak_rss_mb": 10576.166912
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.6503593917050157,
            "peak_rss_mb": 4597.878784
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.4519375741152623,
            "peak_rss_mb": 3724.935168
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.8818770670079749,
            "peak_rss_mb": 4071.636992
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.8944954058734824,
            "peak_rss_mb": 4092.321792
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.8876223075341163,
            "peak_rss_mb": 4097.204224
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.7265883000539762,
            "peak_rss_mb": 4078.108672
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.2022150543952288,
            "peak_rss_mb": 10582.347776
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.21940506753365835,
            "peak_rss_mb": 10593.665024
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.44093793766589223,
            "peak_rss_mb": 6223.568896
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 4096,
            "batches_per_s": 0.8850099911091386,
            "peak_rss_mb": 4074.840064
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.6528776164003844,
            "peak_rss_mb": 4277.37088
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.2194197695711285,
            "peak_rss_mb": 10599.002112
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.440485649271331,
            "peak_rss_mb": 6232.825856
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.4404814084249121,
            "peak_rss_mb": 6217.170944
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.4404718332241796,
            "peak_rss_mb": 6237.974528
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.3680425345857779,
            "peak_rss_mb": 6235.234304
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.43904910342635356,
            "peak_rss_mb": 6202.970112
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.4408048971556835,
            "peak_rss_mb": 6229.151744
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.22405112258648938,
            "peak_rss_mb": 10602.852352
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.19150197545056818,
            "peak_rss_mb": 989.540352
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.21916766234993923,
            "peak_rss_mb": 10590.552064
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 8192,
            "batches_per_s": 0.442792072886185,
            "peak_rss_mb": 6236.909568
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.21460492798825875,
            "peak_rss_mb": 10548.129792
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.22323996729123322,
            "peak_rss_mb": 10589.118464
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.19856501094674253,
            "peak_rss_mb": 895.42656
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.19149993903885798,
            "peak_rss_mb": 988.798976
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.7284696703098754,
            "peak_rss_mb": 6712.000512
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.19856527729217072,
            "peak_rss_mb": 916.512768
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.19856550147658705,
            "peak_rss_mb": 905.715712
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.19149777428318404,
            "peak_rss_mb": 988.205056
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.198566593573328,
            "peak_rss_mb": 909.094912
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.198563966864255,
            "peak_rss_mb": 911.065088
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.19856303058138394,
            "peak_rss_mb": 905.588736
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.19856600881961273,
            "peak_rss_mb": 905.998336
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.19149738712332223,
            "peak_rss_mb": 986.820608
        },
        {
            "kind": "ordered",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.19150002081347656,
            "peak_rss_mb": 989.446144
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.21903110115616584,
            "peak_rss_mb": 10601.332736
        },
        {
            "kind": "shuffled",
            "n_workers": 2,
            "batch_size": 16384,
            "batches_per_s": 0.22061147894482222,
            "peak_rss_mb": 10599.26016
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.780275254547799,
            "peak_rss_mb": 4217.163776
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.7604982125564428,
            "peak_rss_mb": 4295.102464
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.764064181238567,
            "peak_rss_mb": 3620.08576
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.7824270964663742,
            "peak_rss_mb": 4259.172352
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.7537574598441001,
            "peak_rss_mb": 3782.070272
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 2048,
            "batches_per_s": 1.7192344891449403,
            "peak_rss_mb": 4088.057856
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.4328911469309293,
            "peak_rss_mb": 12443.549696
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.8887202590653626,
            "peak_rss_mb": 4762.70592
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.4461681168160332,
            "peak_rss_mb": 8544.493568
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.8836592729393767,
            "peak_rss_mb": 4479.934464
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.87301950122104,
            "peak_rss_mb": 4766.715904
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.8765515772911954,
            "peak_rss_mb": 5422.505984
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.8721677609291993,
            "peak_rss_mb": 4919.205888
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.8911792509268682,
            "peak_rss_mb": 4691.869696
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.8749308860187598,
            "peak_rss_mb": 4636.54912
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.7307666429704258,
            "peak_rss_mb": 5308.329984
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.3639977185806344,
            "peak_rss_mb": 6875.68896
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.4376044803645876,
            "peak_rss_mb": 6821.666816
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.8732244043198831,
            "peak_rss_mb": 5018.353664
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.43679862293204325,
            "peak_rss_mb": 6815.690752
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.3663526707622586,
            "peak_rss_mb": 6905.929728
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.4397824034117375,
            "peak_rss_mb": 6824.026112
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.436174206434647,
            "peak_rss_mb": 6923.689984
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 4096,
            "batches_per_s": 0.8752665619935319,
            "peak_rss_mb": 4903.575552
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.4406422109481057,
            "peak_rss_mb": 7003.713536
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.36178079427472715,
            "peak_rss_mb": 6916.476928
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.43047098945514944,
            "peak_rss_mb": 7089.401856
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 8192,
            "batches_per_s": 0.42911341576397644,
            "peak_rss_mb": 6916.374528
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.22037217458438096,
            "peak_rss_mb": 11212.603392
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.22280384444016607,
            "peak_rss_mb": 11247.321088
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.21567472757815,
            "peak_rss_mb": 11255.697408
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.22289536742791644,
            "peak_rss_mb": 11275.481088
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.21891168897008473,
            "peak_rss_mb": 11316.256768
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.21639174590076657,
            "peak_rss_mb": 11303.780352
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.2202323201380976,
            "peak_rss_mb": 11222.900736
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.21701461882894765,
            "peak_rss_mb": 11233.468416
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.21702734745808971,
            "peak_rss_mb": 11240.443904
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 1.7356597052390155,
            "peak_rss_mb": 5952.417792
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 1.420123835531893,
            "peak_rss_mb": 5717.68832
        },
        {
            "kind": "shuffled",
            "n_workers": 4,
            "batch_size": 16384,
            "batches_per_s": 0.2158463003781542,
            "peak_rss_mb": 11108.786176
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 1.721580848628114,
            "peak_rss_mb": 5716.795392
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 1.7205380569999764,
            "peak_rss_mb": 5995.70432
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 1.7666213650568592,
            "peak_rss_mb": 5646.995456
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 1.7543677392384112,
            "peak_rss_mb": 5809.041408
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 1.7917981336511093,
            "peak_rss_mb": 5060.186112
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 1.5725676595074414,
            "peak_rss_mb": 5691.244544
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 1.4649432736931571,
            "peak_rss_mb": 5911.994368
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.8854576688189292,
            "peak_rss_mb": 7335.002112
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.8638129331309369,
            "peak_rss_mb": 6909.169664
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.8532947230493362,
            "peak_rss_mb": 7175.872512
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.8871578093322723,
            "peak_rss_mb": 6738.96448
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.8831969965767121,
            "peak_rss_mb": 5623.5008
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.7284659607787431,
            "peak_rss_mb": 6816.415744
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.8915313380821921,
            "peak_rss_mb": 6012.956672
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 2048,
            "batches_per_s": 1.6032887621094727,
            "peak_rss_mb": 5933.166592
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.8795629147187851,
            "peak_rss_mb": 6353.854464
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 4096,
            "batches_per_s": 0.8908131078733154,
            "peak_rss_mb": 7686.787072
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.4505018406616765,
            "peak_rss_mb": 8471.392256
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.2054222734932271,
            "peak_rss_mb": 12471.754752
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.450664903187573,
            "peak_rss_mb": 8422.223872
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.4331476270236273,
            "peak_rss_mb": 12182.503424
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.44241297327485457,
            "peak_rss_mb": 8729.96864
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.4126085963383185,
            "peak_rss_mb": 8919.560192
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.42806429294727727,
            "peak_rss_mb": 8691.597312
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.40905859768553376,
            "peak_rss_mb": 8639.488
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.43655059925390416,
            "peak_rss_mb": 8422.797312
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.4361796926048937,
            "peak_rss_mb": 8298.041344
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 8192,
            "batches_per_s": 0.418639030092868,
            "peak_rss_mb": 8498.8928
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.22048164778653037,
            "peak_rss_mb": 12498.644992
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.2208916350759805,
            "peak_rss_mb": 12592.39424
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 1.7081390292968495,
            "peak_rss_mb": 7661.371392
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.8491358846454102,
            "peak_rss_mb": 9933.262848
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.8564802289546398,
            "peak_rss_mb": 10011.455488
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 1.7799084030993828,
            "peak_rss_mb": 7414.894592
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 1.7372534405292217,
            "peak_rss_mb": 8444.719104
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 1.788479384962984,
            "peak_rss_mb": 7617.380352
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.6988864230230737,
            "peak_rss_mb": 9956.851712
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 1.730192221937866,
            "peak_rss_mb": 7831.781376
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 1.7404516361022946,
            "peak_rss_mb": 8018.55488
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.8621071115802147,
            "peak_rss_mb": 10148.139008
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 1.6565026142744026,
            "peak_rss_mb": 7886.0288
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 1.7255606692260177,
            "peak_rss_mb": 8382.021632
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.8342823412112924,
            "peak_rss_mb": 10138.386432
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.69916846163437,
            "peak_rss_mb": 11610.476544
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 1.6735674098515911,
            "peak_rss_mb": 7519.858688
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.8671381118799881,
            "peak_rss_mb": 9897.394176
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.4067942470091606,
            "peak_rss_mb": 12069.474304
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.17165134124451803,
            "peak_rss_mb": 12642.381824
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.7017334066668336,
            "peak_rss_mb": 9708.863488
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.7194003695900237,
            "peak_rss_mb": 12460.642304
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.7107141540956823,
            "peak_rss_mb": 12136.67328
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.43378582153842804,
            "peak_rss_mb": 10924.453888
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.8609190272444848,
            "peak_rss_mb": 10146.869248
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.8556399435021066,
            "peak_rss_mb": 10251.010048
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 2048,
            "batches_per_s": 1.7089036429158064,
            "peak_rss_mb": 8060.47744
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.43355691160087084,
            "peak_rss_mb": 12273.88928
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 4096,
            "batches_per_s": 0.8511170223262406,
            "peak_rss_mb": 9187.848192
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.2175517859839794,
            "peak_rss_mb": 16280.190976
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.2081385194738276,
            "peak_rss_mb": 12506.095616
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.2169759455637315,
            "peak_rss_mb": 15299.776512
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.43021833000490195,
            "peak_rss_mb": 11542.24128
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.7089103032040116,
            "peak_rss_mb": 11435.118592
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.18373142062528453,
            "peak_rss_mb": 15994.6752
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.20546524270586067,
            "peak_rss_mb": 12409.61024
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.43797127421229876,
            "peak_rss_mb": 12276.543488
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.43315866475306364,
            "peak_rss_mb": 11907.784704
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.21851165777172216,
            "peak_rss_mb": 15406.751744
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.21921142063632038,
            "peak_rss_mb": 15881.060352
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.43663430012991145,
            "peak_rss_mb": 11707.789312
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.17593248540287826,
            "peak_rss_mb": 15544.385536
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.43555399281819757,
            "peak_rss_mb": 11237.13024
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.21518823340991547,
            "peak_rss_mb": 14761.406464
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.20476061007005586,
            "peak_rss_mb": 12714.307584
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.2056770569978211,
            "peak_rss_mb": 12380.762112
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.20215271415586145,
            "peak_rss_mb": 12738.78528
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.21952747331854922,
            "peak_rss_mb": 14537.887744
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.20968332741537493,
            "peak_rss_mb": 16568.881152
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 8192,
            "batches_per_s": 0.43820753133923984,
            "peak_rss_mb": 12199.95648
        },
        {
            "kind": "shuffled",
            "n_workers": 16,
            "batch_size": 16384,
            "batches_per_s": 0.20624950926632002,
            "peak_rss_mb": 15679.5904
        },
        {
            "kind": "shuffled",
            "n_workers": 8,
            "batch_size": 16384,
            "batches_per_s": 0.20837276959896875,
            "peak_rss_mb": 12573.065216
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.6207706721653896,
            "peak_rss_mb": 12150.747136
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.7010850028031503,
            "peak_rss_mb": 11953.410048
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.6435831000944665,
            "peak_rss_mb": 12369.252352
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.8318569547759016,
            "peak_rss_mb": 14008.098816
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.6319812442110768,
            "peak_rss_mb": 11949.551616
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 2048,
            "batches_per_s": 1.7226063840150037,
            "peak_rss_mb": 12174.651392
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.434183736084986,
            "peak_rss_mb": 18714.390528
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.8288168544880324,
            "peak_rss_mb": 13922.59072
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.19932323635946902,
            "peak_rss_mb": 21866.47552
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.8321141840827729,
            "peak_rss_mb": 13824.315392
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.4359077072527241,
            "peak_rss_mb": 18957.811712
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.8320719616034349,
            "peak_rss_mb": 13780.537344
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.8513973721399263,
            "peak_rss_mb": 14426.189824
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.8309870115169136,
            "peak_rss_mb": 14349.049856
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.8321052630677311,
            "peak_rss_mb": 14925.7216
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.8286550041779788,
            "peak_rss_mb": 14121.566208
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.42981688936217005,
            "peak_rss_mb": 17777.90976
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.8472323002542645,
            "peak_rss_mb": 14795.583488
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.21998109544112607,
            "peak_rss_mb": 21794.828288
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.4426878925826106,
            "peak_rss_mb": 18733.461504
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.4136159572103559,
            "peak_rss_mb": 18915.704832
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.21909053830912853,
            "peak_rss_mb": 21317.169152
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 4096,
            "batches_per_s": 0.8210785609043231,
            "peak_rss_mb": 14528.970752
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.21474483710488015,
            "peak_rss_mb": 21001.814016
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.41432712586640186,
            "peak_rss_mb": 18257.481728
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.21386215963064983,
            "peak_rss_mb": 22588.86656
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.43397984214771873,
            "peak_rss_mb": 17844.445184
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.44194555914453393,
            "peak_rss_mb": 14307.233792
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.2183014082086592,
            "peak_rss_mb": 17956.384768
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.21606729326609328,
            "peak_rss_mb": 21614.051328
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.43101605754035816,
            "peak_rss_mb": 18033.168384
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.22022810225384598,
            "peak_rss_mb": 20775.903232
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 8192,
            "batches_per_s": 0.41710950598253166,
            "peak_rss_mb": 18571.026432
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.21996938078282124,
            "peak_rss_mb": 21387.780096
        },
        {
            "kind": "shuffled",
            "n_workers": 32,
            "batch_size": 16384,
            "batches_per_s": 0.2150472532463887,
            "peak_rss_mb": 19383.033856
        }
    ]
}
```

Is this fast enough for the shuffled dataloader? What commands would you execute to check?
