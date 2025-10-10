"""
Slurm benchmark harness for saev data loaders.

Usage
-----
uv run python bench/run_bench.py --shard-path /fs/scratch/.../cache/saev/<hash>
"""

import dataclasses
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
    profile: bool = False,
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

    # Start profiling if enabled
    if profile:
        from viztracer import VizTracer

        tracer = VizTracer(output_file=f"{kind}_profile.json", log_torch=True)
        tracer.start()
        logger.info("Started VizTracer profiling for %s", kind)

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

    del dl

    # Stop profiling if enabled
    if profile:
        tracer.stop()
        tracer.save()
        logger.info("Saved VizTracer profile to %s_profile.json", kind)

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
    profile: bool = False,
):
    import saev.data.shards
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
            if profile:
                # When profiling, submit just 3 jobs with fixed parameters
                logger.info("Profiling mode: submitting 3 jobs for profiling")
                job = ex.submit(
                    benchmark_fn,
                    kind,
                    shard_root=shards,
                    layer=layer,
                    n_workers=4,  # Fixed for profiling
                    batch_size=16 * 1024,  # Fixed 16k for profiling
                    warmup_min=warmup_min,
                    run_min=run_min,
                    profile=True,
                )
                jobs.append(job)
            else:
                # Normal benchmarking mode with full grid
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
                                profile=False,
                            )
                            jobs.append(job)

    logger.info("Submitted %d jobs.", len(jobs))

    if profile:
        logger.info("Profile files will be saved to %s/", out / commit)
        logger.info("Monitor jobs with: squeue --me")
        logger.info("View profiles with: vizviewer <profile>.json")

    results = []
    for j, job in enumerate(saev.helpers.progress(jobs)):
        try:
            results.append(job.result())
        except submitit.core.utils.UncompletedJobError:
            logger.warning("Job %d did not finish.", j)

    if not profile:  # Only save results.json for non-profile runs
        meta = saev.data.shards.Metadata.load(shards)
        payload = dict(
            meta=dataclasses.asdict(meta),
            results=[dataclasses.asdict(r) for r in results],
        )
        with open(out / "results.json", "wb") as fd:
            saev.helpers.dump(payload, fd)


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
