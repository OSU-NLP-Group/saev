"""
Slurm benchmark harness for SAEv data loaders.

Usage
-----
uv run python bench/run_bench.py --shard-path /fs/scratch/.../cache/saev/<hash>
"""

import dataclasses
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
import typing

import beartype
import submitit
import tyro


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Result:
    kind: typing.Literal["torch", "iterable"]
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


@beartype.beartype
def benchmark_fn(
    kind: str,
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

    import saev.data.iterable
    import saev.data.torch

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("benchmark")

    if kind == "torch":
        cfg = saev.data.torch.Config(
            shard_root=shard_root,
            patches="patches",
            layer=layer,
            scale_mean=False,
            scale_norm=False,
        )
        ds = saev.data.torch.Dataset(cfg)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=True,
            persistent_workers=True,
        )
    elif kind == "iterable":
        dl = saev.data.iterable.DataLoader(
            saev.data.iterable.Config(
                shard_root=shard_root,
                patches="patches",
                layer=layer,
                batch_size=batch_size,
                n_threads=n_workers,
                seed=0,
            )
        )
    else:
        raise ValueError(kind)

    logger.info("kind: %s, cfg: %s", kind, cfg)

    # warm-up
    logger.info("Warming up for %d minutes.", warmup_min)
    end = time.perf_counter() + warmup_min * 60
    it = iter(dl)
    while time.perf_counter() < end:
        try:
            next(it)
        except StopIteration:
            it = iter(dl)
    logger.info("Warmed up.")

    # measured run
    n_batches = 0
    rss_max = 0
    proc = psutil.Process()
    end = time.perf_counter() + run_min * 60
    it = iter(dl)
    while time.perf_counter() < end:
        try:
            next(it)
        except StopIteration:
            it = iter(dl)
            next(it)

        n_batches += 1
        rss_max = max(rss_max, proc.memory_info().rss)
        logger.info("batches: %d, peak GB: %.1f", n_batches, rss_max / 1e9)

    bps = n_batches / (run_min * 60)
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
    minutes: int = 10,
    warmup: int = 5,
    out: pathlib.Path = pathlib.Path("logs", "benchmarking"),
    slurm_partition: str = "gpu",
    slurm_acct: str = "",
):
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
        time=minutes + warmup + 20,
        stderr_to_stdout=True,
    )
    jobs = []
    with ex.batch():
        # for kind in ["iterable", "torch"]:
        for kind in ["torch"]:
            for n_workers in [2, 4, 8, 16, 32, 64]:
                for batch_size in [4, 8, 16, 32]:
                    jobs.append(
                        ex.submit(
                            benchmark_fn,
                            kind,
                            shard_root=shards,
                            layer=layer,
                            n_workers=n_workers,
                            batch_size=batch_size * 1024,
                            warmup_min=warmup,
                            run_min=minutes,
                        )
                    )

    results = [j.result() for j in jobs]

    import saev.data.writers

    meta = saev.data.writers.Metadata.load(os.path.join(shards, "metadata.json"))
    payload = dict(
        meta=dataclasses.asdict(meta), results=[dataclasses.asdict(r) for r in results]
    )
    with open(out / "results.json", "w") as f:
        json.dump(payload, f, indent=2)


@beartype.beartype
def plot(results: pathlib.Path):
    import altair as alt
    import polars as pl

    alt.renderers.enable("png")

    with open(results) as fd:
        df = pl.DataFrame(json.load(fd)["results"]).with_columns(
            patches_per_s=pl.col("batch_size") * pl.col("batches_per_s")
        )

    alt.Chart(df).mark_line().encode(
        alt.X("n_workers", type="quantitative"),
        alt.Y("patches_per_s", type="quantitative"),
        alt.Color("batch_size", type="nominal"),
        alt.Shape("kind", type="nominal"),
        alt.Detail("kind", type="nominal"),
    ).save("scaling.png")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"benchmark": benchmark, "plot": plot})
