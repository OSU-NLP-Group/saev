""" """

import dataclasses
import logging
import os.path
import tomllib
import typing

import beartype
import submitit
import tyro
from lib import baselines

import saev.data
from saev import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    method: typing.Literal["random", "pca", "kmeans", "sae"] = "random"
    """Which method we are evaluating."""
    n_prototypes: int = 1024 * 32
    """Number of prototypes."""
    data: saev.data.IterableConfig = dataclasses.field(
        default_factory=saev.data.IterableConfig
    )
    """Train activations."""
    dump_to: str = os.path.join(".", "checkpoints")
    """Where to save trained checkpoints."""
    device: typing.Literal["cuda", "cpu"] = "cuda"
    """Hardware device."""
    seed: int = 17
    """Random seed."""
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 24.0
    """Slurm job length in hours."""
    log_to: str = os.path.join(".", "logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
def get_scorer(cfg: Config) -> baselines.Scorer:
    d_vit = saev.data.Metadata.load(cfg.data.shard_root).d_vit

    if cfg.method == "random":
        return baselines.RandomVectors(
            n_prototypes=cfg.n_prototypes, d=d_vit, seed=cfg.seed
        )
    elif cfg.method == "kmeans":
        return baselines.KMeans(n_prototypes=cfg.n_prototypes, d=d_vit, seed=cfg.seed)
    elif cfg.method == "pca":
        return baselines.PCA(n_components=cfg.n_prototypes, d=d_vit, seed=cfg.seed)
    else:
        raise RuntimeError(f"Method '{cfg.method}' not implemented.")


@beartype.beartype
def worker_fn(cfg: Config):
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("worker")

    try:
        dataloader = saev.data.iterable.DataLoader(cfg.data)
    except Exception:
        logger.exception(
            "Could not create dataloader. Please create a dataset using saev.data first. See src/saev/guide.md for more details."
        )
        return

    scorer = get_scorer(cfg)
    scorer.train(dataloader)

    fpath = os.path.join(
        cfg.dump_to,
        f"{cfg.method}__n_prototypes-{cfg.n_prototypes}__seed-{cfg.seed}.bin",
    )
    baselines.dump(scorer, fpath)


@beartype.beartype
def main(
    cfg: typing.Annotated[Config, tyro.conf.arg(name="")], sweep: str | None = None
):
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("main")

    if sweep is not None:
        with open(sweep, "rb") as fd:
            cfgs, errs = helpers.grid(cfg, tomllib.load(fd))

        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return
    else:
        cfgs = [cfg]

    if cfg.slurm_acct:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            # Request 8 CPUs because I want more memory and I don't know how else to get memory.
            cpus_per_task=8,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    with executor.batch():
        jobs = [executor.submit(worker_fn, c) for c in cfgs]

    logger.info("Submitted %d jobs.", len(jobs))
    for j, job in enumerate(jobs):
        job.result()
        logger.info("Job %d finished (%d/%d).", j, j + 1, len(jobs))

    logger.info("Done.")


if __name__ == "__main__":
    tyro.cli(main)
