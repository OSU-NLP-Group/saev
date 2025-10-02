"""
To save lots of activations, we want to do things in parallel, with lots of slurm jobs, and save multiple files, rather than just one.

This script handles that additional complexity.

Conceptually, activations are either thought of as

1. A single [n_imgs x n_layers x (n_patches + 1), d_vit] tensor. This is a *dataset*
2. Multiple [n_imgs_per_shard, n_layers, (n_patches + 1), d_vit] tensors. This is a set of sharded activations.
"""

import dataclasses
import logging
import pathlib
import typing as tp

import beartype
import tyro

from saev.data import datasets

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger("saev.data")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for calculating and saving ViT activations."""

    data: datasets.Config = dataclasses.field(default_factory=datasets.Imagenet)
    """Which dataset to use."""
    shards_root: pathlib.Path = pathlib.Path("$SAEV_SCRATCH/saev/shards/")
    """Where to write shards."""
    family: tp.Literal["clip", "siglip", "dinov2", "dinov3", "fake-clip"] = "clip"
    """Which model family."""
    ckpt: str = "ViT-L-14/openai"
    """Specific model checkpoint."""
    batch_size: int = 1024
    """Batch size for ViT inference."""
    n_workers: int = 8
    """Number of dataloader workers."""
    d_model: int = 1024
    """Dimension of the ViT activations (depends on model)."""
    layers: list[int] = dataclasses.field(default_factory=lambda: [-2])
    """Which layers to save. By default, the second-to-last layer."""
    patches_per_ex: int = 256
    """Number of transformer patches per example (depends on model)."""
    cls_token: bool = True
    """Whether the model has a [CLS] token."""
    pixel_agg: tp.Literal["majority", "prefer-fg"] = "majority"
    patches_per_shard: int = 2_400_000
    """Maximum number of activations per shard; 2.4M is approximately 10GB for 1024-dimensional 4-byte activations."""

    ssl: bool = True
    """Whether to use SSL."""

    # Hardware
    device: str = "cuda"
    """Which device to use."""
    n_hours: float = 24.0
    """Slurm job length."""
    slurm_acct: str = ""
    """Slurm account string."""
    slurm_partition: str = ""
    """Slurm partition."""
    log_to: str = "./logs"
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
def main(cfg: tp.Annotated[Config, tyro.conf.arg(name="")]):
    """
    Save ViT activations for use later on.

    Args:
        cfg: Configuration for activations.
    """
    logger = logging.getLogger("dump")

    if not cfg.ssl:
        logger.warning("Ignoring SSL certs. Try not to do this!")
        # https://github.com/openai/whisper/discussions/734#discussioncomment-4491761
        # Ideally we don't have to disable SSL but we are only downloading weights.
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

    from saev.data import shards

    kwargs = dict(
        family=cfg.family,
        ckpt=cfg.ckpt,
        patches_per_ex=cfg.patches_per_ex,
        cls_token=cfg.cls_token,
        d_model=cfg.d_model,
        layers=cfg.layers,
        data=cfg.data,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_workers,
        patches_per_shard=cfg.patches_per_shard,
        shards_root=cfg.shards_root,
        device=cfg.device,
        pixel_agg=cfg.pixel_agg,
    )

    # Actually record activations.
    if cfg.slurm_acct:
        import submitit

        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=cfg.n_workers + 4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )

        job = executor.submit(shards.worker_fn, **kwargs)
        logger.info("Running job '%s'.", job.job_id)
        job.result()

    else:
        shards.worker_fn(**kwargs)


if __name__ == "__main__":
    tyro.cli(main)
