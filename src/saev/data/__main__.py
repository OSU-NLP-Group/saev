"""
To save lots of activations, we want to do things in parallel, with lots of slurm jobs, and save multiple files, rather than just one.

This module handles that additional complexity.

Conceptually, activations are either thought of as

1. A single [n_imgs x n_layers x (n_patches + 1), d_vit] tensor. This is a *dataset*
2. Multiple [n_imgs_per_shard, n_layers, (n_patches + 1), d_vit] tensors. This is a set of sharded activations.
"""

import logging
import typing

import beartype
import tyro

from . import writers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger("saev.data")


@beartype.beartype
def main(cfg: typing.Annotated[writers.Config, tyro.conf.arg(name="")]):
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

        job = executor.submit(writers.worker_fn, cfg)
        logger.info("Running job '%s'.", job.job_id)
        job.result()

    else:
        writers.worker_fn(cfg)


if __name__ == "__main__":
    tyro.cli(main)
