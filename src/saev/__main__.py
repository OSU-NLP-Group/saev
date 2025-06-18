import logging
import tomllib
import typing

import beartype
import tyro

from . import config

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger("saev")


@beartype.beartype
def activations(cfg: typing.Annotated[config.Activations, tyro.conf.arg(name="")]):
    """
    Save ViT activations for use later on.

    Args:
        cfg: Configuration for activations.
    """
    import saev.activations

    saev.activations.main(cfg)


@beartype.beartype
def train(
    cfg: typing.Annotated[config.Train, tyro.conf.arg(name="")],
    sweep: str | None = None,
):
    """
    Train an SAE over activations, optionally running a parallel grid search over a set of hyperparameters.

    Args:
        cfg: Baseline config for training an SAE.
        sweep: Path to .toml file defining the sweep parameters.
    """
    import submitit

    from . import config, training

    if sweep is not None:
        with open(sweep, "rb") as fd:
            cfgs, errs = config.grid(cfg, tomllib.load(fd))

        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return

    else:
        cfgs = [cfg]

    cfgs = training.split_cfgs(cfgs)

    logger.info("Running %d training jobs.", len(cfgs))

    if cfg.slurm_acct:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(4500),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=cfg.n_workers + 4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    jobs = [executor.submit(training.main, group) for group in cfgs]
    for job in jobs:
        job.result()


@beartype.beartype
def visuals(cfg: typing.Annotated[config.Visuals, tyro.conf.arg(name="")]):
    """
    Save maximally activating images for each SAE latent.

    Args:
        cfg: Config
    """
    from . import visuals

    visuals.main(cfg)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "activations": activations,
        "train": train,
        "visuals": visuals,
    })
    logger.info("Done.")
