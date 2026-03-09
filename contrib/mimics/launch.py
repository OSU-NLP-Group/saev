"""
Launcher script for Cambridge mimicry utilities.
"""

import logging
import pathlib
import typing as tp

import beartype
import mimics.scoring
import tyro
import tyro.extras

import saev.configs

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("mimics.launch")


@beartype.beartype
def score_cli(
    cfg: tp.Annotated[mimics.scoring.Config, tyro.conf.arg(name="")],
    sweep: pathlib.Path | None = None,
) -> int:
    """Score all SAE latents for mimic pair discrimination.

    Runs locally or submits a Slurm job when --slurm-acct is set.

    Args:
        cfg: Score configuration.
        sweep: Optional path to a Python sweep file with `make_cfgs() -> list[dict]`.
    """
    if sweep is not None:
        sweep_dcts = saev.configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            return 1

        cfgs, errs = saev.configs.load_cfgs(
            cfg, default=mimics.scoring.Config(), sweep_dcts=sweep_dcts
        )
        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return 1
    else:
        cfgs = [cfg]

    if not cfgs:
        logger.error("No score configs resolved.")
        return 1

    assert all(c.slurm_acct == cfgs[0].slurm_acct for c in cfgs)

    if not cfgs[0].slurm_acct:
        for i, cfg_item in enumerate(cfgs, start=1):
            logger.info("Running config %d/%d locally.", i, len(cfgs))
            mimics.scoring.worker_fn(cfg_item)
        logger.info("Jobs done.")
        return 0

    import time

    import submitit

    executor = submitit.SlurmExecutor(folder=cfgs[0].log_to)
    executor.update_parameters(
        job_name="mimic-score",
        time=int(cfgs[0].n_hours * 60),
        ntasks_per_node=1,
        mem=f"{cfgs[0].mem_gb}GB",
        stderr_to_stdout=True,
        account=cfgs[0].slurm_acct,
        partition=cfgs[0].slurm_partition,
    )
    with executor.batch():
        jobs = [executor.submit(mimics.scoring.worker_fn, c) for c in cfgs]

    time.sleep(3.0)
    for i, job in enumerate(jobs, start=1):
        logger.info("Job %d/%d: %s (%s).", i, len(jobs), job.job_id, job.state)
    logger.info("Submitted %d scoring job(s).", len(jobs))
    return 0


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "score": score_cli,
    })
