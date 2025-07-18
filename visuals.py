import typing

import beartype
import submitit
import tyro

from saev.scripts.visuals import Config, dump_imgs


@beartype.beartype
def main(cfg: typing.Annotated[Config, tyro.conf.arg(name="")]):
    if cfg.slurm_acct:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=cfg.data.n_threads + 4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    job = executor.submit(dump_imgs, cfg)
    print(f"Running job {job.job_id}.")
    job.result()
    print("Job's done.")


if __name__ == "__main__":
    tyro.cli(main)
