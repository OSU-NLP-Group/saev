import time

import beartype
import submitit
import tyro
from tdiscovery.fishvista.supervised import Config, train


@beartype.beartype
def main(cfg: Config):
    if cfg.slurm_acct:
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
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    job = executor.submit(train, cfg)

    # Give the executor five seconds to fire the jobs off.
    time.sleep(5.0)
    print(f"Job {job.job_id} {job.state}")
    job.result()
    print("Job done.")


if __name__ == "__main__":
    main(tyro.cli(Config))
