"""
Launcher script for activation computation with submitit support.
"""

import tomllib
import typing as tp

import beartype
import saev.scripts.activations
import tyro

import saev.helpers


@beartype.beartype
def main(
    cfg: tp.Annotated[saev.scripts.activations.Config, tyro.conf.arg(name="")],
    sweep: str | None = None,
):
    """Main entry point."""

    if sweep is not None:
        with open(sweep, "rb") as fd:
            cfgs, errs = saev.helpers.grid(cfg, tomllib.load(fd))

        if errs:
            for err in errs:
                print(f"Error in config: {err}")
            return
    else:
        cfgs = [cfg]

    assert all(c.slurm_acct == cfgs[0].slurm_acct for c in cfgs)
    cfg = cfgs[0]

    if cfg.slurm_acct:
        import submitit

        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=8,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
        with executor.batch():
            jobs = [
                executor.submit(saev.scripts.activations.worker_fn, c) for c in cfgs
            ]

        print(f"Submitted {len(jobs)} job(s).")
        for j, job in enumerate(jobs):
            job.result()
            print(f"Job {job.job_id} finished ({j + 1}/{len(jobs)}).")
    else:
        for c in cfgs:
            saev.scripts.activations.worker_fn(c)

    print("Jobs done.")


if __name__ == "__main__":
    tyro.cli(main)
