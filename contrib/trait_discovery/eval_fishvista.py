"""
Unified pipeline for training and evaluating trait discovery methods on FishVista.

This script combines training and evaluation for fast methods (RandomVectors, PCA, KMeans) and evaluation-only for pre-trained SAEs. Unlike CUB200 which has image-level annotations, FishVista has patch-level segmentation masks, allowing for direct patch-level evaluation.

Size key:
* B: Batch size
* D: ViT activation dimension (typically 768 or 1024)
* K: Number of prototypes/components
* N: Number of images
* P: Number of patches (typically 640 for FishVista)
* H: Height in pixels
* W: Width in pixels
* C: Number of segmentation classes (10 for FishVista)
"""

import tomllib
import typing as tp

import beartype
import tdiscovery.fishvista.evaluation
import tyro

import saev.helpers


@beartype.beartype
def main(
    cfg: tp.Annotated[tdiscovery.fishvista.evaluation.Config, tyro.conf.arg(name="")],
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
                executor.submit(tdiscovery.fishvista.evaluation.evaluate, c)
                for c in cfgs
            ]

        print(f"Submitted {len(jobs)} job(s).")
        for j, job in enumerate(jobs):
            job.result()
            print(f"Job {job.job_id} finished ({j + 1}/{len(jobs)}).")
    else:
        for c in cfgs:
            tdiscovery.fishvista.evaluation.evaluate(c)

    print("Jobs done.")


if __name__ == "__main__":
    tyro.cli(main)
