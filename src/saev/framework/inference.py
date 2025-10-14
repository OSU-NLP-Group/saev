"""
Script for dumping image-level activations in a single pass over the dataset.

Dumps 5 files:

1. mean_values.pt
2. sparsity.pt
3. distributions.pt
4. token_acts.npz
5. percentiles_p99.pt
"""

import dataclasses
import logging
import os
import pathlib
import sys
import typing as tp

import beartype
import einops
import scipy.sparse
import torch
import tyro

from saev import configs, disk, helpers, nn
from saev.data import Metadata, OrderedConfig, OrderedDataLoader
from saev.utils import statistics

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("activations.py")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for computing image activations."""

    run: pathlib.Path = pathlib.Path("./runs/abcdefg")
    """Path to the sae.pt file."""
    data: OrderedConfig = OrderedConfig()
    """Data configuration"""

    n_dists: int = 25
    """Number of features to save distributions for."""
    percentile: int = 99
    """Percentile to estimate for outlier detection."""
    ignore_labels: list[int] = dataclasses.field(default_factory=list)
    """Which token labels to ignore when calculating summarized image activations."""

    # Control flags
    force_recompute: bool = False
    """Force recomputation even if files exist."""

    # Hardware
    device: str = "cuda"
    """Which accelerator to use."""
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 4.0
    """Slurm job length in hours."""
    log_to: str = os.path.join(".", "logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
@torch.inference_mode()
def worker_fn(cfg: Config):
    run = disk.Run(cfg.run)
    md = Metadata.load(cfg.data.shards)
    root = run.inference / md.hash
    mean_values_fpath = root / "mean_values.pt"
    sparsity_fpath = root / "sparsity.pt"
    distributions_fpath = root / "distributions.pt"
    token_acts_fpath = root / "token_acts.npz"
    percentiles_fpath = root / f"percentiles_p{cfg.percentile}.pt"

    # Check if we need to compute activations
    fpaths = [
        mean_values_fpath,
        sparsity_fpath,
        distributions_fpath,
        percentiles_fpath,
        token_acts_fpath,
    ]
    missing = [fpath for fpath in fpaths if not fpath.exists()]
    need_compute = cfg.force_recompute or bool(missing)

    if not need_compute:
        logger.info("Found existing activation files.")
        return

    if cfg.force_recompute:
        logger.info("Force recompute flag set; computing activations.")
    else:
        logger.info(
            "Missing files %s; computing activations.",
            ", ".join(str(f) for f in missing),
        )

    with open(run.inference / "config.json", "wb") as fd:
        helpers.dump(cfg, fd)

    assert cfg.data.tokens == "content"
    sae = nn.load(run.ckpt).to(cfg.device)

    sparsity_s = torch.zeros((sae.cfg.d_sae,), device=cfg.device)
    mean_values_s = torch.zeros((sae.cfg.d_sae,), device=cfg.device)

    # Initialize image-level activations
    token_acts_blocks = []

    batch_size = (
        cfg.data.batch_size
        // md.content_tokens_per_example
        * md.content_tokens_per_example
    )
    dataloader = OrderedDataLoader(
        dataclasses.replace(cfg.data, batch_size=batch_size),
    )

    distributions_nm = torch.zeros(
        (dataloader.n_samples, cfg.n_dists), device=cfg.device
    )
    estimator = statistics.PercentileEstimator(
        cfg.percentile, dataloader.n_samples, shape=(sae.cfg.d_sae,)
    )

    ignore = torch.tensor(cfg.ignore_labels)

    logger.info("Loaded SAE and data.")

    prev_i = -1

    for batch in helpers.progress(dataloader, desc="activations"):
        # Get SAE activations (shared computation)
        vit_acts_bd = batch["act"].to(cfg.device)
        _, f_x, *_ = sae(vit_acts_bd)
        bsz, d_sae = f_x.shape

        mask_b = torch.isin(batch["token_labels"], ignore, invert=True)

        # Update percentile estimator
        for sae_act_s in f_x[mask_b]:
            estimator.update(sae_act_s)

        # Update statistics
        distributions_nm[batch["example_idx"][mask_b], :] = f_x[mask_b, : cfg.n_dists]
        mean_values_s += einops.reduce(f_x[mask_b], "batch d_sae -> d_sae", "sum")
        sparsity_s += einops.reduce((f_x[mask_b] > 0), "batch d_sae -> d_sae", "sum")

        batch_idx = (
            batch["example_idx"] * md.content_tokens_per_example + batch["token_idx"]
        )
        # Assert that batch_idx[0].item() == prev_i + 1
        assert batch_idx[0].item() == prev_i + 1
        # Assert that batch_idx is is already sorted
        assert (torch.sort(batch_idx).values == batch_idx).all()
        # Assert that batch_idx is a continuous range from start, ed
        assert (torch.arange(batch_idx[0], batch_idx[-1] + 1) == batch_idx).all()

        # All masked tokens can have at most no activation.
        f_x[~mask_b, :] = 0.0

        token_acts_blocks.append(scipy.sparse.csr_array(f_x.cpu().numpy()))
        prev_i = batch_idx[-1].item()

    # Finalize statistics
    mean_values_s /= sparsity_s
    sparsity_s /= dataloader.n_samples

    torch.save(mean_values_s.cpu(), mean_values_fpath)
    torch.save(sparsity_s.cpu(), sparsity_fpath)
    torch.save(distributions_nm.cpu(), distributions_fpath)
    torch.save(estimator.estimate.cpu(), percentiles_fpath)
    # Sparse CSR array
    token_acts = scipy.sparse.vstack(token_acts_blocks, format="csr")
    scipy.sparse.save_npz(token_acts_fpath, token_acts)


@beartype.beartype
def main(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")], sweep: pathlib.Path | None = None
):
    """
    Run SAE inference over transformer activations, optionally using a sweep file to submit many jobs at once.

    Args:
        cfg: Baseline config inference.
        sweep: Path to .py file defining the sweep parameters.
    """

    if sweep is not None:
        sweep_dcts = configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            sys.exit(1)

        cfgs, errs = configs.load_cfgs(cfg, default=Config(), sweep_dcts=sweep_dcts)

        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
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
            jobs = [executor.submit(worker_fn, c) for c in cfgs]

        logger.info(f"Submitted {len(jobs)} job(s).")
        for j, job in enumerate(jobs):
            job.result()
            logger.info(f"Job {job.job_id} finished ({j + 1}/{len(jobs)}).")
    else:
        for c in cfgs:
            worker_fn(c)

    logger.info("Jobs done.")
