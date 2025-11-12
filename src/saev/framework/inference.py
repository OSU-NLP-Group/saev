"""
Script for dumping image-level activations in a single pass over the dataset.

Dumps 5 files:

1. mean_values.pt
2. sparsity.pt
3. distributions.pt
4. token_acts.npz
5. metrics.json

metrics.json contains some individual metrics that are useful for reporting:

* normalized_mse (float): reconstruction sum squared SAE error / reconstruction sum squared mean baseline error
* sse_sae (float): reconstruction sum squared SAE error, which is ((x - x_hat) ** 2).sum() to get a scalar where x_hat is the output of SAE(x).
* sse_baseline (float): reconstruction sum squared mean baseline error, which is ((x - x_bar) ** 2).sum(dim=0) to get a scalar where x_bar is the mean vector of all x.
"""

import collections.abc
import dataclasses
import logging
import os
import pathlib
import sys
import time
import typing as tp

import beartype
import einops
import orjson
import scipy.sparse
import torch
import tyro

from saev import configs, disk, helpers, nn
from saev.data import Metadata, OrderedConfig, OrderedDataLoader

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("inference.py")


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
    mem_gb: int = 80
    """Node memory in GB."""
    log_to: str = os.path.join(".", "logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Filepaths:
    mean_values: pathlib.Path
    sparsity: pathlib.Path
    distributions: pathlib.Path
    token_acts: pathlib.Path
    metrics: pathlib.Path

    @classmethod
    def from_run(cls, run: disk.Run, md: Metadata):
        root = run.inference / md.hash
        root.mkdir(exist_ok=True, parents=True)
        return cls(
            mean_values=root / "mean_values.pt",
            sparsity=root / "sparsity.pt",
            distributions=root / "distributions.pt",
            token_acts=root / "token_acts.npz",
            metrics=root / "metrics.json",
        )

    def __iter__(self) -> collections.abc.Iterator[pathlib.Path]:
        yield from [
            self.mean_values,
            self.sparsity,
            self.distributions,
            self.token_acts,
            self.metrics,
        ]


@beartype.beartype
def need_compute(cfg: Config) -> tuple[bool, str, Filepaths]:
    # Check if we need to compute activations
    run = disk.Run(cfg.run)
    md = Metadata.load(cfg.data.shards)
    fpaths = Filepaths.from_run(run, md)
    missing = [fpath for fpath in fpaths if not fpath.exists()]

    if not cfg.force_recompute and not missing:
        reason = "Found all files."
        return False, reason, fpaths

    if cfg.force_recompute:
        reason = "Force recompute flag set; computing activations."
        return True, reason, fpaths

    missing_msg = ", ".join(str(f) for f in missing)
    return True, f"Missing files {missing_msg}; computing activations.", fpaths


@beartype.beartype
@torch.inference_mode()
def worker_fn(cfg: Config):
    run = disk.Run(cfg.run)
    md = Metadata.load(cfg.data.shards)
    root = run.inference / md.hash

    # Check if we need to compute activations
    do, reason, fpaths = need_compute(cfg)
    logger.info(reason)
    if not do:
        return

    with open(root / "config.json", "wb") as fd:
        helpers.jdump(cfg, fd)

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
    ignore = torch.tensor(cfg.ignore_labels)

    # Float64 accumulators keep NMSE numerics stable when evaluating Q - |S|^2 / N.
    reconstruction_sse = torch.zeros((), dtype=torch.float64, device=cfg.device)
    sum_sq = torch.zeros((), dtype=torch.float64, device=cfg.device)
    sum_vec_s = torch.zeros((sae.cfg.d_model,), dtype=torch.float64, device=cfg.device)
    n_tokens = 0

    logger.info("Loaded SAE and data.")

    prev_i = -1

    dataloader_iter = tp.cast(
        collections.abc.Iterable[dict[str, torch.Tensor]], dataloader
    )

    for batch in helpers.progress(dataloader_iter):
        # Get SAE activations (shared computation)
        vit_acts_bd = batch["act"].to(cfg.device)
        # Here, p stands for prefixes
        x_hat_bpd, f_x, *_ = sae(vit_acts_bd)
        bsz, d_sae = f_x.shape

        mask_b = torch.isin(batch["token_labels"], ignore, invert=True)

        n_batch_tokens = mask_b.sum().item()
        n_tokens += n_batch_tokens
        if n_batch_tokens > 0:
            vit_masked_bd = vit_acts_bd[mask_b].to(torch.float64)
            diff_bd = vit_masked_bd - x_hat_bpd[:, -1, :][mask_b].to(torch.float64)
            reconstruction_sse += torch.sum(diff_bd * diff_bd)
            sum_sq += (vit_masked_bd * vit_masked_bd).sum()
            sum_vec_s += vit_masked_bd.sum(dim=0)

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

    # Sparse CSR array
    token_acts = scipy.sparse.vstack(token_acts_blocks, format="csr")
    scipy.sparse.save_npz(fpaths.token_acts, token_acts)

    # Statistics
    torch.save(mean_values_s.cpu(), fpaths.mean_values)
    torch.save(sparsity_s.cpu(), fpaths.sparsity)
    torch.save(distributions_nm.cpu(), fpaths.distributions)

    # Metrics
    sse_sae = reconstruction_sse.item()
    sum_sq_item = sum_sq.item()
    sum_vec_s_sq = torch.dot(sum_vec_s, sum_vec_s).item()
    sse_baseline = sum_sq_item - sum_vec_s_sq / n_tokens
    if sse_baseline <= 0.0:
        raise RuntimeError(
            f"Baseline variance is non-positive (sse_baseline={sse_baseline:.6e}); cannot compute normalized MSE."
        )

    metrics = {
        "normalized_mse": sse_sae / sse_baseline,
        "sse_sae": sse_sae,
        "sse_baseline": sse_baseline,
    }

    with open(fpaths.metrics, "wb") as fd:
        helpers.jdump(metrics, fd, option=orjson.OPT_INDENT_2)


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

    if not cfg.slurm_acct:
        for i, cfg_item in enumerate(cfgs, start=1):
            logger.info("Running config %d/%d locally.", i, len(cfgs))
            worker_fn(cfg_item)
        logger.info("Jobs done.")
        return 0

    import submitit
    from submitit.core.utils import UncompletedJobError

    executor = submitit.SlurmExecutor(folder=cfg.log_to)

    n_cpus = 8
    if cfg.mem_gb // 10 > n_cpus:
        logger.info(
            "Using %d CPUs instead of %d to get more RAM.", cfg.mem_gb // 10, n_cpus
        )
        n_cpus = cfg.mem_gb // 10

    executor.update_parameters(
        time=int(cfg.n_hours * 60),
        partition=cfg.slurm_partition,
        gpus_per_node=1,
        ntasks_per_node=1,
        cpus_per_task=n_cpus,
        stderr_to_stdout=True,
        account=cfg.slurm_acct,
    )
    with executor.batch():
        jobs = []
        for i, cfg in enumerate(cfgs):
            do, reason, _ = need_compute(cfg)
            if not do:
                continue

            logger.info(reason)
            jobs.append(executor.submit(worker_fn, cfg))

    time.sleep(5.0)

    for i, job in enumerate(jobs, start=1):
        logger.info("Job %d/%d: %s %s", i, len(jobs), job.job_id, job.state)

    for i, job in enumerate(jobs, start=1):
        try:
            job.result()
            logger.info("Job %d/%d finished.", i, len(jobs))
        except UncompletedJobError:
            logger.warning("Job %s (%d) did not finish.", job.job_id, i)

    logger.info("Jobs done.")
    return 0
