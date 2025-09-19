"""
Script for dumping image-level activations in a single pass over the dataset.

Dumps 5 files:

1. mean_values.pt
2. sparsity.pt
3. distributions.pt
4. img_acts.pt
5. percentiles_p99.pt
"""

import dataclasses
import json
import logging
import os
import tomllib
import typing as tp

import beartype
import einops
import torch
import tyro

from saev import helpers, nn
from saev.data import Metadata, OrderedConfig, OrderedDataLoader
from saev.utils import statistics

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("activations.py")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for computing image activations."""

    # Disk
    ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    data: OrderedConfig = dataclasses.field(default_factory=OrderedConfig)
    """Data configuration"""
    dump_to: str = os.path.join(".", "data")
    """Where to save data."""

    n_distributions: int = 25
    """Number of features to save distributions for."""
    percentile: int = 99
    """Percentile to estimate for outlier detection."""
    top_k: int = 1
    """How many patches per image to use to compute the maximum image activation per latent."""
    ignore_labels: list[int] = dataclasses.field(default_factory=OrderedConfig)
    """Which patch labels to ignore when calculating summarized image activations."""

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

    # Properties for file paths
    @property
    def root(self) -> str:
        ckpt_id = os.path.basename(os.path.dirname(self.ckpt))
        return os.path.join(self.dump_to, ckpt_id)


@beartype.beartype
@torch.inference_mode()
def worker_fn(cfg: Config):
    os.makedirs(cfg.root, exist_ok=True)
    cfg_fpath = os.path.join(cfg.root, "config.json")
    mean_values_fpath = os.path.join(cfg.root, "mean_values.pt")
    sparsity_fpath = os.path.join(cfg.root, "sparsity.pt")
    distributions_fpath = os.path.join(cfg.root, "distributions.pt")
    img_acts_fpath = os.path.join(cfg.root, "img_acts.pt")
    percentiles_fpath = os.path.join(cfg.root, f"percentiles_p{cfg.percentile}.pt")

    # Check if we need to compute activations
    fpaths = [
        mean_values_fpath,
        sparsity_fpath,
        distributions_fpath,
        percentiles_fpath,
        img_acts_fpath,
    ]
    missing = [fpath for fpath in fpaths if not os.path.exists(fpath)]
    need_compute = cfg.force_recompute or bool(missing)

    if not need_compute:
        logger.info("Found existing activation files.")
        return

    if cfg.force_recompute:
        logger.info("Force recompute flag set; computing activations.")
    else:
        logger.info("Missing files %s; computing activations.", ", ".join(missing))

    with open(cfg_fpath, "w") as fd:
        json.dump(dataclasses.asdict(cfg), fd)

    assert cfg.data.patches == "image"
    sae = nn.load(cfg.ckpt).to(cfg.device)
    md = Metadata.load(cfg.data.shard_root)

    sparsity_s = torch.zeros((sae.cfg.d_sae,), device=cfg.device)
    mean_values_s = torch.zeros((sae.cfg.d_sae,), device=cfg.device)

    # Initialize image-level activations
    img_acts_ns = torch.zeros((md.n_imgs, sae.cfg.d_sae), device=cfg.device)

    batch_size = cfg.data.batch_size // md.n_patches_per_img * md.n_patches_per_img
    n_imgs_per_batch = batch_size // md.n_patches_per_img
    dataloader = OrderedDataLoader(
        dataclasses.replace(cfg.data, batch_size=batch_size),
    )

    distributions_nm = torch.zeros(
        (dataloader.n_samples, cfg.n_distributions), device=cfg.device
    )
    estimator = statistics.PercentileEstimator(
        cfg.percentile, dataloader.n_samples, shape=(sae.cfg.d_sae,)
    )

    logger.info("Loaded SAE and data.")

    for batch in helpers.progress(dataloader, desc="activations"):
        # Get SAE activations (shared computation)
        vit_acts_bd = batch["act"].to(cfg.device)
        _, sae_acts_bs, *_ = sae(vit_acts_bd)

        # Update percentile estimator
        # TODO/BUG: ignore samples that have patch labels in cfg.ignore_labels
        keep_b = batch["patch_labels"].isin(cfg.ignore_labels, invert=True)
        for sae_act_s in sae_acts_bs[keep_b]:
            estimator.update(sae_act_s)

        # Update statistics
        distributions_nm[batch["image_i"][keep_b], :] = sae_acts_bs[
            keep_b, : cfg.n_distributions
        ]
        mean_values_s += einops.reduce(sae_acts_bs, "batch d_sae -> d_sae", "sum")
        sparsity_s += einops.reduce((sae_acts_bs > 0), "batch d_sae -> d_sae", "sum")

        # Get unique image indices in this batch
        img_i = torch.sort(torch.unique(batch["image_i"])).values.to(cfg.device)

        values_ips = einops.rearrange(
            sae_acts_bs,
            "(n_img n_patches) d_sae -> n_img n_patches d_sae",
            n_img=len(img_i),
            n_patches=md.n_patches_per_img,
        )

        # Validation
        assert values_ips.shape[0] == img_i.shape[0]
        if not len(img_i) == n_imgs_per_batch:
            logger.warning(
                "Got %d images; expected %d images per batch.",
                len(img_i),
                n_imgs_per_batch,
            )

        # Take the mean of the top k patches as our image-level summary.
        acts_iks = torch.topk(values_ips, cfg.top_k, dim=1).values
        acts_is = einops.reduce(acts_iks, "n_img k d_sae -> n_img d_sae", "mean")
        assert (img_acts_ns[img_i] == 0).all(), "overwriting non-zero activations"
        img_acts_ns[img_i, :] = acts_is

    # Finalize statistics
    mean_values_s /= sparsity_s
    sparsity_s /= dataloader.n_samples

    torch.save(mean_values_s.cpu(), mean_values_fpath)
    torch.save(sparsity_s.cpu(), sparsity_fpath)
    torch.save(distributions_nm.cpu(), distributions_fpath)
    torch.save(estimator.estimate.cpu(), percentiles_fpath)
    torch.save(img_acts_ns.cpu(), img_acts_fpath)


@beartype.beartype
def main(cfg: tp.Annotated[Config, tyro.conf.arg(name="")], sweep: str | None = None):
    """Main entry point."""

    if sweep is not None:
        with open(sweep, "rb") as fd:
            cfgs, errs = helpers.grid(cfg, tomllib.load(fd))

        if errs:
            for err in errs:
                logger.error(f"Error in config: {err}")
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
