"""
Unified script for computing both top-k visual patches and image-level activations in a single pass over the dataset. This combines functionality from visuals.py and dump.py to avoid redundant computation.
"""

import dataclasses
import logging
import os

import beartype
import einops
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

import saev.data
from saev import helpers, nn
from saev.utils import statistics

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("unified_acts")


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Configuration for unified activation computation."""

    # Disk
    ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    data: saev.data.OrderedConfig = dataclasses.field(
        default_factory=saev.data.OrderedConfig
    )
    """Data configuration"""
    dump_to: str = os.path.join(".", "data")
    """Where to save data."""

    # Algorithm - visuals specific
    top_k: int = 128
    """How many images per SAE feature to store."""
    epsilon: float = 1e-12
    """Value to add to avoid log(0)."""
    log_freq_range: tuple[float, float] = (-6.0, -2.0)
    """Log10 frequency range for which to save images."""
    log_value_range: tuple[float, float] = (-1.0, 1.0)
    """Log10 frequency range for which to save images."""
    include_latents: list[int] = dataclasses.field(default_factory=list)
    """Latents to always include, no matter what."""
    n_distributions: int = 25
    """Number of features to save distributions for."""
    percentile: int = 99
    """Percentile to estimate for outlier detection."""
    n_latents: int = 400
    """Number of latents to save images for."""

    # Algorithm - shared
    sae_batch_size: int = 1024 * 8
    """Batch size for SAE inference."""

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

    @property
    def top_values_fpath(self) -> str:
        return os.path.join(self.root, "top_values.pt")

    @property
    def top_img_i_fpath(self) -> str:
        return os.path.join(self.root, "top_img_i.pt")

    @property
    def top_patch_i_fpath(self) -> str:
        return os.path.join(self.root, "top_patch_i.pt")

    @property
    def mean_values_fpath(self) -> str:
        return os.path.join(self.root, "mean_values.pt")

    @property
    def sparsity_fpath(self) -> str:
        return os.path.join(self.root, "sparsity.pt")

    @property
    def distributions_fpath(self) -> str:
        return os.path.join(self.root, "distributions.pt")

    @property
    def percentiles_fpath(self) -> str:
        return os.path.join(self.root, f"percentiles_p{self.percentile}.pt")

    @property
    def img_acts_fpath(self) -> str:
        return os.path.join(self.root, "img_acts.pt")


@jaxtyped(typechecker=beartype.beartype)
def gather_batched(
    value: Float[Tensor, "batch n dim"], i: Int[Tensor, "batch k"]
) -> Float[Tensor, "batch k dim"]:
    batch_size, n, dim = value.shape  # noqa: F841
    _, k = i.shape

    batch_i = torch.arange(batch_size, device=value.device)[:, None].expand(-1, k)
    return value[batch_i, i]


@jaxtyped(typechecker=beartype.beartype)
def get_sae_acts(
    vit_acts: Float[Tensor, "n d_vit"], sae: nn.SparseAutoencoder, cfg: Config
) -> Float[Tensor, "n d_sae"]:
    """
    Get SAE hidden layer activations for a batch of ViT activations.

    Args:
        vit_acts: Batch of ViT activations
        sae: Sparse autoencder.
        cfg: Experimental config.
    """
    sae_acts = []
    for start, end in helpers.batched_idx(len(vit_acts), cfg.sae_batch_size):
        _, f_x, *_ = sae(vit_acts[start:end].to(cfg.device))
        sae_acts.append(f_x)

    sae_acts = torch.cat(sae_acts, dim=0)
    sae_acts = sae_acts.to(cfg.device)
    return sae_acts


@beartype.beartype
@torch.inference_mode()
def dump_activations(cfg: Config):
    """
    Compute both top-k visual patches and image-level activations in a single pass.

    This function combines the functionality of get_topk_patch() from visuals.py and worker_fn() from dump.py, avoiding redundant computation of SAE activations.

    Args:
        cfg: Configuration object.

    Returns:
        UnifiedResults containing both top-k and image-level data.
    """
    assert cfg.data.patches == "image"

    sae = nn.load(cfg.ckpt).to(cfg.device)
    md = saev.data.Metadata.load(cfg.data.shard_root)

    # Initialize top-k tracking (from visuals.py)
    top_values_p = torch.full(
        (sae.cfg.d_sae, cfg.top_k, md.n_patches_per_img), -1.0, device=cfg.device
    )
    top_i_im = torch.zeros(
        (sae.cfg.d_sae, cfg.top_k), dtype=torch.int, device=cfg.device
    )

    sparsity_s = torch.zeros((sae.cfg.d_sae,), device=cfg.device)
    mean_values_s = torch.zeros((sae.cfg.d_sae,), device=cfg.device)

    # Initialize image-level activations (from dump.py)
    img_acts_ns = torch.zeros((md.n_imgs, sae.cfg.d_sae))

    batch_size = cfg.data.batch_size // md.n_patches_per_img * md.n_patches_per_img
    n_imgs_per_batch = batch_size // md.n_patches_per_img
    dataloader = saev.data.OrderedDataLoader(
        dataclasses.replace(cfg.data, batch_size=batch_size),
    )

    distributions_mn = torch.zeros(
        (cfg.n_distributions, dataloader.n_samples), device="cpu"
    )
    estimator = statistics.PercentileEstimator(
        cfg.percentile, dataloader.n_samples, shape=(sae.cfg.d_sae,)
    )

    logger.info("Loaded SAE and data.")

    for batch in helpers.progress(dataloader, desc="activations"):
        # Get SAE activations (shared computation)
        vit_acts_bd = batch["act"]
        sae_acts_bs = get_sae_acts(vit_acts_bd, sae, cfg)

        # Update percentile estimator
        for sae_act_s in sae_acts_bs:
            estimator.update(sae_act_s)

        # Reshape for per-image processing
        sae_acts_sb = einops.rearrange(sae_acts_bs, "batch d_sae -> d_sae batch")
        distributions_mn[:, batch["image_i"]] = sae_acts_sb[: cfg.n_distributions].to(
            "cpu"
        )

        # Update statistics
        mean_values_s += einops.reduce(sae_acts_sb, "d_sae batch -> d_sae", "sum")
        sparsity_s += einops.reduce((sae_acts_sb > 0), "d_sae batch -> d_sae", "sum")

        # Get unique image indices in this batch
        i_im = torch.sort(torch.unique(batch["image_i"])).values
        values_p = sae_acts_sb.view(sae.cfg.d_sae, len(i_im), md.n_patches_per_img)

        # Validation
        assert values_p.shape[1] == i_im.shape[0]
        if not len(i_im) == n_imgs_per_batch:
            logger.warning(
                "Got %d images; expected %d images per batch.",
                len(i_im),
                n_imgs_per_batch,
            )

        # Update top-k tracking (from visuals.py)
        _, k = torch.topk(sae_acts_sb, k=cfg.top_k, dim=1)
        k_im = k // md.n_patches_per_img

        values_p_topk = gather_batched(values_p, k_im)
        i_im_device = i_im.to(cfg.device)[k_im]

        all_values_p = torch.cat((top_values_p, values_p_topk), dim=1)
        _, k = torch.topk(all_values_p.max(dim=-1).values, k=cfg.top_k, dim=1)

        top_values_p = gather_batched(all_values_p, k)
        top_i_im = torch.gather(torch.cat((top_i_im, i_im_device), dim=1), 1, k)

        acts_sb = torch.topk(values_p, 3, dim=-1).values.mean(dim=-1)
        img_acts_ns[i_im] = acts_sb.cpu().T

    # Finalize statistics
    mean_values_s /= sparsity_s
    sparsity_s /= dataloader.n_samples

    # Save top-k visual results
    torch.save(top_values_p, cfg.top_values_fpath)
    torch.save(top_i_im, cfg.top_img_i_fpath)
    torch.save(mean_values_s, cfg.mean_values_fpath)
    torch.save(sparsity_s, cfg.sparsity_fpath)
    torch.save(distributions_mn, cfg.distributions_fpath)
    torch.save(estimator.estimate.cpu(), cfg.percentiles_fpath)
    torch.save(img_acts_ns, cfg.img_acts_fpath)


@beartype.beartype
@torch.inference_mode()
def worker_fn(cfg: Config):
    """Main entry point for unified activation computation."""
    os.makedirs(cfg.root, exist_ok=True)

    # Check if we need to compute activations
    fpaths = [
        cfg.top_values_fpath,
        cfg.top_img_i_fpath,
        cfg.mean_values_fpath,
        cfg.sparsity_fpath,
        cfg.distributions_fpath,
        cfg.percentiles_fpath,
        cfg.img_acts_fpath,
    ]
    missing = [fpath for fpath in fpaths if not os.path.exists(fpath)]
    need_compute = cfg.force_recompute or bool(missing)

    if need_compute:
        if cfg.force_recompute:
            logger.info("Force recompute flag set; computing activations.")
        else:
            logger.info("Missing files %s; computing activations.", ", ".join(missing))
        dump_activations(cfg)
    else:
        logger.info("Found existing activation files.")
