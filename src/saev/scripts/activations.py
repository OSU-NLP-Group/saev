"""
Unified script for computing both top-k visual patches and image-level activations in a single pass over the dataset. This combines functionality from visuals.py and dump.py to avoid redundant computation.
"""

import dataclasses
import json
import logging
import math
import os
import random

import beartype
import einops
import polars as pl
import torch
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

import saev.data
import saev.data.datasets
import saev.data.transforms
from saev import helpers, nn, viz

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
    images: saev.data.datasets.Config = dataclasses.field(
        default_factory=saev.data.datasets.Imagenet
    )
    """Which images to use."""
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
    seed: int = 42
    """Random seed."""
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

    @property
    def img_obs_fpath(self) -> str:
        return os.path.join(self.root, "img_obs.parquet")


@beartype.beartype
def safe_load(path: str) -> object:
    return torch.load(path, map_location="cpu", weights_only=True)


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
class PercentileEstimator:
    def __init__(
        self,
        percentile: float | int,
        total: int,
        lr: float = 1e-3,
        shape: tuple[int, ...] = (),
    ):
        self.percentile = percentile
        self.total = total
        self.lr = lr

        self._estimate = torch.zeros(shape)
        self._step = 0

    def update(self, x):
        """
        Update the estimator with a new value.

        This method maintains the marker positions using the P2 algorithm rules. When a new value arrives, it's placed in the appropriate position relative to existing markers, and marker positions are adjusted to maintain their desired percentile positions.

        Arguments:
            x: The new value to incorporate into the estimation
        """
        self._step += 1

        step_size = self.lr * (self.total - self._step) / self.total

        # Is a no-op if it's already on the same device.
        if isinstance(x, Tensor):
            self._estimate = self._estimate.to(x.device)

        self._estimate += step_size * (
            torch.sign(x - self._estimate) + 2 * self.percentile / 100 - 1.0
        )

    @property
    def estimate(self):
        return self._estimate


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Results:
    """Results from activation computation."""

    # Top-k visual results
    top_values: Float[Tensor, "d_sae k n_patches_per_img"]
    """For each latent and its top k images, the per-patch activation values."""
    top_i: Int[Tensor, "d_sae k"]
    """Global image indices for those top-`k` images per latent."""
    mean_values: Float[Tensor, " d_sae"]
    """Mean positive activation per latent."""
    sparsity: Float[Tensor, " d_sae"]
    """Fraction of samples where each latent is active."""
    distributions: Float[Tensor, "m n"]
    """Activation distributions for first m latents."""
    percentiles: Float[Tensor, " d_sae"]
    """Percentile estimates per latent."""

    # Image-level activations
    img_acts: Float[Tensor, "n_imgs d_sae"]
    """Image-level activations (n_imgs, d_sae)."""
    obs: list[dict]
    """Image metadata."""


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
    estimator = PercentileEstimator(
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

    # Get image metadata
    vit_cls = saev.data.models.load_vit_cls(md.vit_family)
    img_transform = vit_cls.make_resize(md.vit_ckpt, scale=1)
    img_ds = saev.data.datasets.get_dataset(cfg.images, img_transform=img_transform)
    if hasattr(img_ds, "samples") and hasattr(img_ds, "classes"):
        obs = [
            {"path": path, "target": target, "label": img_ds.classes[target]}
            for path, target in img_ds.samples
        ]
    else:
        logger.warning("Can't save .obs without '.samples'.")
        obs = []

    # Save top-k visual results
    torch.save(top_values_p, cfg.top_values_fpath)
    torch.save(top_i_im, cfg.top_img_i_fpath)
    torch.save(mean_values_s, cfg.mean_values_fpath)
    torch.save(sparsity_s, cfg.sparsity_fpath)
    torch.save(distributions_mn, cfg.distributions_fpath)
    torch.save(estimator.estimate.cpu(), cfg.percentiles_fpath)
    torch.save(img_acts_ns, cfg.img_acts_fpath)
    pl.DataFrame(obs).write_parquet(cfg.img_obs_fpath)


@beartype.beartype
def patch_label_ignore_bg_bincount(
    pixel_labels_nd: Int[Tensor, "n d"],
    *,
    background_idx: int = 0,
    num_classes: int | None = None,
) -> Int[Tensor, " n"]:
    """
    Compute patch-level labels from pixel labels, with foreground priority.

    This uses a foreground-prior mode: compute the per-patch histogram, ignore the background count, and pick the most frequent non-background class if any exists; only assign background when the patch is 100% background.
    """
    x = pixel_labels_nd.to(torch.long)
    N, _ = x.shape
    if num_classes is None:
        num_classes = int(x.max().item()) + 1

    # counts[i, c] = number of times class c appears in patch i
    offsets = torch.arange(N, device=x.device).unsqueeze(1) * num_classes
    flat = (x + offsets).reshape(-1)
    counts = torch.bincount(flat, minlength=N * num_classes).reshape(N, num_classes)

    nonbg = counts.clone()
    nonbg[:, background_idx] = 0
    has_nonbg = nonbg.sum(dim=1) > 0
    nonbg_arg = nonbg.argmax(dim=1)
    bg = torch.full_like(nonbg_arg, background_idx)
    return torch.where(has_nonbg, nonbg_arg, bg)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class GridElement:
    img: Image.Image
    label: str
    patches: Float[Tensor, " n_patches"]
    segmentation: Int[Tensor, " n_patches"] | None = None
    image_index: int | None = None


@jaxtyped(typechecker=beartype.beartype)
def plot_activation_distributions(cfg: Config, distributions: Float[Tensor, "m n"]):
    import matplotlib.pyplot as plt
    import numpy as np

    m, _ = distributions.shape

    n_rows = int(math.sqrt(m))
    n_cols = n_rows
    fig, axes = plt.subplots(
        figsize=(4 * n_cols, 4 * n_rows),
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        sharey=True,
    )

    _, bins = np.histogram(np.log10(distributions[distributions > 0].numpy()), bins=100)

    percentiles = [90, 95, 99, 100]
    colors = ("red", "darkorange", "gold", "lime")

    for dist, ax in zip(distributions, axes.reshape(-1)):
        vals = np.log10(dist[dist > 0].numpy())

        ax.hist(vals, bins=bins)

        if vals.size == 0:
            continue

        for i, (percentile, color) in enumerate(
            zip(np.percentile(vals, percentiles), colors)
        ):
            ax.axvline(percentile, color=color, label=f"{percentiles[i]}th %-ile")

        for i, (percentile, color) in enumerate(zip(percentiles, colors)):
            estimator = PercentileEstimator(percentile, len(vals))
            for v in vals:
                estimator.update(v)
            ax.axvline(
                estimator.estimate,
                color=color,
                linestyle="--",
                label=f"Est. {percentiles[i]}th %-ile",
            )

    ax.legend()

    fig.tight_layout()
    return fig


@beartype.beartype
@torch.inference_mode()
def generate_visuals(cfg: Config):
    """
    Generate visual outputs from computed activations.

    This is equivalent to dump_imgs() from visuals.py but uses pre-computed results.
    """
    try:
        top_values = safe_load(cfg.top_values_fpath)
        sparsity = safe_load(cfg.sparsity_fpath)
        mean_values = safe_load(cfg.mean_values_fpath)
        top_i = safe_load(cfg.top_img_i_fpath)
        distributions = safe_load(cfg.distributions_fpath)
        _ = safe_load(cfg.percentiles_fpath)
    except FileNotFoundError as err:
        logger.error("Required activation files not found: %s", err)
        logger.error("Please run the script first to compute activations.")
        return

    d_sae, cached_topk, *rest = top_values.shape
    assert cfg.top_k == cached_topk
    assert len(rest) == 1
    n_patches = rest[0]
    assert n_patches > 0

    logger.info("Loaded sorted data.")

    fig_fpath = os.path.join(
        cfg.root, f"{cfg.n_distributions}_activation_distributions.png"
    )
    plot_activation_distributions(cfg, distributions).savefig(fig_fpath, dpi=300)
    logger.info(
        "Saved %d activation distributions to '%s'.", cfg.n_distributions, fig_fpath
    )

    metadata = saev.data.Metadata.load(cfg.data.shard_root)
    vit_cls = saev.data.models.load_vit_cls(metadata.vit_family)
    img_transform = vit_cls.make_resize(metadata.vit_ckpt, scale=1)
    dataset = saev.data.datasets.get_dataset(cfg.images, img_transform=img_transform)

    # Check if this is a SegFolder dataset and prepare segmentation dataset if so
    seg_dataset = None
    is_segfolder = isinstance(cfg.images, saev.data.datasets.SegFolder)
    if is_segfolder:
        seg_transform = v2.Compose([
            saev.data.transforms.FlexResize(
                patch_size=16,
                n_patches=metadata.n_patches_per_img,
                resample=Image.NEAREST,
            ),
            v2.ToImage(),
        ])
        sample_transform = v2.Compose([
            saev.data.transforms.Patchify(
                patch_size=16, n_patches=metadata.n_patches_per_img, key="segmentation"
            ),
        ])
        seg_dataset = saev.data.datasets.SegFolderDataset(
            cfg.images,
            img_transform=img_transform,
            seg_transform=seg_transform,
            sample_transform=sample_transform,
        )

    min_log_freq, max_log_freq = cfg.log_freq_range
    min_log_value, max_log_value = cfg.log_value_range

    mask = (
        (min_log_freq < torch.log10(sparsity))
        & (torch.log10(sparsity) < max_log_freq)
        & (min_log_value < torch.log10(mean_values))
        & (torch.log10(mean_values) < max_log_value)
    )

    neurons = cfg.include_latents
    random_neurons = torch.arange(d_sae)[mask.cpu()].tolist()
    random.seed(cfg.seed)
    random.shuffle(random_neurons)
    neurons += random_neurons[: cfg.n_latents]

    for i in helpers.progress(neurons, desc="saving visuals"):
        neuron_dir = os.path.join(cfg.root, "neurons", str(i))
        os.makedirs(neuron_dir, exist_ok=True)

        # Image grid
        elems = []
        seen_i_im = set()
        for i_im, values_p in zip(top_i[i].tolist(), top_values[i]):
            if i_im in seen_i_im:
                continue

            example = dataset[i_im]

            # Get segmentation if available
            segmentation = None
            if seg_dataset is not None:
                seg_example = seg_dataset[i_im]
                if "segmentation" in seg_example:
                    # Convert pixel-level to patch-level labels
                    segmentation = patch_label_ignore_bg_bincount(
                        seg_example["segmentation"],
                        background_idx=0,
                        num_classes=10,  # FishVista has 10 classes
                    )

            elem = GridElement(
                example["image"], example["label"], values_p, segmentation, i_im
            )
            elems.append(elem)

            seen_i_im.add(i_im)

        # How to scale values.
        upper = None
        if top_values[i].numel() > 0:
            upper = top_values[i].max().item()

        for j, elem in enumerate(elems):
            # Save SAE highlighted image
            img = viz.add_highlights(
                elem.img, elem.patches.numpy(), patch_size=16, upper=upper
            )
            img.save(os.path.join(neuron_dir, f"{j}_sae.png"))

            # Save segmentation visualization if available
            if elem.segmentation is not None:
                seg_img = viz.colorize_segmentation_patches(
                    elem.segmentation,
                    patch_size=16,
                    img_width=elem.img.width,
                    img_height=elem.img.height,
                    n_classes=10,  # FishVista has 10 classes
                    background_idx=0,
                )
                seg_img.save(os.path.join(neuron_dir, f"{j}_seg.png"))

            # Save per-image metadata as JSON
            img_metadata = {
                "label": elem.label,
                "dataset_type": "SegFolder" if is_segfolder else "ImageFolder",
                "image_index": elem.image_index,
            }
            with open(os.path.join(neuron_dir, f"{j}.json"), "w") as fd:
                json.dump(img_metadata, fd)

        # Neuron-level metadata
        neuron_metadata = {
            "neuron": i,
            "log10_freq": torch.log10(sparsity[i]).item(),
            "log10_value": torch.log10(mean_values[i]).item(),
        }
        with open(os.path.join(neuron_dir, "metadata.json"), "w") as fd:
            json.dump(neuron_metadata, fd)


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
        cfg.img_obs_fpath,
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
        logger.info("Found existing activation files, skipping computation.")

    generate_visuals(cfg)

    logger.info("Complete.")
