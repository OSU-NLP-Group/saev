"""
Only has to work for butterflies, beetles and fish. And let's start with just butterflies.
"""

import dataclasses
import logging
import math
import os
import random

import beartype
import polars as pl
import torch
import tyro
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from tdiscovery import datasets
from torch import Tensor

import saev.data
import saev.data.datasets
import saev.data.indexed
import saev.data.transforms
import saev.helpers
import saev.nn
import saev.utils.statistics
import saev.viz

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("visuals")


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Configuration for unified activation computation."""

    # Disk
    root: str = os.path.join(".", "acts", "butterflies", "abcdefg")
    """Path to the activations directory."""
    shard_root: str = os.path.join(".", "shards")
    """Directory with .bin shards and a metadata.json file."""
    ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    imgs: datasets.Config = dataclasses.field(default_factory=datasets.Butterflies)
    """Which image dataset to use."""
    img_scale: float = 1.0
    """How much to scale images by (use higher numbers for high-res visuals)."""

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Which accelerator to use."""
    sae_batch_size: int = 1024 * 8
    """Batch size for SAE inference."""

    log_freq_range: tuple[float, float] = (-6.0, 1.0)
    """Log10 frequency range for which to save images."""
    log_value_range: tuple[float, float] = (-3.0, 3.0)
    """Log10 frequency range for which to save images."""
    include_latents: list[int] = dataclasses.field(default_factory=list)
    """Latents to always include, no matter what."""
    n_distributions: int = 25
    """Number of features to save distributions for."""
    n_latents: int = 400
    """Number of latents to save images for."""

    seed: int = 42
    """Random seed."""

    # Properties for file paths

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
    def img_acts_fpath(self) -> str:
        return os.path.join(self.root, "img_acts.pt")


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Example:
    img: Image.Image
    patches: Float[Tensor, " n_patches"]
    # Metadata
    img_i: int
    target: int
    label: str
    extra: dict[str, object] = dataclasses.field(default_factory=dict)


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
            estimator = saev.utils.statistics.PercentileEstimator(percentile, len(vals))
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
def safe_load(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


@beartype.beartype
def unique_no_sort(lst: list[int]) -> list[int]:
    unique = []
    seen = set()
    for x in lst:
        if x in seen:
            continue

        unique.append(x)
        seen.add(x)

    return unique


@jaxtyped(typechecker=beartype.beartype)
def get_sae_acts(
    vit_acts: Float[Tensor, "n d_vit"], sae: saev.nn.SparseAutoencoder, cfg: Config
) -> Float[Tensor, "n d_sae"]:
    """
    Get SAE hidden layer activations for a batch of ViT activations.

    Args:
        vit_acts: Batch of ViT activations
        sae: Sparse autoencoder.
        cfg: Experimental config.
    """
    sae_acts = []
    for start, end in saev.helpers.batched_idx(len(vit_acts), cfg.sae_batch_size):
        _, f_x, *_ = sae(vit_acts[start:end].to(cfg.device))
        sae_acts.append(f_x)

    sae_acts = torch.cat(sae_acts, dim=0)
    sae_acts = sae_acts.to(cfg.device)
    return sae_acts


@beartype.beartype
@torch.inference_mode()
def main(cfg: Config):
    """
    Generate visual outputs from computed activations.

    This is equivalent to dump_imgs() from visuals.py but uses pre-computed results.
    """
    try:
        top_values = safe_load(cfg.top_values_fpath)
        sparsity = safe_load(cfg.sparsity_fpath)
        mean_values = safe_load(cfg.mean_values_fpath)
        top_img_i = safe_load(cfg.top_img_i_fpath)
        distributions = safe_load(cfg.distributions_fpath)
    except FileNotFoundError as err:
        logger.error("Required activation files not found: %s", err)
        logger.error("Please first compute activations.")
        return

    d_sae, cached_topk = top_values.shape

    logger.info("Loaded sorted data.")

    metadata = saev.data.Metadata.load(cfg.shard_root)
    vit_cls = saev.data.models.load_vit_cls(metadata.vit_family)
    img_transform = vit_cls.make_resize(
        metadata.vit_ckpt, metadata.n_patches_per_img, scale=cfg.img_scale
    )
    dataset = datasets.get_dataset(cfg.imgs, img_transform=img_transform)

    # Load SAE model for on-demand reconstruction
    sae = saev.nn.load(cfg.ckpt).to(cfg.device)

    # Create indexed activations dataset for efficient patch retrieval
    indexed_cfg = saev.data.indexed.Config(
        shard_root=cfg.shard_root,
        patches="image",
        layer=-2,  # Use second-to-last layer by default
    )
    indexed_dataset = saev.data.indexed.Dataset(indexed_cfg)

    # Cache for SAE activations to avoid recomputation
    sae_acts_cache = {}

    # Create img_obs.parquet with metadata for all images
    n_imgs = len(dataset)
    metadata_rows = []

    for i in range(n_imgs):
        metadata_dict = dataset.get_metadata(i)
        metadata_dict["index"] = i
        metadata_rows.append(metadata_dict)

    # Create DataFrame with metadata
    obs_df = pl.DataFrame(metadata_rows, infer_schema_length=None)

    # Save to parquet
    img_obs_fpath = os.path.join(cfg.root, "img_obs.parquet")
    obs_df.write_parquet(img_obs_fpath)
    logger.info("Saved img_obs.parquet with %d rows to '%s'.", n_imgs, img_obs_fpath)

    sae_var_df = pl.DataFrame({
        "feature": range(d_sae),
        "log10_freq": torch.log10(sparsity).tolist(),
        "log10_value": torch.log10(mean_values).tolist(),
        "top_i_im": [unique_no_sort(ims.tolist()) for ims in top_img_i],
    })
    sae_var_fpath = os.path.join(cfg.root, "sae_var.parquet")
    sae_var_df.write_parquet(sae_var_fpath)
    logger.info("Saved sae_var.parquet with %d rows to '%s'.", d_sae, sae_var_fpath)

    fig_fpath = os.path.join(
        cfg.root, f"{cfg.n_distributions}_activation_distributions.png"
    )
    plot_activation_distributions(cfg, distributions).savefig(fig_fpath, dpi=300)
    logger.info(
        "Saved %d activation distributions to '%s'.", cfg.n_distributions, fig_fpath
    )

    min_log_freq, max_log_freq = cfg.log_freq_range
    min_log_value, max_log_value = cfg.log_value_range

    mask = (
        (min_log_freq < torch.log10(sparsity))
        & (torch.log10(sparsity) < max_log_freq)
        & (min_log_value < torch.log10(mean_values))
        & (torch.log10(mean_values) < max_log_value)
    )

    features = cfg.include_latents
    random_features = torch.arange(d_sae)[mask.cpu()].tolist()
    random.seed(cfg.seed)
    random.shuffle(random_features)
    features += random_features[: cfg.n_latents]

    for i in saev.helpers.progress(features, desc="saving visuals"):
        feature_dir = os.path.join(cfg.root, "features", str(i))
        os.makedirs(feature_dir, exist_ok=True)

        # Image grid
        examples = []
        seen_img_i = set()

        # Get unique image indices for this feature
        unique_img_indices = unique_no_sort(top_img_i[i].tolist())

        for img_i in unique_img_indices:
            if img_i in seen_img_i:
                continue

            # Get cached SAE activations or compute them
            if img_i not in sae_acts_cache:
                # Fetch all patches for this image from indexed dataset
                # get_img_patches returns numpy array with shape [n_layers, n_patches, d_vit]
                img_patches_np = indexed_dataset.get_img_patches(img_i)
                # Select the appropriate layer (using layer_index from indexed dataset config)
                vit_acts_np = img_patches_np[indexed_dataset.layer_index]
                # Skip CLS token if present (first patch)
                if metadata.cls_token:
                    vit_acts_np = vit_acts_np[1:]
                # Convert to tensor and reshape to [n_patches, d_vit]
                vit_acts = torch.from_numpy(vit_acts_np.copy())
                # Run SAE forward pass to get activations
                sae_acts = get_sae_acts(vit_acts, sae, cfg)
                # Cache the result (keep on device for faster access)
                sae_acts_cache[img_i] = sae_acts

            # Get activations for this specific feature and move to CPU for visualization
            patches = sae_acts_cache[img_i][:, i].cpu()

            sample = dataset[img_i]

            example = Example(
                img=sample["image"],
                patches=patches,
                img_i=img_i,
                target=sample["target"],
                label=sample["label"],
            )
            examples.append(example)

            seen_img_i.add(img_i)

        # How to scale values.
        upper = None
        if top_values[i].numel() > 0:
            upper = top_values[i].max().item()

        for j, example in enumerate(examples):
            # Save SAE highlighted image
            # Get patch size from the VIT model
            vit_cls = saev.data.models.load_vit_cls(metadata.vit_family)
            patch_size = vit_cls.patch_size
            img = saev.viz.add_highlights(
                example.img, example.patches.numpy(), patch_size=patch_size, upper=upper
            )
            img.save(os.path.join(feature_dir, f"{j}_sae.png"))


if __name__ == "__main__":
    main(tyro.cli(Config))
