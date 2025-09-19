"""
Only has to work for butterflies, beetles and fish. And let's start with just butterflies.
"""

import dataclasses
import logging
import math
import os
import random
import typing as tp

import beartype
import polars as pl
import torch
import tyro
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from torch import Tensor

import saev.data
import saev.data.datasets
import saev.data.transforms
import saev.helpers
import saev.nn
import saev.utils.statistics
import saev.viz

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("visuals")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for unified activation computation."""

    # Disk
    root: str = os.path.join(".", "acts", "butterflies", "abcdefg")
    """Path to the activations directory."""
    acts: saev.data.IndexedConfig = saev.data.IndexedConfig()
    """Activations."""
    ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    imgs: saev.data.datasets.SegFolder = saev.data.datasets.SegFolder()
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


@beartype.beartype
def first_k_unique(values: Int[Tensor, " n"], *, k: int) -> list[int]:
    unique = []
    seen = set()
    for x in values:
        x = x.item()
        if x in seen:
            continue

        unique.append(x)
        seen.add(x)

        if len(unique) >= k:
            break

    return unique


@beartype.beartype
@torch.inference_mode()
def main(cfg: tp.Annotated[Config, tyro.conf.arg(name="")]):
    """
    Generate visual outputs for particular latents.
    """

    try:
        img_acts_ns = safe_load(os.path.join(cfg.root, "img_acts.pt"))
        sparsity_s = safe_load(os.path.join(cfg.root, "sparsity.pt"))
        mean_values_s = safe_load(os.path.join(cfg.root, "mean_values.pt"))
        # distributions = safe_load(os.path.join(cfg.root, "distributions.pt"))
    except FileNotFoundError as err:
        logger.error("Required activation files not found: %s", err)
        logger.error("Please first compute activations.")
        return

    logger.info("Loaded sorted data.")

    # Create indexed activations dataset for efficient patch retrieval
    act_ds = saev.data.Dataset(cfg.acts)
    vit_cls = saev.data.models.load_vit_cls(act_ds.metadata.vit_family)
    resize_tr = vit_cls.make_resize(
        act_ds.metadata.vit_ckpt, act_ds.metadata.n_patches_per_img, scale=cfg.img_scale
    )
    img_ds = saev.data.datasets.get_dataset(
        cfg.imgs, img_transform=resize_tr, seg_transform=resize_tr
    )

    # Load SAE model for on-demand reconstruction
    sae = saev.nn.load(cfg.ckpt).to(cfg.device)

    var_df = pl.DataFrame({
        "feature": range(sae.cfg.d_sae),
        "log10_freq": torch.log10(sparsity_s).tolist(),
        "log10_value": torch.log10(mean_values_s).tolist(),
    })
    var_fpath = os.path.join(cfg.root, "var.parquet")
    var_df.write_parquet(var_fpath)
    logger.info("Saved var.parquet with %d rows to '%s'.", var_df.height, var_fpath)

    # fig_fpath = os.path.join(
    #     cfg.root, f"{cfg.n_distributions}_activation_distributions.png"
    # )
    # plot_activation_distributions(cfg, distributions).savefig(fig_fpath, dpi=300)
    # logger.info(
    #     "Saved %d activation distributions to '%s'.", cfg.n_distributions, fig_fpath
    # )

    min_log_freq, max_log_freq = cfg.log_freq_range
    min_log_value, max_log_value = cfg.log_value_range

    mask = (
        (min_log_freq < torch.log10(sparsity_s))
        & (torch.log10(sparsity_s) < max_log_freq)
        & (min_log_value < torch.log10(mean_values_s))
        & (torch.log10(mean_values_s) < max_log_value)
    )

    features = cfg.include_latents
    random_features = torch.arange(sae.cfg.d_sae)[mask.cpu()].tolist()
    random.seed(cfg.seed)
    random.shuffle(random_features)
    features += random_features[: cfg.n_latents]

    # Algorithm
    # =========
    # 1. We want to run inference on all (1920) patches on the top k (20) images for each of the selected SAE latents (400).
    # 2. Use a torch.utils.data.Subset on top of act_ds with the selected patches in combination with a torch.utils.data.DataLoader to get the relevant patches using multiple dataloading processes.
    # 3. As you iterate over those patches in batches, use the SAE to calculate f_x[:, latents] for the latents we care about.
    # Keep these values in RAM (400 x 1920 x 20 x 4 bytes = 61.4 MB), then iterate through the images and save the actual highlighted images themselves.

    # TODO: keep a list of all the patches we want.
    patch_i = []

    # This needs to be filled in. Every value should be non-negative at the end.
    patch_values_skp = torch.full((len(features), cfg.top_k, sae.cfg.d_sae), value=-1.0)

    # TODO: loop over patches.
    dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(act_ds, patch_i),
        batch_size=cfg.sae_batch_size,
        drop_last=False,
        shuffle=False,
    )

    # Pseudocode:
    # for batch in dl:
    #     sae(batch)
    #

    assert (patch_values_skp >= 0).all()

    for i in saev.helpers.progress(features, desc="saving imgs"):
        feature_dir = os.path.join(cfg.root, "features", str(i))
        os.makedirs(feature_dir, exist_ok=True)

        # Image grid
        examples = []

        k = 20
        img_i = first_k_unique(torch.argsort(img_acts_ns[:, i], descending=True), k=k)

        patch_i = (
            torch.arange(act_ds.metadata.n_patches_per_img).expand(k, 1)
            + torch.tensor(img_i).view(k, 1)
        ).ravel()

        breakpoint()

        for img_i in top_img_i[i]:
            if img_i in seen_img_i:
                continue

            # Fetch all patches for this image from indexed dataset
            # get_img_patches returns numpy array with shape [n_layers, n_patches, d_vit]
            img_patches_np = img_ds.get_img_patches(img_i)
            # Select the appropriate layer (using layer_index from indexed dataset config)
            vit_acts_np = img_patches_np[img_ds.layer_index]
            # Skip CLS token if present (first patch)
            if act_ds.metadata.cls_token:
                vit_acts_np = vit_acts_np[1:]
            # Convert to tensor and reshape to [n_patches, d_vit]
            vit_acts = torch.from_numpy(vit_acts_np.copy())
            # Run SAE forward pass to get activations
            sae_acts = get_sae_acts(vit_acts, sae, cfg)

            # Get activations for this specific feature and move to CPU for visualization
            patches = sae_acts[:, i].cpu()

            sample = img_ds[img_i]

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
            img = saev.viz.add_highlights(
                example.img,
                example.patches.numpy(),
                patch_size=vit_cls.patch_size,
                upper=upper,
            )
            img.save(os.path.join(feature_dir, f"{j}_sae.png"))
