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
import glasbey
import numpy as np
import polars as pl
import tdiscovery.datasets
import torch
import tyro
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor

import saev.data
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

    root: str = os.path.join(".", "acts", "butterflies", "abcdefg")
    """Path to the saved SAE image activations."""
    acts: saev.data.IndexedConfig = saev.data.IndexedConfig()
    """Activations."""
    ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    imgs: tdiscovery.datasets.Config = tdiscovery.datasets.Config()
    """Which image dataset to use."""
    img_scale: float = 1.0
    """How much to scale images by (use higher numbers for high-res visuals)."""
    ignore_labels: list[int] = dataclasses.field(default_factory=list)
    """Which patch labels to ignore when calculating summarized image activations."""

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
    top_k: int = 20
    """Number of top images to visualize per feature."""

    seed: int = 42
    """Random seed."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Example:
    img: Image.Image
    seg: Image.Image | None  # Segmentation mask if available
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


@jaxtyped(typechecker=beartype.beartype)
def make_seg(
    seg: Image.Image,
    n_patches: int,
    patch_size: int,
    pixel_agg: tp.Literal["majority", "prefer-fg"],
    bg_label: int,
    palette: list[tuple[float, float, float]],
) -> Image.Image:
    """Create a colored visualization of segmentation patches."""

    w, h = seg.size
    patch_grid_h = h // patch_size
    patch_grid_w = w // patch_size
    patch_labels = (
        saev.data.writers.pixel_to_patch_labels(
            seg, n_patches, patch_size, pixel_agg, bg_label
        )
        .numpy()
        .reshape(patch_grid_h, patch_grid_w)
    )
    img = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(patch_grid_h):
        for x in range(patch_grid_w):
            class_id = patch_labels[y][x].item()
            img[
                y * patch_size : (y + 1) * patch_size,
                x * patch_size : (x + 1) * patch_size,
            ] = [int(c * 255) for c in palette[class_id]]

    return Image.fromarray(img)


@beartype.beartype
@torch.inference_mode()
def main(cfg: tp.Annotated[Config, tyro.conf.arg(name="")]):
    """Generate visual outputs for particular latents."""

    try:
        img_acts_ns = safe_load(os.path.join(cfg.root, "img_acts.pt"))
        sparsity_s = safe_load(os.path.join(cfg.root, "sparsity.pt"))
        mean_values_s = safe_load(os.path.join(cfg.root, "mean_values.pt"))
        distributions = safe_load(os.path.join(cfg.root, "distributions.pt"))
    except FileNotFoundError as err:
        logger.error("Required activation files not found: %s", err)
        logger.error("Please first compute activations.")
        return

    # Create indexed activations dataset for efficient patch retrieval
    act_ds = saev.data.Dataset(cfg.acts)
    md = act_ds.metadata
    n_patches = md.n_patches_per_img
    vit_cls = saev.data.models.load_vit_cls(md.vit_family)
    resize_tr = vit_cls.make_resize(md.vit_ckpt, n_patches, scale=cfg.img_scale)
    img_ds = tdiscovery.datasets.get_dataset(
        cfg.imgs, img_tr=resize_tr, seg_tr=resize_tr
    )

    logger.info("Loaded data.")

    obs_df = pl.DataFrame(
        [img_ds.get_metadata(i) for i in range(len(img_ds))], infer_schema_length=None
    )
    obs_fpath = os.path.join(cfg.root, "obs.parquet")
    obs_df.write_parquet(obs_fpath)
    logger.info("Saved obs.parquet with %d rows to '%s'.", obs_df.height, obs_fpath)

    # Load SAE model for on-demand reconstruction
    sae = saev.nn.load(cfg.ckpt).to(cfg.device)

    var_df = pl.DataFrame({
        "feature": range(sae.cfg.d_sae),
        "log10_freq": torch.log10(sparsity_s).tolist(),
        "log10_value": torch.log10(mean_values_s).tolist(),
        "top_img_i": torch.topk(img_acts_ns, k=cfg.top_k, dim=0).indices.T.tolist(),
    })
    var_fpath = os.path.join(cfg.root, "var.parquet")
    var_df.write_parquet(var_fpath)
    logger.info("Saved var.parquet with %d rows to '%s'.", var_df.height, var_fpath)

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

    INIT = -1

    # Algorithm
    # =========
    # 1. We want to run inference on all (1920) patches on the top k (20) images for each of the selected SAE latents (400).
    # 2. Use a torch.utils.data.Subset on top of act_ds with the selected patches in combination with a torch.utils.data.DataLoader to get the relevant patches using multiple dataloading processes.
    # 3. As you iterate over those patches in batches, use the SAE to calculate f_x[:, latents] for the latents we care about.
    # Keep these values in RAM (400 x 1920 x 20 x 4 bytes = 61.4 MB), then iterate through the images and save the actual highlighted images themselves.

    # Build list of all patch indices we need and track which features need which images

    # Collect all top images for all features at once
    # mapping is a lookup from (image index, feature index) -> rank.
    rank_lookup_nf = torch.full((len(img_ds), len(features)), INIT)
    all_img_i, all_patch_i = [], []

    for f_i, f in enumerate(saev.helpers.progress(features, desc="pick patches")):
        # Get top k unique images for this feature
        feature_acts_n = img_acts_ns[:, f]

        # Assertions
        assert feature_acts_n.ndim == 1
        assert (feature_acts_n >= 0.0).all(), "SAE activations must be non-negative"

        img_i = torch.argsort(feature_acts_n, descending=True)[: cfg.top_k]

        assert img_i.numel() == cfg.top_k
        assert (img_i >= 0).all(), "Image indices must be non-negative"
        msg = "Image indices must be in [0, {len(img_ds)})"
        assert (img_i < len(img_ds)).all(), msg

        rank_lookup_nf[img_i, f_i] = torch.arange(len(img_i))

        # Vectorized computation of all patch indices for this feature's top images
        all_patch_i.append(
            (
                img_i.view(-1, 1) * n_patches + torch.arange(n_patches).view(1, -1)
            ).ravel()
        )
        all_img_i.append(img_i)

    assert (rank_lookup_nf < cfg.top_k).all(), f"Ranks must be in [{INIT}, {cfg.top_k})"

    # Concatenate all patch indices
    all_patch_i = torch.unique(torch.cat(all_patch_i)).sort().values
    all_img_i = torch.stack(all_img_i)

    assert all_img_i.shape == (len(features), cfg.top_k)

    # Store patch activations for each feature's top k images
    patch_values_fkp = torch.full(
        (len(features), cfg.top_k, n_patches), INIT, dtype=torch.float32
    )
    logger.info(
        "%.1f%% unique patches.", all_patch_i.numel() / patch_values_fkp.numel() * 100
    )

    # Create DataLoader with the patches we need
    dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(act_ds, all_patch_i),
        batch_size=cfg.sae_batch_size,
        drop_last=False,
        shuffle=False,
    )

    ignore = torch.tensor(cfg.ignore_labels)

    # Process patches and compute SAE activations
    for batch in saev.helpers.progress(dl, desc="SAE inference"):
        # Patch indices must be in [0, n_patches).
        assert (batch["patch_i"] >= 0).all(), "Patch indices must be non-negative"
        msg = f"Patch indices must be in [0, {n_patches})"
        assert (batch["patch_i"] < n_patches).all(), msg

        vit_acts_bd = batch["act"].to(cfg.device)
        # Run SAE encoding to get latent activations
        f_x = sae.encode(vit_acts_bd)

        bsz, d_sae = f_x.shape

        # All values of f_x should be non-negative.
        assert (f_x >= 0).all(), "SAE activations must be non-negative"

        b_i, f_i = torch.where(rank_lookup_nf[batch["image_i"]] >= 0)

        assert b_i.ndim == 1, "img_i must be a vector"
        assert len(b_i) >= bsz
        assert (b_i >= 0).all(), "Batch indices must be non-negative."
        assert (b_i <= bsz).all(), "Batch indices must index in the batch."

        # Feature indices must also be non-negative
        assert f_i.ndim == 1, "f_i must be a vector"
        assert len(f_i) >= bsz
        assert (f_i >= 0).all(), "Feature indices must be non-negative."
        # Feature indices must be less than the number of features
        msg = "Feature indices must be in [0, {len(features)})."
        assert (f_i < len(features)).all(), msg

        rank = rank_lookup_nf[batch["image_i"]][b_i, f_i]
        assert (rank >= 0).all(), "Ranks must be non-negative."
        assert (rank < cfg.top_k).all(), f"Ranks must be in [0, {cfg.top_k})."

        # Every patch value should be set exactly once.
        set_at = (f_i, rank, batch["patch_i"][b_i])
        msg = "No patches should be updated twice."
        assert (patch_values_fkp[set_at] == INIT).all(), msg

        patch_acts = f_x[b_i, f_i]
        patch_labels = batch["patch_label"][b_i]
        patch_acts[torch.isin(patch_labels, ignore)] = 0.0

        # After indexing, latents should still be non-negative.
        msg = "SAE latent activations should be non-negative"
        assert (patch_acts >= 0).all(), msg

        patch_values_fkp[set_at] = patch_acts.cpu()

    # Since INIT is negative, >= 0 is the same as checking that all values have been set appropriately.
    assert (patch_values_fkp >= 0).all()

    palette = [
        tuple(rgb) for rgb in glasbey.create_palette(palette_size=256, as_hex=False)
    ]
    for f_i, f in enumerate(
        saev.helpers.progress(features, desc="saving imgs", every=1)
    ):
        feature_dir = os.path.join(cfg.root, "features", str(f))
        os.makedirs(feature_dir, exist_ok=True)

        examples = []

        patch_values_kp = patch_values_fkp[f_i]

        for img_i, patch_values_p in zip(all_img_i[f_i].tolist(), patch_values_kp):
            sample = img_ds[img_i]

            example = Example(
                img=sample["image"],
                seg=sample.get("patch_labels", None),
                patches=patch_values_p,
                img_i=img_i,
                target=sample["target"],
                label=sample["label"],
            )
            examples.append(example)

        # How to scale values.
        upper = patch_values_kp.max().item()

        for j, example in enumerate(examples):
            # 1. Save original image under {j}_img.png
            example.img.save(os.path.join(feature_dir, f"{j}_img.png"))
            # 2. Save SAE highlighted image under {j}_sae_img.png
            img_with_highlights = saev.viz.add_highlights(
                example.img, example.patches.numpy(), vit_cls.patch_size, upper=upper
            )
            img_with_highlights.save(os.path.join(feature_dir, f"{j}_sae_img.png"))

            if example.seg is not None:
                # 3. Save original segmentation under {j}_seg.png
                seg = make_seg(
                    example.seg,
                    n_patches,
                    vit_cls.patch_size,
                    md.pixel_agg,
                    md.data.get("bg-label", 0),
                    palette,
                )
                seg.save(os.path.join(feature_dir, f"{j}_seg.png"))

                # 4. Save SAE highlighted segmentation under {j}_sae_seg.png
                seg_with_highlights = saev.viz.add_highlights(
                    seg, example.patches.numpy(), vit_cls.patch_size, upper=upper
                )
                seg_with_highlights.save(os.path.join(feature_dir, f"{j}_sae_seg.png"))
