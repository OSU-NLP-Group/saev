""" """

import dataclasses
import logging
import math
import pathlib
import random
import typing as tp

import beartype
import numpy as np
import polars as pl
import scipy.sparse
import torch
import tyro
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor

import saev.data
import saev.disk
import saev.helpers
import saev.utils.statistics
import saev.viz

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("visuals")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for unified activation computation."""

    run: pathlib.Path = pathlib.Path("./runs/016lmihg")
    """Run directory."""
    shards: pathlib.Path = pathlib.Path("./shards/abcdef01")
    """Activations."""
    img_scale: float = 1.0
    """How much to scale images by (use higher numbers for high-res visuals)."""
    ignore_labels: list[int] = dataclasses.field(default_factory=list)
    """Which patch labels to ignore when calculating summarized image activations."""
    palette: pathlib.Path | None = None
    """Path to a palette .txt file."""

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Which accelerator to use."""
    sae_batch_size: int = 1024 * 8
    """Batch size for SAE inference."""

    log_freq_range: tuple[float, float] = (-6.0, 1.0)
    """Log10 frequency range for which to save images."""
    log_value_range: tuple[float, float] = (-3.0, 3.0)
    """Log10 frequency range for which to save images."""
    latents: list[int] = dataclasses.field(default_factory=list)
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
    tokens: Float[np.ndarray, " content_tokens_per_example"]
    # Metadata
    idx: int
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


@jaxtyped(typechecker=beartype.beartype)
def make_seg(
    seg: Image.Image,
    n_patches: int,
    patch_size: int,
    pixel_agg: saev.data.shards.PixelAgg,
    bg_label: int,
    palette: list[tuple[float, float, float]],
) -> Image.Image:
    """Create a colored visualization of segmentation patches."""

    w, h = seg.size
    patch_grid_h = h // patch_size
    patch_grid_w = w // patch_size
    patch_labels = (
        saev.data.shards.pixel_to_patch_labels(
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
def safe_load(path: pathlib.Path) -> Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


@beartype.beartype
@torch.inference_mode()
def cli(cfg: tp.Annotated[Config, tyro.conf.arg(name="")]):
    """Generate visual outputs for particular latents."""

    try:
        run = saev.disk.Run(cfg.run)
        # MASSIVE HACK
        # d_sae = run.config["k"]
        d_sae = run.config["sae"]["d_sae"]
        token_acts = scipy.sparse.load_npz(
            run.inference / cfg.shards.name / "token_acts.npz"
        )
        mean_values_s = safe_load(run.inference / cfg.shards.name / "mean_values.pt")
        sparsity_s = safe_load(run.inference / cfg.shards.name / "sparsity.pt")
        # distributions = safe_load(run.inference / "distributions.pt")
    except FileNotFoundError as err:
        logger.error("Required activation files not found: %s. Run inference.py", err)
        return

    # Create indexed activations dataset for efficient patch retrieval
    md = saev.data.Metadata.load(cfg.shards)
    vit = saev.data.models.load_model_cls(md.family)(md.ckpt)
    resize_tr = vit.make_resize(
        md.ckpt, md.content_tokens_per_example, scale=cfg.img_scale
    )
    img_cfg = md.make_data_cfg()
    img_ds = saev.data.datasets.get_dataset(
        img_cfg, data_transform=resize_tr, mask_transform=resize_tr
    )

    logger.info("Loaded data.")

    # obs_df = pl.DataFrame(
    #     [img_ds.get_metadata(i) for i in range(len(img_ds))], infer_schema_length=None
    # )
    # obs_fpath = os.path.join(cfg.root, "obs.parquet")
    # obs_df.write_parquet(obs_fpath)
    # logger.info("Saved obs.parquet with %d rows to '%s'.", obs_df.height, obs_fpath)

    topk_example_idx = (
        saev.helpers.csr_topk(token_acts, k=cfg.top_k, axis=0).indices
        // md.content_tokens_per_example
    )

    var_df = pl.DataFrame({
        "feature": range(d_sae),
        "log10_freq": torch.log10(sparsity_s).tolist(),
        "log10_value": torch.log10(mean_values_s).tolist(),
        "topk_example_idx": topk_example_idx.T.tolist(),
    })
    var_fpath = run.inference / cfg.shards.name / "var.parquet"
    var_df.write_parquet(var_fpath)
    logger.info("Saved var.parquet with %d rows to '%s'.", var_df.height, var_fpath)

    # fig_fpath = run.inference / f"{cfg.n_distributions}_activation_distributions.png"
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

    features = cfg.latents
    random_features = torch.arange(d_sae)[mask.cpu()].tolist()
    random.seed(cfg.seed)
    random.shuffle(random_features)
    features += random_features[: cfg.n_latents]

    topk_example_idx = np.stack(var_df.get_column("topk_example_idx").to_numpy())
    assert topk_example_idx.shape == (d_sae, cfg.top_k)
    topk_example_idx = topk_example_idx[features]
    topk_token_idx = (
        topk_example_idx[:, :, None] * md.content_tokens_per_example
        + np.arange(md.content_tokens_per_example)[None, None, :]
    )
    assert topk_token_idx.max() < token_acts.shape[0]
    logger.info("Calculated top-k for each latent.")

    if cfg.palette is None:
        import glasbey

        palette = [
            tuple(rgb) for rgb in glasbey.create_palette(palette_size=256, as_hex=False)
        ]
    else:
        palette = saev.viz.load_palette(cfg.palette)

    logger.info("Generated palette with %d colors.", len(palette))

    patch_size = int(vit.patch_size * cfg.img_scale)

    for f_i, f in enumerate(
        saev.helpers.progress(features, desc="saving imgs", every=1)
    ):
        feature_dir = run.inference / cfg.shards.name / "images" / str(f)
        feature_dir.mkdir(exist_ok=True, parents=True)

        f_token_idx = topk_token_idx[f_i]
        token_values_kp = token_acts[f_token_idx.ravel()][:, f].reshape(cfg.top_k, -1)
        token_values_kp = token_values_kp.toarray()
        examples = []

        seen_example_idx = set()
        for example_idx, token_values_p in zip(
            topk_example_idx[f_i].tolist(), token_values_kp
        ):
            if example_idx in seen_example_idx:
                continue
            sample = img_ds[example_idx]

            example = Example(
                img=sample["image"],
                seg=sample.get("patch_labels", None),
                tokens=token_values_p,
                idx=example_idx,
                target=sample["target"],
                label=sample["label"],
            )
            examples.append(example)
            seen_example_idx.add(example_idx)

        # How to scale values.
        upper = token_values_kp.max().item()

        for j, example in enumerate(examples):
            # 1. Save original image under {j}_img.png
            example.img.save(feature_dir / f"{j}_img.png")
            # 2. Save SAE highlighted image under {j}_sae_img.png
            img_with_highlights = saev.viz.add_highlights(
                example.img, example.tokens, patch_size, upper=upper
            )
            img_with_highlights.save(feature_dir / f"{j}_sae_img.png")

            if example.seg is not None:
                # 3. Save original segmentation under {j}_seg.png
                seg = make_seg(
                    example.seg,
                    md.content_tokens_per_example,
                    patch_size,
                    md.pixel_agg,
                    img_ds.cfg.bg_label,
                    palette,
                )
                seg.save(feature_dir / f"{j}_seg.png")

                # 4. Save SAE highlighted segmentation under {j}_sae_seg.png
                seg_with_highlights = saev.viz.add_highlights(
                    seg, example.tokens, patch_size, upper=upper
                )
                seg_with_highlights.save(feature_dir / f"{j}_sae_seg.png")
