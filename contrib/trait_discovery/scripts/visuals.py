"""
Only has to work for butterflies, beetles and fish. And let's start with just butterflies.
"""

import dataclasses
import json
import logging
import math
import os
import random

import beartype
import torch
import tyro
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from tdiscovery import datasets
from torch import Tensor

import saev.data
import saev.data.datasets
import saev.data.transforms
import saev.helpers
import saev.utils.statistics
import saev.viz

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("unified_acts")


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Configuration for unified activation computation."""

    # Disk
    root: str = os.path.join(".", "acts", "butterflies", "abcdefg")
    """Path to the sae.pt file."""
    shard_root: str = os.path.join(".", "shards")
    """Directory with .bin shards and a metadata.json file."""
    imgs: datasets.Config = dataclasses.field(default_factory=datasets.Butterflies)
    """Which image dataset to use."""
    dump_to: str = os.path.join(".", "data")
    """Where to save images."""

    log_freq_range: tuple[float, float] = (-6.0, -2.0)
    """Log10 frequency range for which to save images."""
    log_value_range: tuple[float, float] = (-1.0, 1.0)
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
    index: int
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
def safe_load(path: str) -> object:
    return torch.load(path, map_location="cpu", weights_only=True)


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
        top_i = safe_load(cfg.top_img_i_fpath)
        distributions = safe_load(cfg.distributions_fpath)
    except FileNotFoundError as err:
        logger.error("Required activation files not found: %s", err)
        logger.error("Please first compute activations.")
        return

    d_sae, cached_topk, *rest = top_values.shape
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

    metadata = saev.data.Metadata.load(cfg.shard_root)
    vit_cls = saev.data.models.load_vit_cls(metadata.vit_family)
    img_transform = vit_cls.make_resize(metadata.vit_ckpt, scale=1)
    dataset = datasets.get_dataset(cfg.imgs, img_transform=img_transform)

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

    for i in saev.helpers.progress(neurons, desc="saving visuals"):
        neuron_dir = os.path.join(cfg.root, "neurons", str(i))
        os.makedirs(neuron_dir, exist_ok=True)

        # Image grid
        elems = []
        seen_i_im = set()
        for i_im, values_p in zip(top_i[i].tolist(), top_values[i]):
            if i_im in seen_i_im:
                continue

            example = dataset[i_im]

            elem = GridElement(example["image"], example["label"], values_p, i_im)
            elems.append(elem)

            seen_i_im.add(i_im)

        # How to scale values.
        upper = None
        if top_values[i].numel() > 0:
            upper = top_values[i].max().item()

        for j, elem in enumerate(elems):
            # Save SAE highlighted image
            img = saev.viz.add_highlights(
                elem.img, elem.patches.numpy(), patch_size=16, upper=upper
            )
            img.save(os.path.join(neuron_dir, f"{j}_sae.png"))

            # Save per-image metadata as JSON
            img_metadata = {
                "label": elem.label,
                "dataset_type": dataset.__class__.__name__,
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


if __name__ == "__main__":
    main(tyro.cli(Config))
