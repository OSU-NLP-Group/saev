"""
There is some important notation used only in this file to dramatically shorten variable names.

Variables suffixed with `_im` refer to entire images, and variables suffixed with `_p` refer to patches.
"""

import dataclasses
import json
import logging
import math
import os
import random
import typing

import beartype
import einops
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
logger = logging.getLogger("visuals")


@beartype.beartype
def patch_label_ignore_bg_bincount(
    pixel_labels_nd: torch.Tensor,  # [N, P*P], int
    *,
    background_idx: int = 0,
    num_classes: int | None = None,
) -> torch.Tensor:  # [N]
    """
    Compute patch-level labels from pixel labels, with foreground priority.

    This uses a foreground-prior mode: compute the per-patch histogram,
    ignore the background count, and pick the most frequent non-background
    class if any exists; only assign background when the patch is 100% background.
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


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Configuration for generating visuals from trained SAEs."""

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

    # Algorithm
    top_k: int = 128
    """How many images per SAE feature to store."""
    epsilon: float = 1e-9
    """Value to add to avoid log(0)."""
    sort_by: typing.Literal["cls", "img", "patch"] = "patch"
    """How to find the top k images. 'cls' picks images where the SAE latents of the ViT's [CLS] token are maximized without any patch highligting. 'img' picks images that maximize the sum of an SAE latent over all patches in the image, highlighting the patches. 'patch' pickes images that maximize an SAE latent over all patches (not summed), highlighting the patches and only showing unique images."""
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
    """Maximum number of latents to save images for."""
    sae_batch_size: int = 1024 * 16
    """Batch size for SAE inference."""

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

    @property
    def root(self) -> str:
        return os.path.join(self.dump_to, f"sort_by_{self.sort_by}")

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


# def test_gather_batched_small():
#     values = torch.arange(0, 64, dtype=torch.float).view(4, 2, 8)
#     i = torch.tensor([[0], [0], [1], [1]])
#     actual = gather_batched(values, i)
#     expected = torch.tensor([
#         [[0, 1, 2, 3, 4, 5, 6, 7]],
#         [[16, 17, 18, 19, 20, 21, 22, 23]],
#         [[40, 41, 42, 43, 44, 45, 46, 47]],
#         [[56, 57, 58, 59, 60, 61, 62, 63]],
#     ]).float()
#     torch.testing.assert_close(actual, expected)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class GridElement:
    img: Image.Image
    label: str
    patches: Float[Tensor, " n_patches"]
    segmentation: Int[Tensor, " n_patches"] | None = None
    image_index: int | None = None


@jaxtyped(typechecker=beartype.beartype)
def get_new_topk(
    val1: Float[Tensor, "d_sae k"],
    i1: Int[Tensor, "d_sae k"],
    val2: Float[Tensor, "d_sae k"],
    i2: Int[Tensor, "d_sae k"],
    k: int,
) -> tuple[Float[Tensor, "d_sae k"], Int[Tensor, "d_sae k"]]:
    """
    Picks out the new top k values among val1 and val2. Also keeps track of i1 and i2, then indices of the values in the original dataset.

    Args:
        val1: top k original SAE values.
        i1: the patch indices of those original top k values.
        val2: top k incoming SAE values.
        i2: the patch indices of those incoming top k values.
        k: k.

    Returns:
        The new top k values and their patch indices.
    """
    all_val = torch.cat([val1, val2], dim=1)
    new_values, top_i = torch.topk(all_val, k=k, dim=1)

    all_i = torch.cat([i1, i2], dim=1)
    new_indices = torch.gather(all_i, 1, top_i)
    return new_values, new_indices


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


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class TopKPatch:
    """Summary of which images and patches most activate each SAE latent."""

    top_values: Float[Tensor, "d_sae k n_patches_per_img"]
    """For each latent and its top k images, the per-patch activation values over that image's patches. Used for heatmaps/highlighting of contributing regions."""
    top_i: Int[Tensor, "d_sae k"]
    """Global image indices for those top-`k` images per latent (aligned with the dataset's `image_i` indexing)."""
    mean_values: Float[Tensor, " d_sae"]
    """Mean positive activation per latent, computed as the sum of activations divided by the count of positives (`> 0`). With nonnegative activations (e.g., ReLU/TopK), this is the average active value."""
    sparsity: Float[Tensor, " d_sae"]
    """Fraction of samples where each latent is active (`> 0`), i.e., empirical activation rate."""
    distributions: Float[Tensor, "m n"]
    """For the first `m = cfg.n_distributions` latents, the raw activation values across all `n` samples. Useful for plotting activation histograms."""
    percentiles: Float[Tensor, " d_sae"]
    """Online estimate of the `cfg.percentile` activation threshold per latent over the dataset (used for outlier detection/thresholding)."""


@beartype.beartype
@torch.inference_mode()
def get_topk_patch(cfg: Config) -> TopKPatch:
    """
    Gets the top k images for each latent in the SAE.
    The top k images are for latent i are sorted by

        max over all patches: f_x(patch)[i]

    Thus, we could end up with duplicate images in the top k, if an image has more than one patch that maximally activates an SAE latent.

    Args:
        cfg: Config.

    Returns:
        A tuple of TopKPatch and m randomly sampled activation distributions.
    """
    assert cfg.sort_by == "patch"
    assert cfg.data.patches == "image"

    sae = nn.load(cfg.ckpt).to(cfg.device)
    metadata = saev.data.Metadata.load(cfg.data.shard_root)

    top_values_p = torch.full(
        (sae.cfg.d_sae, cfg.top_k, metadata.n_patches_per_img), -1.0, device=cfg.device
    )
    top_i_im = torch.zeros(
        (sae.cfg.d_sae, cfg.top_k), dtype=torch.int, device=cfg.device
    )

    sparsity_S = torch.zeros((sae.cfg.d_sae,), device=cfg.device)
    mean_values_S = torch.zeros((sae.cfg.d_sae,), device=cfg.device)

    batch_size = (
        cfg.data.batch_size // metadata.n_patches_per_img * metadata.n_patches_per_img
    )
    n_imgs_per_batch = batch_size // metadata.n_patches_per_img
    dataloader = saev.data.OrderedDataLoader(
        dataclasses.replace(cfg.data, batch_size=batch_size),
    )

    distributions_MN = torch.zeros(
        (cfg.n_distributions, dataloader.n_samples), device="cpu"
    )
    estimator = PercentileEstimator(
        cfg.percentile, dataloader.n_samples, shape=(sae.cfg.d_sae,)
    )

    logger.info("Loaded SAE and data.")

    for batch in helpers.progress(dataloader, desc="picking top-k"):
        vit_acts_BD = batch["act"]
        sae_acts_BS = get_sae_acts(vit_acts_BD, sae, cfg)

        for sae_act_S in sae_acts_BS:
            estimator.update(sae_act_S)

        sae_acts_SB = einops.rearrange(sae_acts_BS, "batch d_sae -> d_sae batch")
        distributions_MN[:, batch["image_i"]] = sae_acts_SB[: cfg.n_distributions].to(
            "cpu"
        )

        mean_values_S += einops.reduce(sae_acts_SB, "d_sae batch -> d_sae", "sum")
        sparsity_S += einops.reduce((sae_acts_SB > 0), "d_sae batch -> d_sae", "sum")

        i_im = torch.sort(torch.unique(batch["image_i"])).values
        values_p = sae_acts_SB.view(
            sae.cfg.d_sae, len(i_im), metadata.n_patches_per_img
        )

        # Checks that I did my reshaping correctly.
        assert values_p.shape[1] == i_im.shape[0]
        if not len(i_im) == n_imgs_per_batch:
            logger.warning(
                "Got %d images; expected %d images per batch.",
                len(i_im),
                n_imgs_per_batch,
            )

        _, k = torch.topk(sae_acts_SB, k=cfg.top_k, dim=1)
        k_im = k // metadata.n_patches_per_img

        values_p = gather_batched(values_p, k_im)
        i_im = i_im.to(cfg.device)[k_im]

        all_values_p = torch.cat((top_values_p, values_p), axis=1)
        _, k = torch.topk(all_values_p.max(axis=-1).values, k=cfg.top_k, axis=1)

        top_values_p = gather_batched(all_values_p, k)
        top_i_im = torch.gather(torch.cat((top_i_im, i_im), axis=1), 1, k)

    mean_values_S /= sparsity_S
    sparsity_S /= dataloader.n_samples

    return TopKPatch(
        top_values_p,
        top_i_im,
        mean_values_S,
        sparsity_S,
        distributions_MN,
        estimator.estimate.cpu(),
    )


@beartype.beartype
@torch.inference_mode()
def dump_activations(cfg: Config):
    """Dump ViT activation statistics for later use.

    The dataset described by ``cfg`` is processed to find the images or patches that maximally activate each SAE latent.  Various tensors summarising these activations are then written to ``cfg.root`` so they can be loaded by other tools.

    Args:
        cfg: options controlling which activations are processed and where the resulting files are saved.

    Returns:
        None. All data is saved to disk.
    """
    if cfg.sort_by == "img":
        raise helpers.RemovedFeatureError("Feature removed.")
    elif cfg.sort_by == "patch":
        topk = get_topk_patch(cfg)
    else:
        typing.assert_never(cfg.sort_by)

    os.makedirs(cfg.root, exist_ok=True)

    torch.save(topk.top_values, cfg.top_values_fpath)
    torch.save(topk.top_i, cfg.top_img_i_fpath)
    torch.save(topk.mean_values, cfg.mean_values_fpath)
    torch.save(topk.sparsity, cfg.sparsity_fpath)
    torch.save(topk.distributions, cfg.distributions_fpath)
    torch.save(topk.percentiles, cfg.percentiles_fpath)


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
def dump_imgs(cfg: Config):
    """
    .. todo:: document this function.

    Dump top-k images to a directory.

    Args:
        cfg: Configuration object.
    """

    try:
        top_values = safe_load(cfg.top_values_fpath)
        sparsity = safe_load(cfg.sparsity_fpath)
        mean_values = safe_load(cfg.mean_values_fpath)
        top_i = safe_load(cfg.top_img_i_fpath)
        distributions = safe_load(cfg.distributions_fpath)
        _ = safe_load(cfg.percentiles_fpath)
    except FileNotFoundError as err:
        logger.warning("Need to dump files: %s", err)
        dump_activations(cfg)
        return dump_imgs(cfg)

    d_sae, cached_topk, *rest = top_values.shape
    # Check that the data is at least shaped correctly.
    assert cfg.top_k == cached_topk
    if cfg.sort_by == "img":
        assert len(rest) == 0
    elif cfg.sort_by == "patch":
        assert len(rest) == 1
        n_patches = rest[0]
        assert n_patches > 0
    else:
        typing.assert_never(cfg.sort_by)

    logger.info("Loaded sorted data.")

    os.makedirs(cfg.root, exist_ok=True)
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

            if cfg.sort_by == "img":
                elem = GridElement(
                    example["image"],
                    example["label"],
                    torch.tensor([]),
                    segmentation,
                    i_im,
                )
            elif cfg.sort_by == "patch":
                elem = GridElement(
                    example["image"], example["label"], values_p, segmentation, i_im
                )
            else:
                typing.assert_never(cfg.sort_by)
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

        This method maintains the marker positions using the P2 algorithm rules.
        When a new value arrives, it's placed in the appropriate position relative to existing markers, and marker positions are adjusted to maintain their desired percentile positions.

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
