# visuals.py
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

import saev.data
import saev.data.images
from saev import helpers, imaging, nn

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("visuals")


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
    images: saev.data.images.Config = dataclasses.field(
        default_factory=saev.data.images.Imagenet
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
@dataclasses.dataclass
class GridElement:
    img: Image.Image
    label: str
    patches: Float[Tensor, " n_patches"]


@jaxtyped(typechecker=beartype.beartype)
def make_img(elem: GridElement, *, upper: float | None = None) -> Image.Image:
    resize_size_px = (512, 512)
    resize_w_px, resize_h_px = resize_size_px
    crop_size_px = (448, 448)
    crop_w_px, crop_h_px = crop_size_px
    crop_coords_px = (
        (resize_w_px - crop_w_px) // 2,
        (resize_h_px - crop_h_px) // 2,
        (resize_w_px + crop_w_px) // 2,
        (resize_h_px + crop_h_px) // 2,
    )

    img = elem.img.resize(resize_size_px).crop(crop_coords_px)
    img = imaging.add_highlights(img, elem.patches.numpy(), upper=upper)
    return img


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
@dataclasses.dataclass(frozen=True)
class TopKImg:
    ".. todo:: Document this class."

    top_values: Float[Tensor, "k d_sae"]
    top_i: Int[Tensor, "k d_sae"]
    mean_values: Float[Tensor, " d_sae"]
    sparsity: Float[Tensor, " d_sae"]
    distributions: Float[Tensor, "m n"]
    percentiles: Float[Tensor, " d_sae"]


@jaxtyped(typechecker=beartype.beartype)
def get_topk_img(cfg: Config) -> TopKImg:
    """
    Gets the top k images for each latent in the SAE.
    The top k images are for latent i are sorted by

        max over all images: f_x(cls)[i]

    Thus, we will never have duplicate images for a given latent.
    But we also will not have patch-level activations (a nice heatmap).

    Args:
        cfg: Config.

    Returns:
        A tuple of TopKImg and the first m features' activation distributions.
    """
    # assert cfg.sort_by == "img"
    # assert cfg.data.patches == "cls"

    # sae = nn.load(cfg.ckpt).to(cfg.device)
    # dataloader = saev.data.DataLoader(cfg.data)

    # top_values_im_SK = torch.full((sae.cfg.d_sae, cfg.top_k), -1.0, device=cfg.device)
    # top_i_im_SK = torch.zeros(
    #     (sae.cfg.d_sae, cfg.top_k), dtype=torch.int, device=cfg.device
    # )
    # sparsity_S = torch.zeros((sae.cfg.d_sae,), device=cfg.device)
    # mean_values_S = torch.zeros((sae.cfg.d_sae,), device=cfg.device)

    # distributions_MN = torch.zeros(
    #     (cfg.n_distributions, dataloader.n_samples), device="cpu"
    # )
    # estimator = PercentileEstimator(
    #     cfg.percentile, dataloader.n_samples, shape=(sae.cfg.d_sae,)
    # )

    # logger.info("Loaded SAE and data.")

    # for batch in helpers.progress(dataloader, desc="picking top-k"):
    #     vit_acts_BD = batch["act"]
    #     sae_acts_BS = get_sae_acts(vit_acts_BD, sae, cfg)

    #     for sae_act_S in sae_acts_BS:
    #         estimator.update(sae_act_S)

    #     sae_acts_SB = einops.rearrange(sae_acts_BS, "batch d_sae -> d_sae batch")
    #     distributions_MN[:, batch["image_i"]] = sae_acts_SB[: cfg.n_distributions].to(
    #         "cpu"
    #     )

    #     mean_values_S += einops.reduce(sae_acts_SB, "d_sae batch -> d_sae", "sum")
    #     sparsity_S += einops.reduce((sae_acts_SB > 0), "d_sae batch -> d_sae", "sum")

    #     sae_acts_SK, k = torch.topk(sae_acts_SB, k=cfg.top_k, dim=1)
    #     i_im_SK = batch["image_i"].to(cfg.device)[k]

    #     all_values_im_2SK = torch.cat((top_values_im_SK, sae_acts_SK), axis=1)

    #     top_values_im_SK, k = torch.topk(all_values_im_2SK, k=cfg.top_k, axis=1)
    #     top_i_im_SK = torch.gather(torch.cat((top_i_im_SK, i_im_SK), axis=1), 1, k)

    # mean_values_S /= sparsity_S
    # sparsity_S /= dataloader.n_samples

    # return TopKImg(
    #     top_values_im_SK,
    #     top_i_im_SK,
    #     mean_values_S,
    #     sparsity_S,
    #     distributions_MN,
    #     estimator.estimate.cpu(),
    # )


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class TopKPatch:
    ".. todo:: Document this class."

    top_values_KPD: Float[Tensor, "k n_patches_per_img d_sae"]
    top_i_KD: Int[Tensor, "k d_sae"]
    mean_values_D: Float[Tensor, " d_sae"]
    sparsity_D: Float[Tensor, " d_sae"]
    distributions_MN: Float[Tensor, "m n"]
    percentiles_D: Float[Tensor, " d_sae"]


@jaxtyped(typechecker=beartype.beartype)
def get_topk_patch(cfg: Config) -> TopKPatch:
    """
    Gets the top k images for each latent in the SAE.
    The top k images are for latent i are sorted by

        max over all patches: f_x(patch)[i]

    Thus, we could end up with duplicate images in the top k, if an image has more than one patch that maximally activates an SAE latent.

    Args:
        cfg: Config.

    Returns:
        A TopKPatch object.
    """
    assert cfg.sort_by == "patch"
    assert cfg.data.patches == "image"

    sae = nn.load(cfg.ckpt).to(cfg.device)
    dataloader = saev.data.DataLoader(cfg.data)

    # Patch values in [0, inf)
    top_values = torch.full((cfg.top_k, sae.cfg.d_sae), -1.0, device="cpu")
    top_i_p = torch.full(
        (cfg.top_k, sae.cfg.d_sae), -1.0, dtype=torch.int32, device="cpu"
    )
    top_i_im = torch.full(
        (cfg.top_k, sae.cfg.d_sae), -1.0, dtype=torch.int32, device="cpu"
    )

    sparsity_S = torch.zeros((sae.cfg.d_sae,), device=cfg.device)
    mean_values_S = torch.zeros((sae.cfg.d_sae,), device=cfg.device)

    distributions_MN = torch.zeros(
        (cfg.n_distributions, dataloader.n_samples), device="cpu"
    )
    estimator = PercentileEstimator(
        cfg.percentile, dataloader.n_samples, shape=(sae.cfg.d_sae,)
    )

    logger.info("Loaded SAE and data.")

    for i, batch in enumerate(helpers.progress(dataloader, desc="picking top-k")):
        vit_acts_BD = batch["act"]
        _, sae_acts_BS, *_ = sae(vit_acts_BD.to(cfg.device))

        for sae_act_S in sae_acts_BS:
            estimator.update(sae_act_S)

        # TODO: clean this line up
        # distributions_MN[:, batch["image_i"]] = sae_acts_BS[
        #     :, : cfg.n_distributions
        # ].to("cpu")

        mean_values_S += einops.reduce(sae_acts_BS, "batch d_sae -> d_sae", "sum")
        sparsity_S += einops.reduce((sae_acts_BS > 0), "batch d_sae -> d_sae", "sum")

        # Get the top k values across the batch for each latent
        values, k = torch.topk(sae_acts_BS.cpu(), k=cfg.top_k, dim=0)

        i_im = batch["image_i"][k]
        i_p = batch["patch_i"][k]

        all_values = torch.cat((top_values, values), axis=0)
        top_values, k = torch.topk(all_values, k=cfg.top_k, axis=0)
        top_i_p = torch.gather(torch.cat((top_i_p, i_p), axis=0), 0, k)
        top_i_im = torch.gather(torch.cat((top_i_im, i_im), axis=0), 0, k)

        if i % 10 == 0:
            unique_count = len(torch.unique(top_i_im[top_i_im >= 0]))
            logger.info(
                f"Iteration {i}: Found {unique_count:,} unique images in top_i_im"
            )

    mean_values_S /= sparsity_S
    sparsity_S /= dataloader.n_samples

    # Now we need the values for each patch of the top images. This tensor is (k, d_sae, n_patches_per_img) with dtype float16. To achieve this, we need particular values from the dataset.

    values_KPS = get_top_im_acts(cfg, top_i_im)

    return TopKPatch(
        values_KPS,
        top_i_im,
        mean_values_S,
        sparsity_S,
        distributions_MN,
        estimator.estimate.cpu(),
    )


@jaxtyped(typechecker=beartype.beartype)
def get_top_im_acts(
    cfg: Config, top_i_im: Int[Tensor, "k d_sae"]
) -> Float[Tensor, "k d_sae n_patches_per_img"]:
    assert (top_i_im >= 0).all(), "All top images must be in dataset"

    # Create indexed dataset with same configuration
    dataset = saev.data.Dataset(
        saev.data.IndexedConfig(
            shard_root=cfg.data.shard_root,
            patches=cfg.data.patches,
            layer=cfg.data.layer,
            seed=cfg.data.seed,
            debug=cfg.data.debug,
        )
    )
    logger.info("Loaded indexed dataset.")

    # Get all unique images across all latents
    unique_im_i = torch.unique(top_i_im[top_i_im >= 0])

    # Calculate all patch indices we need to load
    indices = []
    for img_i in unique_im_i.tolist():
        # Add all patches for this image
        indices.append(
            img_i * dataset.metadata.n_tokens_per_img
            + int(dataset.metadata.cls_token)
            + torch.arange(dataset.metadata.n_patches_per_img)
        )
    indices = torch.cat(indices)
    logger.info("Need %d indices.", len(indices))
    assert len(indices) > 0, "No indices needed."

    # Create subset and dataloader
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1024, shuffle=False, num_workers=4
    )

    # Load all activations
    all_acts = []
    for batch in helpers.progress(loader, desc="loading acts"):
        all_acts.append(batch["act"])
    vit_acts_ND = torch.cat(all_acts, dim=0).to(cfg.device)

    # Get SAE activations for all patches
    sae = nn.load(cfg.ckpt).to(cfg.device)
    logger.info("Loaded SAE.")
    sae_acts = []
    for start, end in helpers.progress(
        helpers.batched_idx(len(vit_acts_ND), cfg.data.batch_size), desc="SAE", every=50
    ):
        _, f_x, *_ = sae(vit_acts_ND[start:end].to(cfg.device))
        sae_acts.append(f_x)
    sae_acts_NS = torch.cat(sae_acts, dim=0).to(cfg.device)

    sae_acts_ISP = einops.rearrange(
        sae_acts_NS,
        "(n_img n_patch) d_sae -> n_img d_sae n_patch",
        n_img=len(unique_im_i),
    )

    img_to_idx = torch.full((unique_im_i.max() + 1,), -1, dtype=torch.long)
    img_to_idx[unique_im_i] = torch.arange(len(unique_im_i))
    # Map top_i_im to indices in one shot
    gather_indices_K = img_to_idx[top_i_im]
    assert (gather_indices_K >= 0).all()

    # Expand for all patches and use gather
    gather_indices_ISP = (
        gather_indices_K[:, :, None]
        .to(sae_acts_ISP.device)
        .expand(-1, -1, dataset.metadata.n_patches_per_img)
    )

    sae_acts_KSP = torch.gather(sae_acts_ISP, 0, gather_indices_ISP)
    return sae_acts_KSP.cpu()


@beartype.beartype
def dump_activations(cfg: Config):
    """Dump ViT activation statistics for later use.

    The dataset described by ``cfg`` is processed to find the images or patches that maximally activate each SAE latent.  Various tensors summarising these activations are then written to ``cfg.root`` so they can be loaded by other tools.

    Args:
        cfg: options controlling which activations are processed and where the resulting files are saved.

    Returns:
        None. All data is saved to disk.
    """
    with torch.no_grad():
        if cfg.sort_by == "img":
            topk = get_topk_img(cfg)
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


@beartype.beartype
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
        # return dump_imgs(cfg)

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

    dataset = saev.data.images.get_dataset(cfg.images, img_transform=None)

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
            if cfg.sort_by == "img":
                elem = GridElement(example["image"], example["label"], torch.tensor([]))
            elif cfg.sort_by == "patch":
                elem = GridElement(example["image"], example["label"], values_p)
            else:
                typing.assert_never(cfg.sort_by)
            elems.append(elem)

            seen_i_im.add(i_im)

        # How to scale values.
        upper = None
        if top_values[i].numel() > 0:
            upper = top_values[i].max().item()

        for j, elem in enumerate(elems):
            img = make_img(elem, upper=upper)
            img.save(os.path.join(neuron_dir, f"{j}.png"))
            with open(os.path.join(neuron_dir, f"{j}.txt"), "w") as fd:
                fd.write(elem.label + "\n")

        # Metadata
        metadata = {
            "neuron": i,
            "log10_freq": torch.log10(sparsity[i]).item(),
            "log10_value": torch.log10(mean_values[i]).item(),
        }
        with open(os.path.join(neuron_dir, "metadata.json"), "w") as fd:
            json.dump(metadata, fd)


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
        self._estimate = self._estimate.to(x.device)

        self._estimate += step_size * (
            torch.sign(x - self._estimate) + 2 * self.percentile / 100 - 1.0
        )

    @property
    def estimate(self):
        return self._estimate
