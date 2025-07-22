"""
All configs for all saev jobs.

## Import Times

This module should be very fast to import so that `python main.py --help` is fast.
This means that the top-level imports should not include big packages like numpy, torch, etc.
For example, `TreeOfLife.n_imgs` imports numpy when it's needed, rather than importing it at the top level.

Also contains code for expanding configs with lists into lists of configs (grid search).
Might be expanded in the future to support pseudo-random sampling from distributions to support random hyperparameter search, as in [this file](https://github.com/samuelstevens/sax/blob/main/sax/sweep.py).
"""

import collections.abc
import dataclasses
import itertools
import os
import typing

import beartype


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class ImagenetDataset:
    """Configuration for HuggingFace Imagenet."""

    name: str = "ILSVRC/imagenet-1k"
    """Dataset name on HuggingFace. Don't need to change this.."""
    split: str = "train"
    """Dataset split. For the default ImageNet-1K dataset, can either be 'train', 'validation' or 'test'."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires loading the dataset. If you need to reference this number very often, cache it in a local variable."""
        import datasets

        dataset = datasets.load_dataset(
            self.name, split=self.split, trust_remote_code=True
        )
        return len(dataset)


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class ImageFolderDataset:
    """Configuration for a generic image folder dataset."""

    root: str = os.path.join(".", "data", "split")
    """Where the class folders with images are stored."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires walking the directory structure. If you need to reference this number very often, cache it in a local variable."""
        n = 0
        for _, _, files in os.walk(self.root):
            n += len(files)
        return n


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Ade20kDataset:
    """ """

    root: str = os.path.join(".", "data", "ade20k")
    """Where the class folders with images are stored."""
    split: typing.Literal["training", "validation"] = "training"
    """Data split."""

    @property
    def n_imgs(self) -> int:
        if self.split == "validation":
            return 2000
        else:
            return 20210


DatasetConfig = ImagenetDataset | ImageFolderDataset | Ade20kDataset


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Activations:
    """
    Configuration for calculating and saving ViT activations.
    """

    data: DatasetConfig = dataclasses.field(default_factory=ImagenetDataset)
    """Which dataset to use."""
    dump_to: str = os.path.join(".", "shards")
    """Where to write shards."""
    vit_family: typing.Literal["clip", "siglip", "dinov2"] = "clip"
    """Which model family."""
    vit_ckpt: str = "ViT-L-14/openai"
    """Specific model checkpoint."""
    vit_batch_size: int = 1024
    """Batch size for ViT inference."""
    n_workers: int = 8
    """Number of dataloader workers."""
    d_vit: int = 1024
    """Dimension of the ViT activations (depends on model)."""
    vit_layers: list[int] = dataclasses.field(default_factory=lambda: [-2])
    """Which layers to save. By default, the second-to-last layer."""
    n_patches_per_img: int = 256
    """Number of ViT patches per image (depends on model)."""
    cls_token: bool = True
    """Whether the model has a [CLS] token."""
    n_patches_per_shard: int = 2_400_000
    """Number of activations per shard; 2.4M is approximately 10GB for 1024-dimensional 4-byte activations."""

    seed: int = 42
    """Random seed."""
    ssl: bool = True
    """Whether to use SSL."""

    # Hardware
    device: str = "cuda"
    """Which device to use."""
    n_hours: float = 24.0
    """Slurm job length."""
    slurm_acct: str = ""
    """Slurm account string."""
    slurm_partition: str = ""
    """Slurm partition."""
    log_to: str = "./logs"
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class DataLoad:
    """
    Configuration for loading activation data from disk.
    """

    shard_root: str = os.path.join(".", "shards")
    """Directory with .bin shards and a metadata.json file."""
    patches: typing.Literal["cls", "patches", "meanpool"] = "patches"
    """Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'patches' indicates it will return all patches. 'meanpool' returns the mean of all image patches."""
    layer: int | typing.Literal["all", "meanpool"] = -2
    """Which ViT layer(s) to read from disk. ``-2`` selects the second-to-last layer. ``"all"`` enumerates every recorded layer, and ``"meanpool"`` averages activations across layers."""
    clamp: float = 1e5
    """Maximum value for activations; activations will be clamped to within [-clamp, clamp]`."""
    n_random_samples: int = 2**19
    """Number of random samples used to calculate approximate dataset means at startup."""
    scale_mean: bool | str = True
    """Whether to subtract approximate dataset means from examples. If a string, manually load from the filepath."""
    scale_norm: bool | str = True
    """Whether to scale average dataset norm to sqrt(d_vit). If a string, manually load from the filepath."""


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Relu:
    d_vit: int = 1024
    exp_factor: int = 16
    """Expansion factor for SAE."""
    n_reinit_samples: int = 1024 * 16 * 32
    """Number of samples to use for SAE re-init. Anthropic proposes initializing b_dec to the geometric median of the dataset here: https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-bias. We use the regular mean."""
    remove_parallel_grads: bool = True
    """Whether to remove gradients parallel to W_dec columns (which will be ignored because we force the columns to have unit norm). See https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-optimization for the original discussion from Anthropic."""
    normalize_w_dec: bool = True
    """Whether to make sure W_dec has unit norm columns. See https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder for original citation."""
    seed: int = 0
    """Random seed."""

    @property
    def d_sae(self) -> int:
        return self.d_vit * self.exp_factor


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class JumpRelu:
    """Implementation of the JumpReLU activation function for SAEs. Not implemented."""

    pass


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class TopK:
    d_vit: int = 1024
    exp_factor: int = 16
    """Expansion factor for SAE."""
    n_reinit_samples: int = 1024 * 16 * 32
    """Number of samples to use for SAE re-init. Anthropic proposes initializing b_dec to the geometric median of the dataset here: https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-bias. We use the regular mean."""
    remove_parallel_grads: bool = True
    """Whether to remove gradients parallel to W_dec columns (which will be ignored because we force the columns to have unit norm). See https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-optimization for the original discussion from Anthropic."""
    normalize_w_dec: bool = True
    """Whether to make sure W_dec has unit norm columns. See https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder for original citation."""
    seed: int = 0
    """Random seed."""
    top_k: int = 32

    @property
    def d_sae(self) -> int:
        return self.d_vit * self.exp_factor


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class BatchTopK:
    d_vit: int = 1024
    exp_factor: int = 16
    """Expansion factor for SAE."""
    n_reinit_samples: int = 1024 * 16 * 32
    """Number of samples to use for SAE re-init. Anthropic proposes initializing b_dec to the geometric median of the dataset here: https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-bias. We use the regular mean."""
    remove_parallel_grads: bool = True
    """Whether to remove gradients parallel to W_dec columns (which will be ignored because we force the columns to have unit norm). See https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-optimization for the original discussion from Anthropic."""
    normalize_w_dec: bool = True
    """Whether to make sure W_dec has unit norm columns. See https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder for original citation."""
    seed: int = 0
    """Random seed."""
    top_k: int = 32

    @property
    def d_sae(self) -> int:
        return self.d_vit * self.exp_factor


SparseAutoencoder = Relu | JumpRelu | TopK | BatchTopK


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Vanilla:
    sparsity_coeff: float = 4e-4
    """How much to weight sparsity loss term."""


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Matryoshka:
    """
    Config for the Matryoshka loss for another arbitrary SAE class.

    Reference code is here: https://github.com/noanabeshima/matryoshka-saes and the original reading is https://sparselatents.com/matryoshka.html and https://arxiv.org/pdf/2503.17547.
    """

    sparsity_coeff: float = 4e-4
    """How much to weight sparsity loss term (if not using TopK/BatchTopK)."""
    n_prefixes: int = 10
    """Number of random length prefixes to use for loss calculation."""


Objective = Vanilla | Matryoshka


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Train:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    data: DataLoad = dataclasses.field(default_factory=DataLoad)
    """Data configuration"""
    n_workers: int = 32
    """Number of dataloader workers."""
    n_patches: int = 100_000_000
    """Number of SAE training examples."""
    sae: SparseAutoencoder = dataclasses.field(default_factory=Relu)
    """SAE configuration."""
    objective: Objective = dataclasses.field(default_factory=Vanilla)
    """SAE loss configuration."""
    n_sparsity_warmup: int = 0
    """Number of sparsity coefficient warmup steps."""
    lr: float = 0.0004
    """Learning rate."""
    n_lr_warmup: int = 500
    """Number of learning rate warmup steps."""
    sae_batch_size: int = 1024 * 16
    """Batch size for SAE training."""

    # Matryoshka-specific
    n_prefixes: int = 10
    """Number of prefixes to sample at each loss step (Matryoshka-specific)"""
    pareto_power: float = 0.5
    """Pareto power of prefix sampling distribution"""

    # Logging
    track: bool = True
    """Whether to track with WandB."""
    wandb_project: str = "saev"
    """WandB project name."""
    tag: str = ""
    """Tag to add to WandB run."""
    log_every: int = 25
    """How often to log to WandB."""
    ckpt_path: str = os.path.join(".", "checkpoints")
    """Where to save checkpoints."""

    device: typing.Literal["cuda", "cpu"] = "cuda"
    """Hardware device."""
    seed: int = 42
    """Random seed."""
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 24.0
    """Slurm job length in hours."""
    log_to: str = os.path.join(".", "logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Visuals:
    """Configuration for generating visuals from trained SAEs."""

    ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    data: DataLoad = dataclasses.field(default_factory=DataLoad)
    """Data configuration."""
    images: DatasetConfig = dataclasses.field(default_factory=ImagenetDataset)
    """Which images to use."""
    top_k: int = 128
    """How many images per SAE feature to store."""
    n_workers: int = 16
    """Number of dataloader workers."""
    topk_batch_size: int = 1024 * 16
    """Number of examples to apply top-k op to."""
    sae_batch_size: int = 1024 * 16
    """Batch size for SAE inference."""
    epsilon: float = 1e-9
    """Value to add to avoid log(0)."""
    sort_by: typing.Literal["cls", "img", "patch"] = "patch"
    """How to find the top k images. 'cls' picks images where the SAE latents of the ViT's [CLS] token are maximized without any patch highligting. 'img' picks images that maximize the sum of an SAE latent over all patches in the image, highlighting the patches. 'patch' pickes images that maximize an SAE latent over all patches (not summed), highlighting the patches and only showing unique images."""
    device: str = "cuda"
    """Which accelerator to use."""
    dump_to: str = os.path.join(".", "data")
    """Where to save data."""
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
    seed: int = 42
    """Random seed."""

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


##########
# SWEEPS #
##########


@beartype.beartype
def grid(cfg: Train, sweep_dct: dict[str, object]) -> tuple[list[Train], list[str]]:
    cfgs, errs = [], []
    for d, dct in enumerate(expand(sweep_dct)):
        # .sae is a nested field that cannot be naively expanded.
        sae_dct = dct.pop("sae")
        if sae_dct:
            sae_dct["seed"] = sae_dct.pop("seed", cfg.sae.seed) + cfg.seed + d
            dct["sae"] = dataclasses.replace(cfg.sae, **sae_dct)

        # .objective is a nested field that cannot be naively expanded.
        objective_dct = dct.pop("objective")
        if objective_dct:
            dct["objective"] = dataclasses.replace(cfg.objective, **objective_dct)

        # .data is a nested field that cannot be naively expanded.
        data_dct = dct.pop("data")
        if data_dct:
            dct["data"] = dataclasses.replace(cfg.data, **data_dct)

        try:
            cfgs.append(dataclasses.replace(cfg, **dct, seed=cfg.seed + d))
        except Exception as err:
            errs.append(str(err))

    return cfgs, errs


@beartype.beartype
def expand(config: dict[str, object]) -> collections.abc.Iterator[dict[str, object]]:
    """
    Expands dicts with (nested) lists into a list of (nested) dicts.
    """
    yield from _expand_discrete(config)


@beartype.beartype
def _expand_discrete(
    config: dict[str, object],
) -> collections.abc.Iterator[dict[str, object]]:
    """
    Expands any (possibly nested) list values in `config`
    """
    if not config:
        yield config
        return

    key, value = config.popitem()

    if isinstance(value, list):
        # Expand
        for c in _expand_discrete(config):
            for v in value:
                yield {**c, key: v}
    elif isinstance(value, dict):
        # Expand
        for c, v in itertools.product(
            _expand_discrete(config), _expand_discrete(value)
        ):
            yield {**c, key: v}
    else:
        for c in _expand_discrete(config):
            yield {**c, key: value}
