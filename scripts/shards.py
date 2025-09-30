"""
To save lots of activations, we want to do things in parallel, with lots of slurm jobs, and save multiple files, rather than just one.

This script handles that additional complexity.

Conceptually, activations are either thought of as

1. A single [n_imgs x n_layers x (n_patches + 1), d_vit] tensor. This is a *dataset*
2. Multiple [n_imgs_per_shard, n_layers, (n_patches + 1), d_vit] tensors. This is a set of sharded activations.
"""

import dataclasses
import logging
import math
import os
import pathlib
import typing as tp
from collections.abc import Callable

import beartype
import torch
import tyro

from saev import helpers
from saev.data import datasets, shards

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger("saev.data")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for calculating and saving ViT activations."""

    data: datasets.Config = dataclasses.field(default_factory=datasets.Imagenet)
    """Which dataset to use."""
    dump_to: str = os.path.join(".", "shards")
    """Where to write shards."""
    vit_family: tp.Literal["clip", "siglip", "dinov2", "dinov3", "fake-clip"] = "clip"
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
    pixel_agg: tp.Literal["majority", "prefer-fg"] = "majority"
    max_patches_per_shard: int = 2_400_000
    """Maximum number of activations per shard; 2.4M is approximately 10GB for 1024-dimensional 4-byte activations."""

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
def _is_segmentation_dataset(data_cfg: datasets.Config) -> bool:
    """
    Check if a dataset configuration is for a segmentation dataset.

    Args:
        data_cfg: Dataset configuration

    Returns:
        True if this is a segmentation dataset that should have labels.bin
    """
    # Check if it's FakeSeg or SegFolder
    return isinstance(data_cfg, (datasets.FakeSeg, datasets.SegFolder))


@beartype.beartype
def get_acts_dir(cfg) -> pathlib.Path:
    """
    Return the activations directory based on the relevant values of a config.
    Also saves a metadata.json file to that directory for human reference.

    Args:
        cfg: Config for experiment.

    Returns:
        Directory to where activations should be dumped/loaded from.
    """

    pixel_agg = None
    if _is_segmentation_dataset(cfg.data):
        pixel_agg = cfg.pixel_agg

    metadata = shards.Metadata(
        cfg.vit_family,
        cfg.vit_ckpt,
        tuple(cfg.vit_layers),
        cfg.n_patches_per_img,
        cfg.cls_token,
        cfg.d_vit,
        cfg.data.n_imgs,
        cfg.max_patches_per_shard,
        {**dataclasses.asdict(cfg.data), "__class__": cfg.data.__class__.__name__},
        pixel_agg,
    )
    acts_dir = os.path.join(cfg.dump_to, metadata.hash)
    os.makedirs(acts_dir, exist_ok=True)

    metadata.dump(acts_dir)

    return acts_dir


@beartype.beartype
def get_dataloader(
    cfg: Config,
    *,
    img_tr: Callable | None = None,
    seg_tr: Callable | None = None,
    sample_tr: Callable | None = None,
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for a default map-style dataset.

    Args:
        cfg: Config.
        img_tr: Image transform to be applied to each image.
        seg_tr: Segmentation transform to be applied to masks.
        sample_tr: Transform to be applied to sample dicts.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches, `'index'` keys containing original dataset indices and `'label'` keys containing label batches.
    """
    dataset = datasets.get_dataset(
        cfg.data, img_transform=img_tr, seg_transform=seg_tr, sample_transform=sample_tr
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.vit_batch_size,
        drop_last=False,
        num_workers=cfg.n_workers,
        persistent_workers=cfg.n_workers > 0,
        shuffle=False,
        pin_memory=False,
    )
    return dataloader


@beartype.beartype
def worker_fn(cfg: Config):
    """
    Args:
        cfg: Config for activations.
    """
    from saev.data import models

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("worker_fn")

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA device available, using CPU.")
        cfg = dataclasses.replace(cfg, device="cpu")

    vit_cls = models.load_vit_cls(cfg.vit_family)
    vit_instance = vit_cls(cfg.vit_ckpt).to(cfg.device)
    vit = shards.RecordedVisionTransformer(
        vit_instance, cfg.n_patches_per_img, cfg.cls_token, cfg.vit_layers
    )

    img_tr, sample_tr = vit_cls.make_transforms(cfg.vit_ckpt, cfg.n_patches_per_img)

    seg_tr = None
    if _is_segmentation_dataset(cfg.data):
        # For segmentation datasets, create a transform that converts pixels to patches
        # Use make_resize with NEAREST interpolation for segmentation masks
        from PIL import Image

        seg_resize_tr = vit_cls.make_resize(
            cfg.vit_ckpt, cfg.n_patches_per_img, scale=1.0, resample=Image.NEAREST
        )

        def seg_to_patches(seg):
            """Transform that resizes segmentation and converts to patch labels."""

            # Convert to patch labels
            return shards.pixel_to_patch_labels(
                seg_resize_tr(seg),
                cfg.n_patches_per_img,
                patch_size=vit_instance.patch_size,
                pixel_agg=cfg.pixel_agg,
                bg_label=cfg.data.bg_label,
            )

        seg_tr = seg_to_patches

    dataloader = get_dataloader(cfg, img_tr=img_tr, seg_tr=seg_tr, sample_tr=sample_tr)

    n_batches = math.ceil(cfg.data.n_imgs / cfg.vit_batch_size)
    logger.info("Dumping %d batches of %d examples.", n_batches, cfg.vit_batch_size)

    vit = vit.to(cfg.device)
    # vit = torch.compile(vit)

    # Use context manager for proper cleanup
    with shards.ShardWriter(cfg) as writer:
        i = 0
        # Calculate and write ViT activations.
        with torch.inference_mode():
            for batch in helpers.progress(dataloader, total=n_batches):
                imgs = batch.get("image").to(cfg.device)
                grid = batch.get("grid")
                if grid is not None:
                    grid = grid.to(cfg.device)
                    out, cache = vit(imgs, grid=grid)
                else:
                    out, cache = vit(imgs)
                # cache has shape [batch size, n layers, n patches + 1, d vit]
                del out

                # Write activations and labels (if present) in one call
                patch_labels = batch.get("patch_labels")
                if patch_labels is not None:
                    logger.debug(
                        f"Found patch_labels in batch: shape={patch_labels.shape if hasattr(patch_labels, 'shape') else 'unknown'}"
                    )
                    # Ensure correct shape
                    assert patch_labels.shape == (len(cache), cfg.n_patches_per_img)
                else:
                    logger.debug(f"No patch_labels in batch. Keys: {batch.keys()}")

                writer.write_batch(cache, i, patch_labels=patch_labels)

                i += len(cache)


@beartype.beartype
def main(cfg: tp.Annotated[Config, tyro.conf.arg(name="")]):
    """
    Save ViT activations for use later on.

    Args:
        cfg: Configuration for activations.
    """
    logger = logging.getLogger("dump")

    if not cfg.ssl:
        logger.warning("Ignoring SSL certs. Try not to do this!")
        # https://github.com/openai/whisper/discussions/734#discussioncomment-4491761
        # Ideally we don't have to disable SSL but we are only downloading weights.
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

    # Actually record activations.
    if cfg.slurm_acct:
        import submitit

        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=cfg.n_workers + 4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )

        job = executor.submit(worker_fn, cfg)
        logger.info("Running job '%s'.", job.job_id)
        job.result()

    else:
        worker_fn(cfg)


if __name__ == "__main__":
    tyro.cli(main)
