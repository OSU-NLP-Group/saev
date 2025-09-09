# src/saev/scripts/dump.py
import dataclasses
import logging
import os.path

import beartype
import einops
import numpy as np
import polars as pl
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

import saev.data
import saev.helpers
import saev.nn


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
    sae_batch_size: int = 1024 * 8
    """Batch size for SAE inference."""

    # Hardware
    device: str = "cuda"
    """Which accelerator to use."""
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 4.0
    """Slurm job length in hours."""
    log_to: str = os.path.join(".", "logs")
    """Where to log Slurm job stdout/stderr."""


@jaxtyped(typechecker=beartype.beartype)
def get_sae_acts(
    vit_acts: Float[Tensor, "n d_vit"], sae: saev.nn.SparseAutoencoder, cfg: Config
) -> Float[Tensor, "n d_sae"]:
    """
    Get SAE hidden layer activations for a batch of ViT activations.

    Args:
        vit_acts: Batch of ViT activations
        sae: Sparse autoencder.
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
def worker_fn(cfg: Config):
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("dump")

    assert cfg.data.patches == "image"

    sae = saev.nn.load(cfg.ckpt).to(cfg.device)
    md = saev.data.Metadata.load(cfg.data.shard_root)
    acts_ns = np.zeros((md.n_imgs, sae.cfg.d_sae))

    batch_size = cfg.data.batch_size // md.n_patches_per_img * md.n_patches_per_img
    n_imgs_per_batch = batch_size // md.n_patches_per_img
    dataloader = saev.data.OrderedDataLoader(
        dataclasses.replace(cfg.data, batch_size=batch_size),
    )

    logger.info("Loaded SAE and data.")

    for batch in saev.helpers.progress(dataloader, desc="acts"):
        vit_acts_bd = batch["act"]
        sae_acts_bs = get_sae_acts(vit_acts_bd, sae, cfg)

        i_im = torch.sort(torch.unique(batch["image_i"])).values
        values_sbp = einops.rearrange(
            sae_acts_bs,
            "(imgs patches) d_sae -> d_sae imgs patches",
            imgs=len(i_im),
            patches=md.n_patches_per_img,
        )

        # Checks that I did my reshaping correctly.
        assert values_sbp.shape[1] == i_im.shape[0]
        if not len(i_im) == n_imgs_per_batch:
            logger.warning(
                "Got %d images; expected %d images per batch.",
                len(i_im),
                n_imgs_per_batch,
            )

        acts_sb = torch.topk(values_sbp, 3, dim=-1).values.mean(dim=-1)
        acts_ns[i_im] = acts_sb.cpu().numpy().T
        break

    # Need to get the images with all their labels.
    vit_cls = saev.data.models.load_vit_cls(md.vit_family)
    img_transform = vit_cls.make_resize(md.vit_ckpt, scale=1)
    img_ds = saev.data.datasets.get_dataset(cfg.images, img_transform=img_transform)
    if not hasattr(img_ds, "samples") or not hasattr(img_ds, "classes"):
        # We're in trouble.
        logging.warning("Can't save .obs without '.samples'.")
        return

    obs = [
        {"path": path, "target": target, "label": img_ds.classes[target]}
        for path, target in img_ds.samples
    ]

    sae_ckpt_id = os.path.basename(os.path.dirname(cfg.ckpt))
    dump_to = os.path.join(cfg.dump_to, f"{sae_ckpt_id}.h5pl")
    os.makedirs(dump_to, exist_ok=True)
    np.savez(os.path.join(dump_to, "acts.npz"), acts_ns)
    pl.DataFrame(obs).write_parquet(os.path.join(dump_to, "obs.parquet"))
    # pl.DataFrame(var).write_parquet(os.path.join(dump_to, "var.parquet"))
    logger.info("Wrote acts, obs and vars to '%s'.", dump_to)
