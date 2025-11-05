"""
Measures average precision for trait discovery.

Evaluates a trained probe1d logistic regression model on a validation split.

1. Loads a probe1d logistic regression model (w matrix, b matrix)
2. Picks the best (latent, w, b) triplets for each class using training loss.
3. Measure AP using probe predictions on the validation split.


"""

import dataclasses
import logging
import pathlib
import typing as tp

import beartype
import numpy as np
import torch
import tyro

import saev.configs
import saev.disk
import saev.helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for AP evaluation pipeline on ImgSeg datasets."""

    run: pathlib.Path = pathlib.Path("./runs/abcdefg")
    """Run directory."""

    train_shards: pathlib.Path = pathlib.Path("./shards/01234567")
    """Training shards directory."""
    test_shards: pathlib.Path = pathlib.Path("./shards/abcdef01")
    """Test shards directory."""

    debug: bool = False
    """Debug logging."""

    # Hardware
    device: str = "cuda"
    """Which accelerator to use."""
    mem_gb: int = 80
    """Node memory in GB."""
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 6.0
    """Slurm job length in hours."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
def worker_fn(cfg: Config):
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger("avgprec")

    logger.info("Started worker_fn().")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA device available, using CPU.")
        cfg = dataclasses.replace(cfg, device="cpu")

    run = saev.disk.Run(cfg.run)

    train_inference_dpath = run.inference / cfg.train_shards.name
    train_token_acts_fpath = train_inference_dpath / "token_acts.npz"

    if not train_token_acts_fpath.exists():
        msg = f"Train SAE activations missing: '{train_token_acts_fpath}'. Run inference.py."
        logger.error(msg)
        raise FileNotFoundError(msg)

    val_inference_dpath = run.inference / cfg.test_shards.name
    val_token_acts_fpath = val_inference_dpath / "token_acts.npz"

    if not val_token_acts_fpath.exists():
        msg = f"Validation SAE activations missing: '{val_token_acts_fpath}'. Run inference.py."
        logger.error(msg)
        raise FileNotFoundError(msg)

    train_probe_metrics_fpath = train_inference_dpath / "probe1d_metrics.npz"
    if not train_probe_metrics_fpath.exists():
        msg = f"Probe metrics missing: '{train_probe_metrics_fpath}'. Run probe1d.py."
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Look at the call to np.saevz in probe1d.py
    with np.load(train_probe_metrics_fpath) as fd:
        train_loss_lc = fd["loss"]
        weights_lc = fd["weights"]
        biases_lc = fd["biases"]

    if train_loss_lc.ndim != 2:
        msg = f"Expected train loss to be 2D (latents x classes); found shape {train_loss_lc.shape}."
        logger.error(msg)
        raise ValueError(msg)

    if (
        weights_lc.shape != train_loss_lc.shape
        or biases_lc.shape != train_loss_lc.shape
    ):
        msg = "Probe metric shapes are inconsistent: loss{train_loss_lc.shape}, weights{weights_lc.shape}, biases{biases_lc.shape}."
        logger.error(msg)
        raise ValueError(msg)
    n_latents, n_classes = train_loss_lc.shape

    best_latent_idx_c = np.argmin(train_loss_lc, axis=0)
    class_idx_c = np.arange(n_classes)
    best_train_loss_c = train_loss_lc[best_latent_idx_c, class_idx_c]
    best_weights_c = weights_lc[best_latent_idx_c, class_idx_c]
    best_biases_c = biases_lc[best_latent_idx_c, class_idx_c]

    logger.info(
        "Selected best latents per class: n_classes=%d, unique_latents=%d, train_loss[mean]=%.6f, train_loss[min]=%.6f, train_loss[max]=%.6f.",
        n_classes,
        np.unique(best_latent_idx_c).size,
        best_train_loss_c.mean().item(),
        best_train_loss_c.min().item(),
        best_train_loss_c.max().item(),
    )

    # TODO: Measure AP using labels from the validation shards and inputs from SAE activations. Since validation is typically small (< 1M examples) and we only care about the best latents (151 for ADE20K) we can just put it all in GPU memory (1M x 151 x 4 bytes = 604MB) and do one inference pass. Furthermore, we can use the GPU to do the AP calculation.
    raise NotImplementedError()


@beartype.beartype
def cli(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")], sweep: pathlib.Path | None = None
) -> int:
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger("avgprec")

    logger.info("Started cli().")

    if sweep is not None:
        sweep_dcts = saev.configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            return 1

        cfgs, errs = saev.configs.load_cfgs(
            cfg, default=Config(), sweep_dcts=sweep_dcts
        )
        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return 1
    else:
        cfgs = [cfg]

    if not cfgs:
        logger.error("No configs resolved for avgprec sweep.")
        return 1

    logger.info("Prepared %d config(s).", len(cfgs))
    base_cfg = cfgs[0]
    if any(c.slurm_acct != base_cfg.slurm_acct for c in cfgs):
        logger.error("All configs must share the same slurm_acct.")
        return 1
    if any(c.slurm_partition != base_cfg.slurm_partition for c in cfgs):
        logger.error("All configs must share the same slurm_partition.")
        return 1
    if any(c.log_to != base_cfg.log_to for c in cfgs):
        logger.error("All configs must share the same log directory.")
        return 1

    if not base_cfg.slurm_acct:
        for idx, c in enumerate(cfgs, start=1):
            logger.info("Running config %d/%d locally.", idx, len(cfgs))
            worker_fn(c)
        logger.info("Jobs done.")
        return 0

    return 0
