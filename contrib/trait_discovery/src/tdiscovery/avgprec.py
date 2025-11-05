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

import saev.data
import saev.data.datasets
import saev.data.transforms
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

    # TODO: check that the train shard SAE acts exist.
    # TODO: check that the val shard SAE acts exist.
    # TODO: check that the train probe metrics exist.
    # See docs/src/developers/disk-layout.md, src/save/disk.py and probe1d.py for how to do this.

    # Look at the call to np.saevz in probe1d.py
    with np.load(train_probe_metrics_fpath) as fd:
        train_loss = fd["loss"]
        w = fd["weights"]
        b = fd["biases"]

    # TODO: Choose best latents for each class using minimum train_loss for each class.
    # TODO: Measure AP using labels from the validation shards and inputs from SAE activations.
    # Since validation is typically small (< 1M examples) and we only care about the best latents (151 for ADE20K) we can just put it all in GPU memory (1M x 151 x 4 bytes = 604MB) and do one inference pass. Furthermore, we can use the GPU to do the AP calculation.
    raise NotImplementedError()


@beartype.beartype
def cli(cfg: tp.Annotated[Config, tyro.conf.arg(name="")], sweep: str) -> int:
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger("avgprec")

    logger.info("Started cli().")

    if sweep:
        sweep_fpath = pathlib.Path(sweep)
        sweep_dcts = saev.configs.load_sweep(sweep_fpath)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep_fpath)
            return 1

        cfgs, errs = saev.configs.load_cfgs(
            cfg,
            default=Config(),
            sweep_dcts=sweep_dcts,
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
    return 0
