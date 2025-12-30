"""
Measures metrics (average precision, precision, recall, F1) for trait discovery.

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
import scipy.sparse
import tyro

import saev.configs
import saev.data
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
    max_k: int = 256
    """How many patches to record labels (for purity@k)."""

    debug: bool = False
    """Debug logging."""

    # Hardware
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
    logger = logging.getLogger("metrics")

    logger.info("Started worker_fn().")

    run = saev.disk.Run(cfg.run)

    train_inference_dpath = run.inference / cfg.train_shards.name
    train_token_acts_fpath = train_inference_dpath / "token_acts.npz"
    msg = f"Train SAE acts missing: '{train_token_acts_fpath}'. Run inference.py."
    assert train_token_acts_fpath.exists(), msg

    val_inference_dpath = run.inference / cfg.test_shards.name
    val_token_acts_fpath = val_inference_dpath / "token_acts.npz"
    msg = f"Validation SAE acts missing: '{val_token_acts_fpath}'. Run inference.py."
    assert val_token_acts_fpath.exists(), msg

    train_probe_metrics_fpath = train_inference_dpath / "probe1d_metrics.npz"
    msg = f"Probe metrics missing: '{train_probe_metrics_fpath}'. Run probe1d.py."
    assert train_probe_metrics_fpath.exists(), msg

    val_labels_fpath = cfg.test_shards / "labels.bin"
    msg = f"Validation labels missing: '{val_labels_fpath}'."
    assert val_labels_fpath.exists(), msg

    with np.load(train_probe_metrics_fpath) as fd:
        train_loss_lc = fd["loss"]
        weights_lc = fd["weights"]
        biases_lc = fd["biases"]

    msg = f"Expected train loss to be 2D (latents x classes); found shape {train_loss_lc.shape}."
    assert train_loss_lc.ndim == 2, msg

    msg = f"Probe metric shapes are inconsistent: loss{train_loss_lc.shape}, weights{weights_lc.shape}, biases{biases_lc.shape}."
    assert weights_lc.shape == train_loss_lc.shape, msg
    assert biases_lc.shape == train_loss_lc.shape, msg

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

    val_md = saev.data.Metadata.load(cfg.test_shards)

    # Load SAE activations for the best latents.
    val_token_acts_csr = scipy.sparse.load_npz(val_token_acts_fpath)
    val_n_samples, val_n_latents = val_token_acts_csr.shape
    msg = f"Validation activations have {val_n_latents} latents; expected {n_latents}."
    assert val_n_latents == n_latents, msg

    val_labels = (
        np
        .memmap(
            val_labels_fpath,
            mode="r",
            dtype=np.uint8,
            shape=(val_md.n_examples, val_md.content_tokens_per_example),
        )
        .copy()
        .reshape(-1)
    )

    msg = f"Validation labels provide {val_labels.size} samples; expected {val_n_samples}."
    assert val_labels.size == val_n_samples, msg

    val_max_label = val_labels.max().item()
    msg = f"Validation labels include class id {val_max_label}; expected < {n_classes}."
    assert val_max_label < n_classes, msg

    msg = f"max_k must be positive; got {cfg.max_k}."
    assert cfg.max_k > 0, msg

    msg = f"max_k ({cfg.max_k}) exceeds validation samples ({val_n_samples})."
    assert cfg.max_k <= val_n_samples, msg

    topk = saev.helpers.csr_topk(val_token_acts_csr, k=cfg.max_k, axis=0)
    top_labels_dk = np.take(val_labels, topk.indices.T).astype(np.uint8, copy=False)
    logger.info("Captured top-%d labels for %d latents.", cfg.max_k, val_n_latents)

    def _purity_stats(k: int) -> tuple[float, float, float]:
        msg = f"Cannot compute purity@{k} when only top-{cfg.max_k} labels are stored."
        assert k <= cfg.max_k, msg
        labels_dk = top_labels_dk[:, :k]
        purities = np.empty(labels_dk.shape[0], dtype=np.float32)
        for latent_i in range(labels_dk.shape[0]):
            _, counts = np.unique(labels_dk[latent_i], return_counts=True)
            purities[latent_i] = counts.max() / k
        return (
            purities.mean().item(),
            purities.min().item(),
            purities.max().item(),
        )

    for purity_k in (16, 64, 256):
        if purity_k > cfg.max_k:
            logger.info("Skipping purity@%d because max_k=%d.", purity_k, cfg.max_k)
            continue
        purity_mean, purity_min, purity_max = _purity_stats(purity_k)
        logger.info(
            "purity@%d across %d latents: mean=%.6f, min=%.6f, max=%.6f.",
            purity_k,
            val_n_latents,
            purity_mean,
            purity_min,
            purity_max,
        )

    val_latents_best = val_token_acts_csr[:, best_latent_idx_c].toarray()
    val_scores_nc = val_latents_best * best_weights_c + best_biases_c

    val_labels_one_hot = np.zeros((val_n_samples, n_classes), dtype=np.float32)
    np.put_along_axis(val_labels_one_hot, val_labels[:, None], 1.0, axis=1)

    n_pos_c = val_labels_one_hot.sum(axis=0)
    pos_class_mask_c = n_pos_c > 0

    sort_idx_nc = np.argsort(val_scores_nc, axis=0)[::-1]
    labels_sorted_nc = np.take_along_axis(val_labels_one_hot, sort_idx_nc, axis=0)

    tp_nc = labels_sorted_nc.cumsum(axis=0)
    ranks = np.arange(1, val_n_samples + 1, dtype=np.float32).reshape(-1, 1)
    precision_nc = tp_nc / ranks
    n_pos_safe_c = np.clip(n_pos_c, a_min=1.0, a_max=None)
    recall_nc = tp_nc / n_pos_safe_c
    recall_shift_nc = np.concatenate(
        (np.zeros((1, n_classes), dtype=recall_nc.dtype), recall_nc[:-1, :]), axis=0
    )
    delta_recall_nc = recall_nc - recall_shift_nc
    ap_c = (precision_nc * delta_recall_nc).sum(axis=0)
    ap_c = np.where(pos_class_mask_c, ap_c, np.zeros_like(ap_c))

    pred_pos_nc = (val_scores_nc > 0.0).astype(np.float32)
    tp_c = (pred_pos_nc * val_labels_one_hot).sum(axis=0)
    n_pred_pos_c = pred_pos_nc.sum(axis=0)

    precision_c = np.divide(
        tp_c,
        n_pred_pos_c,
        out=np.zeros_like(tp_c),
        where=n_pred_pos_c > 0.0,
    )
    precision_c = np.where(pos_class_mask_c, precision_c, np.zeros_like(precision_c))

    recall_c = np.divide(
        tp_c,
        n_pos_c,
        out=np.zeros_like(tp_c),
        where=n_pos_c > 0.0,
    )
    recall_c = np.where(pos_class_mask_c, recall_c, np.zeros_like(recall_c))

    denom_c = precision_c + recall_c
    f1_c = np.divide(
        2.0 * precision_c * recall_c,
        denom_c,
        out=np.zeros_like(precision_c),
        where=denom_c > 0.0,
    )

    n_pos_classes = pos_class_mask_c.sum().item()
    pos_precision_c = precision_c[pos_class_mask_c]
    pos_recall_c = recall_c[pos_class_mask_c]
    pos_f1_c = f1_c[pos_class_mask_c]
    logger.info(
        "Computed validation precision/recall/F1 on %d positive classes: precision[mean]=%.6f, precision[min]=%.6f, precision[max]=%.6f, recall[mean]=%.6f, recall[min]=%.6f, recall[max]=%.6f, f1[mean]=%.6f, f1[min]=%.6f, f1[max]=%.6f.",
        n_pos_classes,
        pos_precision_c.mean().item(),
        pos_precision_c.min().item(),
        pos_precision_c.max().item(),
        pos_recall_c.mean().item(),
        pos_recall_c.min().item(),
        pos_recall_c.max().item(),
        pos_f1_c.mean().item(),
        pos_f1_c.min().item(),
        pos_f1_c.max().item(),
    )

    logger.info(
        "Computed validation AP on %d samples: n_pos_classes=%d, ap[mean]=%.6f, ap[min]=%.6f, ap[max]=%.6f.",
        val_n_samples,
        pos_class_mask_c.sum().item(),
        ap_c.mean().item(),
        ap_c.min().item(),
        ap_c.max().item(),
    )

    metrics_fname = f"probe1d_metrics__train-{cfg.train_shards.name}.npz"
    metrics_fpath = val_inference_dpath / metrics_fname
    np.savez(
        metrics_fpath,
        ap=ap_c,
        precision=precision_c,
        recall=recall_c,
        f1=f1_c,
        top_labels=top_labels_dk,
    )
    logger.info("Saved metrics to '%s'.", metrics_fpath)


@beartype.beartype
def cli(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")], sweep: pathlib.Path | None = None
) -> int:
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger("metrics")

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
        logger.error("No configs resolved for metrics sweep.")
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
        n_errs = 0
        for idx, c in enumerate(
            saev.helpers.progress(cfgs, desc="jobs", every=1), start=1
        ):
            try:
                logger.info("Running config %d/%d locally.", idx, len(cfgs))
                worker_fn(c)
            except Exception as err:
                logger.exception("Error with job %d/%d: %s", idx, len(cfgs), err)
                n_errs += 1

        logger.info("Jobs done. %d error(s)", n_errs)
        return 0

    return 0
