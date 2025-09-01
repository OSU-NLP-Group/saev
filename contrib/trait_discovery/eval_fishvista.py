"""
Unified pipeline for training and evaluating trait discovery methods on FishVista.

This script combines training and evaluation for fast methods (RandomVectors, PCA, KMeans) and evaluation-only for pre-trained SAEs. Unlike CUB200 which has image-level annotations, FishVista has patch-level segmentation masks, allowing for direct patch-level evaluation.

Size key:
* B: Batch size
* D: ViT activation dimension (typically 768 or 1024)
* K: Number of prototypes/components
* N: Number of images
* P: Number of patches (typically 640 for FishVista)
* H: Height in pixels
* W: Width in pixels
* C: Number of segmentation classes (10 for FishVista)
"""

import csv
import dataclasses
import json
import logging
import os
import os.path
import time
import tomllib
import typing as tp

import beartype
import einops
import numpy as np
import submitit
import torch
import tyro
from jaxtyping import Bool, Float, Int, jaxtyped
from tdiscovery import baselines, fishvista, metrics, saes
from torch import Tensor

import saev.data
import saev.data.images
import saev.data.transforms
import saev.helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for unified trait discovery evaluation pipeline on FishVista.
    """

    # Method configuration
    method: tp.Literal["random", "pca", "kmeans", "sae"] = "random"
    """Which method we are evaluating."""
    n_prototypes: int = 1024 * 32
    """Number of prototypes/components."""

    sae_ckpt: str = ""
    """Path to pre-trained SAE checkpoint (only for method='sae')."""

    # Data configuration
    train_acts: saev.data.ShuffledConfig = saev.data.ShuffledConfig()
    """Train activations shard configuration."""
    train_imgs: saev.data.images.SegFolder = saev.data.images.SegFolder()
    test_acts: saev.data.OrderedConfig = saev.data.OrderedConfig()
    """Test activations shard configuration."""
    test_imgs: saev.data.images.SegFolder = saev.data.images.SegFolder()

    # FishVista-specific configuration
    n_classes: int = 10
    """Number of segmentation classes in FishVista."""
    patch_size: int = 16
    """Patch size in pixels."""

    # Evaluation configuration
    top_k_prototypes: list[int] = dataclasses.field(
        default_factory=lambda: [1, 5, 10, 20, 50, 100]
    )
    """Different numbers of top prototypes to evaluate."""
    n_train: int = -1
    """Number of images to use to pick best prototypes. Less than 0 indicates all images."""

    # Output configuration
    dump_to: str = os.path.join(".", "results")
    """Where to save evaluation results."""
    output_format: tp.Literal["json", "csv", "both"] = "json"
    """Output format for results."""

    seed: int = 42
    """Random seed."""
    device: str = "cuda"
    """Device to use for computation."""

    # Slurm configuration
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 6.0
    """Slurm job length in hours."""
    log_to: str = os.path.join(".", "logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
def get_scorer(cfg: Config) -> baselines.Scorer:
    """Initialize the scorer based on the method."""
    d_vit = saev.data.Metadata.load(cfg.train_acts.shard_root).d_vit

    if cfg.method == "random":
        return baselines.RandomVectors(
            n_prototypes=cfg.n_prototypes, d=d_vit, seed=cfg.seed
        )
    elif cfg.method == "kmeans":
        return baselines.KMeans(
            n_means=cfg.n_prototypes, d=d_vit, seed=cfg.seed, device=cfg.device
        )
    elif cfg.method == "pca":
        return baselines.PCA(
            n_components=cfg.n_prototypes, d=d_vit, seed=cfg.seed, device=cfg.device
        )
    elif cfg.method == "sae":
        if not cfg.sae_ckpt:
            raise ValueError("sae_ckpt must be provided for method='sae'")
        return saes.load(cfg.sae_ckpt)
    else:
        raise RuntimeError(f"Method '{cfg.method}' not implemented.")


@jaxtyped(typechecker=beartype.beartype)
def make_keep_mask(n_total: int, n_keep: int, *, seed: int) -> Bool[Tensor, " n_total"]:
    if n_keep < 0:
        return torch.ones(n_total, dtype=torch.bool)

    if n_keep >= n_total:
        return torch.ones(n_total, dtype=torch.bool)

    g = torch.Generator(device="cpu").manual_seed(seed)
    keep_idx_k = torch.randperm(n_total, generator=g)[:n_keep]
    keep_mask_n = torch.zeros(n_total, dtype=torch.bool)
    keep_mask_n[keep_idx_k] = True
    return keep_mask_n


@torch.no_grad()
@jaxtyped(typechecker=beartype.beartype)
def compute_patch_scores(
    cfg: Config,
    scorer: baselines.Scorer,
    acts_dl: saev.data.OrderedDataLoader,
    imgs_dl: torch.utils.data.DataLoader,
    *,
    n: int = -1,
) -> tuple[Float[Tensor, "n k"], Int[Tensor, " n"]]:
    """
    Compute prototype scores for all patches in the dataset.

    Returns:
        scores: Prototype scores for each patch
        labels: Ground truth segmentation labels for each patch
    """
    n_patches = acts_dl.metadata.n_imgs * acts_dl.metadata.n_patches_per_img
    keep_mask_n = make_keep_mask(n_patches, n, seed=cfg.seed)
    if n < 0:
        scores_nk = torch.full((n_patches, scorer.n_prototypes), -torch.inf)
        labels_n = torch.full((n_patches,), -1, dtype=int)
    else:
        scores_nk = torch.full((n, scorer.n_prototypes), -torch.inf)
        labels_n = torch.full((n,), -1, dtype=int)

    assert len(labels_n) == keep_mask_n.int().sum().item()

    breakpoint()

    scorer = scorer.to(cfg.device)
    scorer.eval()

    n_kept_total = 0
    for acts, imgs in zip(
        saev.helpers.progress(acts_dl, desc="Computing scores"), imgs_dl
    ):
        # Check that our two dataloaders are synced up.
        image_i_b = imgs["index"].repeat_interleave(acts_dl.metadata.n_patches_per_img)
        assert (image_i_b == acts["image_i"]).all()

        patch_i_b = torch.arange(acts_dl.metadata.n_patches_per_img).repeat(
            len(imgs["index"])
        )
        assert (patch_i_b == acts["patch_i"]).all()

        # Check if any of these patches are kept.
        i_b = image_i_b * acts_dl.metadata.n_patches_per_img + patch_i_b
        keep_b = keep_mask_n[i_b]
        if not keep_b.any():
            continue

        # Do inference.
        acts_bd = acts["act"][keep_b].to(cfg.device)
        # Open a fresh jaxtyping dynamic shape context so B rebinds per batch; without this, the outer @jaxtyped pins B from earlier iterations (e.g., 16000) and a smaller final batch (e.g., 10240) triggers a shape violation.
        with jaxtyped("context"):
            scores_bk = scorer(acts_bd)

        # Update scores_nk and labels_n in the correct indices.
        n_kept_batch = keep_b.sum().item()
        scores_nk[n_kept_total : n_kept_total + n_kept_batch] = scores_bk
        labels_n[n_kept_total : n_kept_total + n_kept_batch] = einops.rearrange(
            imgs["patch_labels"], "imgs patches -> (imgs patches)"
        )[keep_b]
        n_kept_total += n_kept_batch

    return scores_nk, labels_n


@jaxtyped(typechecker=beartype.beartype)
def evaluate(
    train_scores_nk: Float[Tensor, "n_train k"],
    train_labels_n: Int[Tensor, " n_train"],
    test_scores_nk: Float[Tensor, "n_test k"],
    test_labels_n: Int[Tensor, " n_test"],
    cfg: Config,
    k: int,
) -> dict[str, float]:
    train_labels_nc = torch.nn.functional.one_hot(
        train_labels_n, num_classes=fishvista.n_classes
    ).bool()

    # Pick best prototypes
    n, k = train_scores_nk.shape
    best_ap_c = torch.full((fishvista.n_classes,), -1.0, dtype=torch.float32)
    best_idx_c = torch.full((fishvista.n_classes,), -1, dtype=torch.int64)

    # I want the best prototype (measure via AP score) for each of the 10 fishvista classes.
    breakpoint()
    bsz = 512
    for start, end in saev.helpers.batched_idx(k, bsz):
        ap_bc = metrics.calc_avg_prec(train_scores_nk[:, start:end], train_labels_nc)
        max_in_chunk, row_idx = ap_bc.max(dim=0)
        update_mask = max_in_chunk > best_ap_c
        best_ap_c[update_mask] = max_in_chunk[update_mask]
        best_idx_c[update_mask] = start + row_idx[update_mask]
    breakpoint()


@beartype.beartype
def worker_fn(cfg: Config):
    """Main worker function for training and evaluation."""
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("worker")

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load scorer
    scorer = get_scorer(cfg)

    logger.info("Training '%s' with %d prototypes", cfg.method, cfg.n_prototypes)
    train_acts_md = saev.data.Metadata.load(cfg.train_acts.shard_root)
    acts_batch_size = (
        cfg.train_acts.batch_size // train_acts_md.n_patches_per_img
    ) * train_acts_md.n_patches_per_img
    imgs_batch_size = cfg.train_acts.batch_size // train_acts_md.n_patches_per_img
    train_acts_sh_dl = saev.data.ShuffledDataLoader(
        dataclasses.replace(cfg.train_acts, batch_size=acts_batch_size)
    )
    train_acts_dl = saev.data.OrderedDataLoader(
        saev.data.make_ordered_config(cfg.train_acts, batch_size=acts_batch_size)
    )
    train_imgs_dl = torch.utils.data.DataLoader(
        fishvista.utils.ImageDataset(cfg.train_imgs, "no-bg"),
        batch_size=imgs_batch_size,
        num_workers=4,
        shuffle=False,
    )
    scorer.fit(train_acts_sh_dl)

    # Evaluate on test set
    logger.info("Evaluating on test set")

    # We need the activations and the images to be be coordinated. This requires a bit of fiddling with respect to batch sizes.
    test_acts_md = saev.data.Metadata.load(cfg.test_acts.shard_root)
    acts_batch_size = (
        cfg.test_acts.batch_size // test_acts_md.n_patches_per_img
    ) * test_acts_md.n_patches_per_img
    imgs_batch_size = cfg.test_acts.batch_size // test_acts_md.n_patches_per_img
    test_acts_dl = saev.data.OrderedDataLoader(
        dataclasses.replace(cfg.test_acts, batch_size=acts_batch_size)
    )
    test_imgs_dl = torch.utils.data.DataLoader(
        fishvista.utils.ImageDataset(cfg.test_imgs, "no-bg"),
        batch_size=imgs_batch_size,
        num_workers=4,
        shuffle=False,
    )

    # Compute scores for all test patches
    test_scores, test_labels = compute_patch_scores(
        cfg, scorer, test_acts_dl, test_imgs_dl
    )
    # Also compute scores for train set to determine prototype assignments
    train_scores, train_labels = compute_patch_scores(
        cfg, scorer, train_acts_dl, train_imgs_dl, n=cfg.n_train
    )
    breakpoint()

    # Evaluate with different numbers of top prototypes
    # TODO: I'm pretty sure this doesn't work at all. Needs to be fundamentally rewritten/double-checked, in depth.
    results = []
    for k in cfg.top_k_prototypes:
        if k > cfg.n_prototypes:
            continue

        metrics = evaluate(train_scores, train_labels, test_scores, test_labels, cfg, k)

        # Use only top k prototypes
        top_k_indices = train_scores.abs().mean(dim=(0, 1)).topk(k).indices
        train_scores_k = train_scores[:, :, top_k_indices]
        test_scores_k = test_scores[:, :, top_k_indices]

        # Assign prototypes to classes based on train set
        prototype_to_class = assign_prototypes_to_classes(
            train_scores_k, train_labels, cfg.n_classes
        )

        # Evaluate on test set
        metrics = evaluate_segmentation(
            test_scores_k, test_labels, prototype_to_class, cfg.n_classes
        )

        result = {
            "method": cfg.method,
            "n_prototypes": cfg.n_prototypes,
            "top_k": k,
            "mIoU": metrics["mIoU"],
            "accuracy": metrics["accuracy"],
            "seed": cfg.seed,
        }
        results.append(result)

        logger.info(
            "k=%d: mIoU=%.4f, accuracy=%.4f",
            k,
            metrics["mIoU"],
            metrics["accuracy"],
        )

    # Save results
    os.makedirs(cfg.dump_to, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"fishvista_{cfg.method}_n{cfg.n_prototypes}_seed{cfg.seed}_{timestamp}"

    if cfg.output_format in ["json", "both"]:
        json_path = os.path.join(cfg.dump_to, f"{base_name}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved JSON results to %s", json_path)

    if cfg.output_format in ["csv", "both"]:
        csv_path = os.path.join(cfg.dump_to, f"{base_name}.csv")
        with open(csv_path, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        logger.info("Saved CSV results to %s", csv_path)


@beartype.beartype
def main(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")],
    sweep: str | None = None,
):
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("main")

    if sweep is not None:
        with open(sweep, "rb") as fd:
            cfgs, errs = saev.helpers.grid(cfg, tomllib.load(fd))

        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return
    else:
        cfgs = [cfg]

    if cfg.slurm_acct:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=8,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    with executor.batch():
        jobs = [executor.submit(worker_fn, c) for c in cfgs]

    logger.info("Submitted %d jobs.", len(jobs))
    for j, job in enumerate(jobs):
        job.result()
        logger.info("Job %d finished (%d/%d).", j, j + 1, len(jobs))

    logger.info("Done.")


if __name__ == "__main__":
    tyro.cli(main)
