"""
Unified pipeline for training and evaluating trait discovery methods on CUB-200.

This script combines the functionality of train_baseline.py and dump_cub200_scores.py
into a single workflow that:
1. Trains baseline methods (random, PCA, k-means) or loads pre-trained SAEs
2. Evaluates them on CUB-200 bird traits with different n_train values
3. Outputs structured results as JSON/CSV for analysis

Size key:
* B: Batch size
* D: ViT activation dimension (typically 768 or 1024)
* K: Number of prototypes (SAE latent dimension, k for k-means, number of principal components in PCA, etc)
* N: Number of images
* T: Number of traits in CUB-200-2011 (312)

TODO: check that train/test dataloaders have similar metadata (layer, vit checkpoint, vit family)
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
import numpy as np
import submitit
import torch
import tyro
from jaxtyping import Bool, Float, Int
from sklearn.metrics import average_precision_score
from tdiscovery import baselines, cub200, metrics, saes
from torch import Tensor

import saev.data
from saev import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for unified trait discovery evaluation pipeline.
    """

    # Method configuration
    method: tp.Literal["random", "pca", "kmeans", "linear-clf", "sae"] = "random"
    """Which method we are evaluating."""
    n_prototypes: int = 1024 * 32
    """Number of prototypes/components."""

    sae_ckpt: str = ""
    """Path to pre-trained SAE checkpoint (only for method='sae')."""

    # Data configuration
    train_data: saev.data.ShuffledConfig = dataclasses.field(
        default_factory=saev.data.ShuffledConfig
    )
    """Train activations shard configuration."""
    test_data: saev.data.ShuffledConfig = dataclasses.field(
        default_factory=saev.data.ShuffledConfig
    )
    """Test activations shard configuration."""
    cub_root: str = "/fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder"
    """Root with test/, train/ and metadata/ folders."""
    layer: int = 23
    """Layer to use. Overrides both train and test .layer options."""

    # Evaluation configuration
    n_train: int = -1
    """Number of images to use to pick best prototypes. Less than 0 indicates all images."""

    # Output configuration
    dump_to: str = os.path.join(".", "results")
    """Where to save evaluation results."""
    output_format: tp.Literal["json", "csv", "both"] = "json"
    """Output format for results."""

    seed: int = 42
    """Random seed."""

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
    """Get the appropriate scorer based on the method."""
    d_vit = saev.data.Metadata.load(cfg.train_data.shard_root).d_vit

    if cfg.method == "random":
        return baselines.RandomVectors(
            n_prototypes=cfg.n_prototypes, d=d_vit, seed=cfg.seed
        )
    elif cfg.method == "kmeans":
        return baselines.KMeans(n_means=cfg.n_prototypes, d=d_vit, seed=cfg.seed)
    elif cfg.method == "pca":
        return baselines.PCA(n_components=cfg.n_prototypes, d=d_vit, seed=cfg.seed)
    elif cfg.method == "linear-clf":
        return baselines.LinearClassifier(cub_root=cfg.cub_root, d=d_vit, seed=cfg.seed)
    elif cfg.method == "sae":
        if not cfg.sae_ckpt:
            raise ValueError("SAE checkpoint path required when method='sae'")
        return saes.SparseAutoencoderScorer(cfg.sae_ckpt)
    else:
        raise RuntimeError(f"Method '{cfg.method}' not implemented.")


@beartype.beartype
def calc_scores(
    dataloader: saev.data.ShuffledDataLoader,
    scorer: baselines.Scorer,
    *,
    chunk_size: int = 512,
) -> Float[Tensor, "N K"]:
    """Calculate prototype scores for all images."""
    torch.use_deterministic_algorithms(True)

    metadata = saev.data.Metadata.load(dataloader.cfg.shard_root)
    shape = (dataloader.n_samples // metadata.n_patches_per_img, scorer.n_prototypes)
    scores_NK = torch.full(shape, -torch.inf)

    for batch in helpers.progress(dataloader, desc="scoring patches"):
        act_BD = batch["act"]
        img_i_B = batch["image_i"]

        patch_scores_BK = scorer(act_BD)
        bsz, k = patch_scores_BK.shape

        for start, end in helpers.batched_idx(k, chunk_size):
            dst = scores_NK[:, start:end]
            src = patch_scores_BK[:, start:end]
            idx = img_i_B.unsqueeze(1).expand(bsz, end - start).to(torch.int64)
            dst.scatter_reduce_(0, idx, src, reduce="amax")

    return scores_NK


@beartype.beartype
def pick_best_prototypes(
    scores_NK: Float[Tensor, "N K"],
    y_true_NT: Bool[Tensor, "N T"],
    *,
    chunk_size: int = 512,
) -> Int[Tensor, " T"]:
    """Pick the best prototype for each trait based on AP."""
    n, t = y_true_NT.shape
    _, k = scores_NK.shape
    best_ap_T = torch.full((t,), -1.0, dtype=torch.float32)
    best_idx_T = torch.full((t,), -1, dtype=torch.int64)

    for start, end in helpers.batched_idx(k, chunk_size):
        ap_CT = metrics.calc_avg_prec(scores_NK[:, start:end], y_true_NT)
        max_in_chunk, row_idx = ap_CT.max(dim=0)
        update_mask = max_in_chunk > best_ap_T
        best_ap_T[update_mask] = max_in_chunk[update_mask]
        best_idx_T[update_mask] = start + row_idx[update_mask]

    return best_idx_T.cpu()


@beartype.beartype
def eval_cub(
    cfg: Config,
    scorer: baselines.Scorer,
    train_dataloader: saev.data.ShuffledDataLoader,
    test_dataloader: saev.data.ShuffledDataLoader,
) -> list[dict[str, object]]:
    """Evaluate scorer on CUB-200 traits and return per-trait results."""
    logger = logging.getLogger("evaluate")

    try:
        train_y_true_NT = cub200.load_attrs(cfg.cub_root, is_train=True)
        test_y_true_NT = cub200.load_attrs(cfg.cub_root, is_train=False)
    except Exception:
        logger.exception("Could not load CUB attributes.")
        raise

    # Calculate train scores
    logger.info("Calculating training scores...")
    train_scores_NK = calc_scores(train_dataloader, scorer)

    # Sample training data if requested
    if cfg.n_train > 0 and cfg.n_train < len(train_scores_NK):
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(train_scores_NK), size=cfg.n_train, replace=False)
        idx = torch.from_numpy(np.sort(idx))
        train_scores_NK = train_scores_NK[idx]
        train_y_true_NT = train_y_true_NT[idx]

    # Pick best prototypes
    logger.info("Selecting best prototypes per trait...")
    best_idx_T = pick_best_prototypes(train_scores_NK, train_y_true_NT)

    # Calculate test scores
    logger.info("Calculating test scores...")
    test_scores_NK = calc_scores(test_dataloader, scorer)

    # Evaluate each trait
    results = []

    for trait_idx in range(len(best_idx_T)):
        prototype_idx = best_idx_T[trait_idx].item()
        scores = test_scores_NK[:, prototype_idx].cpu().numpy()
        labels = test_y_true_NT[:, trait_idx].cpu().numpy()

        # Calculate average precision
        ap = average_precision_score(labels, scores)

        result = {
            "method": cfg.method,
            "n_prototypes": cfg.n_prototypes,
            "n_train": cfg.n_train if cfg.n_train > 0 else len(train_y_true_NT),
            "seed": cfg.seed,
            "trait_idx": trait_idx,
            "average_precision": float(ap),
            "best_prototype_idx": prototype_idx,
        }

        # Add SAE-specific info if applicable
        if cfg.method == "sae":
            result["sae_ckpt"] = cfg.sae_ckpt
            # Could add more SAE metrics here if available

        results.append(result)

    return results


@beartype.beartype
def worker_fn(cfg: Config):
    """Main worker function that trains and evaluates a single configuration."""
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("worker")

    try:
        # Create data loaders
        train_dataloader = saev.data.ShuffledDataLoader(
            dataclasses.replace(cfg.train_data, layer=cfg.layer)
        )
        test_dataloader = saev.data.ShuffledDataLoader(
            dataclasses.replace(cfg.test_data, layer=cfg.layer)
        )
    except Exception:
        logger.exception(
            "Could not create dataloader. Please create a dataset using saev.data first."
        )
        return

    # Get scorer (train if needed)
    scorer = get_scorer(cfg)

    # Train baseline methods
    logger.info("Training scorer.")
    scorer.train(train_dataloader)

    # Evaluate on CUB-200
    logger.info("Evaluating on CUB-200 traits.")
    results = eval_cub(cfg, scorer, train_dataloader, test_dataloader)

    logger.info("Dumping scores.")
    results = [
        {
            **dct,
            # model/layer info from metadata
            "vit_family": train_dataloader.metadata.vit_family,
            "vit_ckpt": train_dataloader.metadata.vit_ckpt,
            "layer": cfg.layer,
            "d_vit": train_dataloader.metadata.d_vit,
        }
        for dct in results
    ]

    os.makedirs(cfg.dump_to, exist_ok=True)

    # Create unique filename based on config
    safe_ckpt_name = helpers.fssafe(train_dataloader.metadata.vit_ckpt)
    filename_parts = [
        cfg.method,
        f"n-prototypes_{cfg.n_prototypes}",
        f"n-train_{cfg.n_train}",
        f"seed_{cfg.seed}",
        f"vit-ckpt_{safe_ckpt_name}",
        f"layer_{cfg.layer}",
    ]
    base_filename = "__".join(filename_parts)

    # Save as JSON
    if cfg.output_format in ["json", "both"]:
        json_path = os.path.join(cfg.dump_to, f"{base_filename}.json")
        with open(json_path, "w") as f:
            json.dump(results, f)
        logging.info(f"Saved JSON results to {json_path}")

    # Save as CSV
    if cfg.output_format in ["csv", "both"]:
        csv_path = os.path.join(cfg.dump_to, f"{base_filename}.csv")
        if results:
            keys = results[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            logging.info(f"Saved CSV results to {csv_path}")


@beartype.beartype
def main(cfg: tp.Annotated[Config, tyro.conf.arg(name="")], sweep: str = ""):
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("main")

    if sweep:
        with open(sweep, "rb") as fd:
            cfgs, errs = helpers.grid(cfg, tomllib.load(fd))

        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return
    else:
        cfgs = [cfg]

    # Calculate safe limits (95% of max, at least 2 smaller)
    max_array_size = helpers.get_slurm_max_array_size()
    max_submit_jobs = helpers.get_slurm_max_submit_jobs()

    if cfg.slurm_acct:
        safe_array_size = min(int(max_array_size * 0.95), max_array_size - 2)
        safe_array_size = max(1, safe_array_size)  # Ensure at least 1

        safe_submit_jobs = min(int(max_submit_jobs * 0.95), max_submit_jobs - 2)
        safe_submit_jobs = max(1, safe_submit_jobs)  # Ensure at least 1

        logger.info(
            "Using safe limits - Array size: %d (max: %d), Submit jobs: %d (max: %d)",
            safe_array_size,
            max_array_size,
            safe_submit_jobs,
            max_submit_jobs,
        )
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=0,
            ntasks_per_node=1,
            # Request 8 CPUs because I want more memory and I don't know how else to get memory.
            cpus_per_task=8,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
            array_parallelism=safe_array_size,  # Limit array size
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)
        safe_array_size = len(cfgs)  # No limit for debug executor
        safe_submit_jobs = len(cfgs)  # No limit for debug executor

    if cfg.slurm_acct and len(cfgs) > safe_array_size:
        logger.info(
            "Will submit %d jobs in batches of %d",
            len(cfgs),
            safe_array_size,
        )

    # Process jobs in batches to respect Slurm's maximum array size and QOS limits
    all_jobs = []
    batches = helpers.batched_idx(len(cfgs), safe_array_size)

    for b, (start, end) in enumerate(batches):
        batch_cfgs = cfgs[start:end]

        # Check current job count and adjust batch size if needed
        if cfg.slurm_acct:
            current_jobs = helpers.get_slurm_job_count()
            jobs_available = max(0, safe_submit_jobs - current_jobs)

            # Wait if we're at the QOS limit
            while jobs_available < len(batch_cfgs):
                logger.info(
                    "Can only submit %d jobs but need %d. Waiting for more jobs to complete.",
                    jobs_available,
                    len(batch_cfgs),
                )

                # Wait and check again
                time.sleep(120)  # Wait 30 seconds before checking again
                current_jobs = helpers.get_slurm_job_count()
                jobs_available = max(0, safe_submit_jobs - current_jobs)

        logger.info(
            "Submitting batch %d/%d: jobs %d-%d (%d jobs)",
            b + 1,
            len(batches),
            start,
            start + len(batch_cfgs) - 1,
            len(batch_cfgs),
        )

        with executor.batch():
            batch_jobs = [executor.submit(worker_fn, c) for c in batch_cfgs]

        all_jobs.extend(batch_jobs)
        logger.info("Submitted batch %d/%d", b + 1, len(batches))

    logger.info("Submitted %d total jobs.", len(all_jobs))
    for j, job in enumerate(all_jobs):
        job.result()
        logger.info("Job %d finished (%d/%d).", j, j + 1, len(all_jobs))

    logger.info("Done.")


if __name__ == "__main__":
    tyro.cli(main)
