# contrib/trait_discovery/dump_cub200_scores.py
"""
1. Check if activations exist. If they don't, ask user to write them to disk using saev.data then try again.
2. Fit k-means (or whatever method) to dataset. Do a couple hparams in parallel because disk read speeds are slow (multiple values of k, multiple values for the number of principal components, etc).
3. Follow the pseudocode in the experiment description to get some scores.
4. Write the results to disk in a JSON or SQLite format. Tell the reader to explore the results using a marimo notebook of some kind.

Size key:

* B: Batch size
* D: ViT activation dimension (typically 768 or 1024)
* K: Number of prototypes (SAE latent dimension, k for k-means, number of principal components in PCA, etc)
* N: Number of images
* T: Number of traits in CUB-200-2011 (312)
"""

import dataclasses
import gzip
import hashlib
import json
import logging
import os.path
import tomllib
import typing

import beartype
import numpy as np
import submitit
import torch
import tyro
from jaxtyping import Bool, Float, Int
from lib import baselines, cub200, metrics
from torch import Tensor

import saev.data
from saev import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    ckpt: str = ""
    """Path to checkpoint to evaluate."""
    train_data: saev.data.IterableConfig = dataclasses.field(
        default_factory=saev.data.IterableConfig
    )
    """Train activations."""
    test_data: saev.data.IterableConfig = dataclasses.field(
        default_factory=saev.data.IterableConfig
    )
    """Test activations."""
    cub_root: str = os.path.join(".", "CUB_200_2011_ImageFolder")
    """Root with test/, train/ and metadata/ folders."""
    n_train: int = -1
    """Number of images to use to pick best prototypes. Less than 0 indicates all images."""
    dump_to: str = os.path.join(".", "data")
    """Where to save model scores."""

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


def calc_scores(
    dataloader: saev.data.iterable.DataLoader,
    scorer: baselines.Scorer,
    *,
    chunk_size: int = 512,
) -> Float[Tensor, "N K"]:
    torch.use_deterministic_algorithms(True)

    metadata = saev.data.Metadata.load(dataloader.cfg.shard_root)
    shape = (dataloader.n_samples // metadata.n_patches_per_img, scorer.n_prototypes)
    # Initialize score matrix with −inf  so max() works.
    scores_NK = torch.full(shape, -torch.inf)

    for batch in helpers.progress(dataloader, desc="scoring patches"):
        act_BD = batch["act"]
        img_i_B = batch["image_i"]

        patch_scores_BK = scorer(act_BD)
        bsz, k = patch_scores_BK.shape

        # We cannot replicate img_i_B to (B,K) in one go (memory!), so update score_NK in manageable slices.
        for start, end in helpers.batched_idx(k, chunk_size):
            # slice views avoid extra alloc
            dst = scores_NK[:, start:end]
            src = patch_scores_BK[:, start:end]

            # expand image indices once per slice
            idx = img_i_B.unsqueeze(1).expand(bsz, end - start)

            # in-place max-pool across patches → image rows
            dst.scatter_reduce_(0, idx, src, reduce="amax")

    return scores_NK.cpu()


def pick_best_prototypes(
    scores_NK: Float[Tensor, "N K"],
    y_true_NT: Bool[Tensor, "N T"],
    *,
    chunk_size: int = 512,
    device: str = "cpu",
) -> Int[Tensor, " T"]:
    """
    Args:
        scores_NK:
        y_true_NT: Boolean attribute array; y_true_NT[n, t] is True if image n has trait t.

    Returns:
        A matrix of prototype indices (0...K-1) that maximizes AP for each trait.
    """
    n, t = y_true_NT.shape
    _, k = scores_NK.shape
    best_ap_T = torch.full((t,), -1.0, device=device, dtype=torch.float32)
    best_idx_T = torch.full((t,), -1, device=device, dtype=torch.int64)

    for start, end in helpers.batched_idx(k, chunk_size):
        ap_CT = metrics.calc_avg_prec(scores_NK[:, start:end], y_true_NT)
        # need the row index of max per trait inside the chunk
        max_in_chunk, row_idx = ap_CT.max(dim=0)
        update_mask = max_in_chunk > best_ap_T
        best_ap_T[update_mask] = max_in_chunk[update_mask]
        best_idx_T[update_mask] = start + row_idx[update_mask]

    return best_idx_T.cpu()


@beartype.beartype
def dump(cfg: Config, scores_NT: Float[Tensor, "N T"]):
    cfg_json = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    run_id = hashlib.sha256(cfg_json.encode()).hexdigest()[:16]
    dpath = os.path.join(cfg.dump_to, run_id)
    os.makedirs(dpath, exist_ok=False)

    with open(os.path.join(dpath, "config.json"), "w") as fd:
        fd.write(cfg_json + "\n")

    with gzip.open(os.path.join(dpath, "scores.bin.gz"), "wb") as fd:
        np.save(fd, scores_NT.numpy())


@beartype.beartype
def worker_fn(cfg: Config):
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("worker")

    try:
        train_y_true_NT = cub200.load_attrs(cfg.cub_root, is_train=True)
    except Exception:
        logger.exception("Could not load CUB attributes.")
        return

    try:
        train_dataloader = saev.data.iterable.DataLoader(cfg.train_data)
        test_dataloader = saev.data.iterable.DataLoader(cfg.test_data)
    except Exception:
        logger.exception(
            "Could not create dataloader. Please create a dataset using saev.data first. See src/saev/guide.md for more details."
        )
        return

    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    with torch.no_grad():
        scorer = baselines.load(cfg.ckpt)

        train_scores_NK = calc_scores(train_dataloader, scorer)

        # Sample a random subset of train_scores based on n_train.
        n_train, k = train_scores_NK.shape
        if cfg.n_train > 0 and cfg.n_train < n_train:
            rng = np.random.default_rng(seed=cfg.seed)
            indices = rng.choice(n_train, size=cfg.n_train, replace=False)
            train_scores_NK = train_scores_NK[indices]
            train_y_true_NT = train_y_true_NT[indices]

        # Get the best prototypes by calculating AP across all train images.
        prototypes_i_T = pick_best_prototypes(train_scores_NK, train_y_true_NT)

        test_scores_NK = calc_scores(test_dataloader, scorer)
        test_scores_NT = test_scores_NK[:, prototypes_i_T]

    dump(cfg, test_scores_NT)


@beartype.beartype
def main(
    cfg: typing.Annotated[Config, tyro.conf.arg(name="")], sweep: str | None = None
):
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("main")

    if sweep is not None:
        with open(sweep, "rb") as fd:
            cfgs, errs = helpers.grid(cfg, tomllib.load(fd))

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
            # Request 8 CPUs because I want more memory and I don't know how else to get memory.
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
