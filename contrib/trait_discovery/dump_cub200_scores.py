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
import typing

import beartype
import cub200
import numpy as np
import torch
import tyro
from jaxtyping import Bool, Float, Int
from torch import Tensor

import saev.data
from saev import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("baselines")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

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
    n_prototypes: int = 8 * 1024
    """"""
    dump_to: str = os.path.join(".", "data")
    """"""

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


class Scorer:
    def __call__(self, activations: Float[Tensor, "B D"]) -> Float[Tensor, "B K"]:
        raise NotImplementedError()

    @property
    def n_prototypes(self) -> int:
        raise NotImplementedError()


def get_random_vectors(
    dataloader: saev.data.iterable.DataLoader, *, k: int, seed: int
) -> Float[Tensor, "K D"]:
    """Uniformly sample n_prototypes vectors from a streaming DataLoader using reservoir sampling.

    Args:
        dataloader: DataLoader.
    """
    d_vit = saev.data.Metadata.load(dataloader.cfg.shard_root).d_vit
    reservoir_KD = torch.empty(k, d_vit)

    n_seen = 0
    rng = torch.Generator().manual_seed(seed)

    for batch in helpers.progress(dataloader):
        x_BD = batch["act"]
        b, d = x_BD.shape

        # 1. Fill reservoir if not full
        need = max(k - n_seen, 0)
        if need:
            take = min(need, b)
            reservoir_KD[n_seen : n_seen + take] = x_BD[:take]
            n_seen += take
            x_BD = x_BD[take:]
            b -= take
            assert b >= 0  # take is at most bsz, so bsz - take >= 0
            if b == 0:
                continue

        # 2. Vectorized replacement for remaining items in batch
        idxs_B = torch.arange(n_seen, n_seen + b)  # global indices
        probs = (k / (idxs_B + 1)).to(dtype=torch.float32)  # shape (B,)
        keep_mask = torch.rand(b, generator=rng) < probs  # Bernoulli draws
        n_keep = keep_mask.sum().item()
        if n_keep:
            replace_pos = torch.randint(0, k, (n_keep,), generator=rng)
            reservoir_KD[replace_pos] = x_BD[keep_mask]
        n_seen += b

    return reservoir_KD


class RandomVectors(Scorer):
    def __init__(
        self, dataloader: saev.data.iterable.DataLoader, n_prototypes: int, *, seed: int
    ):
        self.prototypes_KD = get_random_vectors(dataloader, k=n_prototypes, seed=seed)

    def __call__(self, activations_BD: Float[Tensor, "B D"]) -> Float[Tensor, "B K"]:
        return activations_BD @ self.prototypes_KD.T

    @property
    def n_prototypes(self) -> int:
        k, d = self.prototypes_KD.shape
        return k


def calc_scores(
    dataloader: saev.data.iterable.DataLoader,
    scorer: Scorer,
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


def calc_avg_prec(
    scores_NC: Float[Tensor, "N C"], y_true_NT: Bool[Tensor, "N T"]
) -> Float[Tensor, "C T"]:
    """
    Vectorized implementation of average precision (AP).

    Step-by-step:
    * sort images by score  (per prototype)
    * walk down the list, accumulate TP and precision = TP / rank
    * AP_t = mean of precision at the ranks where y == 1

    Args:
        scores_NC: Scores for n images and c prototypes (where c << k to batch this calculation).
    """
    n, c = scores_NC.shape
    _, t = y_true_NT.shape

    # total positives per trait   P_t  (shape  (T,))
    pos_T = y_true_NT.sum(dim=0).to(torch.float32)
    pos_mask_T = pos_T > 0  # traits that exist in this split
    pos_T[pos_T == 0] = 1.0  # avoid divide‑by‑zero later

    # 1. Order indices for each prototype  — shape (N, C)
    idx_NC = torch.argsort(scores_NC, dim=0, descending=True)

    # 2. Gather labels in that order
    # y_sorted_NCT[n, c, t] == y_true_NT[idx_NC[n, c], t]
    # Add trait dimension to indices: (N,C) -> (N,C,T) via view expansion
    # einops.repeat(idx_NC, 'n c -> n c t', t=t)
    idx_NCT = idx_NC.unsqueeze(-1).expand(-1, -1, t)

    # Add prototype dimension to labels: (N,T) -> (N,C,T) via view expansion
    # einops.repeat(y_true_NT, 'n t -> n c t', c=c)
    y_NCT = y_true_NT.unsqueeze(1).expand(-1, c, -1)

    y_sorted_NCT = torch.gather(y_NCT, dim=0, index=idx_NCT)

    cum_tp_NCT = torch.cumsum(y_sorted_NCT.float(), dim=0)

    ranks = torch.arange(1, n + 1, device=scores_NC.device, dtype=torch.float32)
    ranks = ranks.view(-1, 1, 1)
    precision_NCT = cum_tp_NCT / ranks

    masked_precision = precision_NCT * y_sorted_NCT.float()
    sum_precision_CT = masked_precision.sum(dim=0)

    avg_precision_CT = sum_precision_CT / pos_T.unsqueeze(0)
    avg_precision_CT = avg_precision_CT * pos_mask_T.unsqueeze(0)

    return avg_precision_CT


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
        ap_CT = calc_avg_prec(scores_NK[:, start:end], y_true_NT)
        # need the row index of max per trait inside the chunk
        max_in_chunk, row_idx = ap_CT.max(dim=0)
        update_mask = max_in_chunk > best_ap_T
        best_ap_T[update_mask] = max_in_chunk[update_mask]
        best_idx_T[update_mask] = start + row_idx[update_mask]

    return best_idx_T.cpu()


def dump(cfg: Config, scores_NT: Float[Tensor, "N T"]):
    cfg_json = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    run_id = hashlib.sha256(cfg_json.encode()).hexdigest()[:16]
    dpath = os.path.join(cfg.dump_to, run_id)
    os.makedirs(dpath, exist_ok=False)

    with open(os.path.join(dpath, "config.json"), "w") as fd:
        json.dump(cfg_json, fd)

    with gzip.open(os.path.join(dpath, "scores.bin.gz"), "wb") as fd:
        np.save(fd, scores_NT.numpy())


@beartype.beartype
def worker_fn(cfg: Config):
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
        scorer = RandomVectors(train_dataloader, cfg.n_prototypes, seed=cfg.seed)

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
def main(cfg: typing.Annotated[Config, tyro.conf.arg(name="")]):
    import submitit

    if cfg.slurm_acct:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    with executor.batch():
        jobs = [executor.submit(worker_fn, cfg)]

    logger.info("Submitted %d jobs.", len(jobs))
    for j, job in enumerate(jobs, start=1):
        job.result()
        logger.info("Job %d/%d finished.", j + 1, len(jobs))

    logger.info("Done.")


if __name__ == "__main__":
    tyro.cli(main)
