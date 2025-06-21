"""
1. Check if activations exist. If they don't, ask user to write them to disk using saev.data then try again.
2. Fit k-means (or whatever method) to dataset. Do a couple hparams in parallel because disk read speeds are slow (multiple values of k, multiple values for the number of principal components, etc).
3. Follow the pseudocode in the experiment description to get some scores.
4. Write the results to disk in a JSON or SQLite format. Tell the reader to explore the results using a marimo notebook of some kind.

Size key:

* B: batch size
* D: ViT activation dimension (typically 768 or 1024)
* K: Number of prototypes (SAE latent dimension, k for k-means, number of principal components in PCA, etc)
* N: total number of images
"""

import dataclasses
import logging
import os.path
import typing

import beartype
import torch
import tyro
from jaxtyping import Float, jaxtyped
from torch import Tensor

import saev.data
import saev.nn
import saev.utils.scheduling
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
    n_samples: int = 500_000_000
    """Number of training samples (vectors)."""
    n_latents: int = 8 * 1024
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


@beartype.beartype
class Scorer:
    def __call__(self, activations: Float[Tensor, "B D"]) -> Float[Tensor, "B S"]:
        raise NotImplementedError()

    @property
    def n_latents(self) -> int:
        raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
def get_random_vectors(
    dataloader: saev.data.iterable.DataLoader,
    *,
    n_latents: int,
    n_samples: int,
    seed: int,
) -> Float[Tensor, "S D"]:
    """Uniformly sample n_latents vectors from a streaming DataLoader using reservoir sampling.

    Args:
        dataloader: DataLoader.
        n: Number of samples to keep.
    """
    reservoir = None  # (n, d) but lazily initialized.
    n_seen = 0
    rng = torch.Generator().manual_seed(seed)

    if dataloader.n_samples > n_samples:
        dataloader = saev.utils.scheduling.BatchLimiter(dataloader, n_samples=n_samples)

    for batch in helpers.progress(dataloader):
        x = batch["act"]
        bsz, d = x.shape

        # 1. Fill reservoir if not full
        # Init reservoir if not initialized.
        if reservoir is None:
            reservoir = torch.empty(n_latents, d, dtype=x.dtype)

        need = max(n - n_seen, 0)
        if need:
            take = min(need, bsz)
            reservoir[n_seen : n_seen + take] = x[:take]
            n_seen += take
            x = x[take:]
            bsz -= take
            assert bsz >= 0  # take is at most bsz, so bsz - take >= 0
            if bsz == 0:
                continue

        # 2. vectorised replacement for remaining items in batch
        idxs = torch.arange(n_seen, n_seen + bsz)  # global indices
        probs = (n / (idxs + 1)).to(dtype=torch.float32)  # shape (B,)
        keep_mask = torch.rand(bsz, generator=rng) < probs  # Bernoulli draws
        k = keep_mask.sum().item()
        if k:
            replace_pos = torch.randint(0, n, (k,), generator=rng)
            reservoir[replace_pos] = x[keep_mask]
        n_seen += bsz

    return reservoir


@beartype.beartype
class RandomVectors(Scorer):
    def __init__(self, dataloader: saev.data.iterable.DataLoader, cfg: Config):
        self.prototypes = get_random_vectors(
            dataloader, n=cfg.n_latents, n_samples=cfg.n_samples, seed=cfg.seed
        )

    def __call__(self, activations: Float[Tensor, "bsz d"]) -> Float[Tensor, "bsz p"]:
        return activations @ self.prototypes.T

    @property
    def n_latents(self) -> int:
        n_latents, d = self.prototypes.shape
        return n_latents


@beartype.beartype
def calc_scores(
    dataloader: saev.data.iterable.DataLoader,
    scorer: Scorer,
    *,
    device: str = "cuda",
    chunk_size: int = 512,
) -> Float[Tensor, "n_imgs n_latents"]:
    # Initialize score matrix with −inf  so max() works.
    scores_NS = torch.full(
        (dataloader.n_samples, scorer.n_latents), -torch.inf, device=device
    )

    for batch in helpers.progress(dataloader, desc="scoring patches"):
        act_BD = batch["act"].to(device, non_blocking=True)
        img_i_B = batch["image_i"].to(device=device)

        patch_scores_BS = scorer(act_BD)
        bsz, n_latents = patch_scores_BS.shape

        # we cannot replicate img_idx to (B,d_feat) in one go (memory!),
        # so update d_feat in manageable slices.
        for start, end in helpers.batched_idx(n_latents, chunk_size):
            # slice views avoid extra alloc
            dst = scores_NS[:, start:end]  # (n_images, C)
            src = patch_scores_BS[:, start:end]  # (B, C)

            # expand image indices once per slice
            idx = img_i_B.unsqueeze(1).expand(bsz, end - start)

            # in-place max-pool across patches → image rows
            dst.scatter_reduce_(0, idx, src, reduce="amax")

    return scores_NS.cpu()


@torch.no_grad()
@beartype.beartype
def main(cfg: typing.Annotated[Config, tyro.conf.arg(name="")]):
    try:
        train_dataloader = saev.data.iterable.DataLoader(cfg.train_data)
        test_dataloader = saev.data.iterable.DataLoader(cfg.test_data)
    except Exception:
        logger.exception(
            "Could not create dataloader. Please create a dataset using saev.data first. See src/saev/guide.md for more details."
        )
        return

    # Get d from cfg.data
    metadata_fpath = os.path.join(cfg.train_data.shard_root, "metadata.json")
    d = saev.data.writers.Metadata.load(metadata_fpath).d_vit

    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    scorer = RandomVectors(train_dataloader, cfg)
    scores = calc_scores(test_dataloader, scorer, d, device=cfg.device)
    breakpoint()


if __name__ == "__main__":
    tyro.cli(main)
