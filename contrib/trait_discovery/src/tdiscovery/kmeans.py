""" """

import dataclasses
import pathlib
import typing as tp

import beartype
import sklearn.base
import torch
import tyro
from jaxtyping import Float, jaxtyped
from torch import Tensor

import saev.data


@jaxtyped(typechecker=beartype.beartype)
class MiniBatchKMeans(sklearn.base.BaseEstimator):
    """
    GPU-accelerated mini-batch k-means classifier, following the scikit-learn API as much as is reasonable.
    """

    def __init__(self, k: int, device="cuda"):
        self.k = k
        self.centroids = None
        self.device = device

    def partial_fit(self, batch: Float[Tensor, "batch d_model"]) -> tp.Self:
        """Update centroids with one batch."""
        if self.centroids is None:
            # Initialize: random selection from first batch
            indices = torch.randperm(batch.shape[0])[: self.K]
            self.centroids = batch[indices].to(self.device)
            return

        # Assign to nearest centroid
        distances = torch.cdist(batch, self.centroids)  # (batch, K)
        assignments = distances.argmin(dim=1)  # (batch,)

        # Update centroids (moving average)
        for k in range(self.k):
            mask = assignments == k
            if mask.sum() > 0:
                self.centroids[k] = 0.9 * self.centroids[k] + 0.1 * batch[mask].mean(
                    dim=0
                )

    def transform(
        self, batch: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch k"]:
        """Return distances to all centroids."""
        distances = torch.cdist(batch, self.centroids)
        # Option 1: raw distances
        return -distances  # negative so higher = closer
        # Option 2: soft assignment (probability-like)
        # return torch.softmax(-distances / temperature, dim=1)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    train_data: saev.data.ShuffledConfig = saev.data.ShuffledConfig()
    """Training data."""
    val_data: saev.data.ShuffledConfig = saev.data.ShuffledConfig()
    """Validation data."""
    n_train: int = 100_000_000
    """Number of training samples."""
    n_val: int = 10_000_000
    """Number of evaluation samples."""
    k: int = 1024 * 16
    """Number of clusters."""
    lr: float = 1e-4  # ? not sure what a good default is
    """Update rate."""  # ? not sure what a good description of LR is with respect to k-means
    device: tp.Literal["cuda", "cpu"] = "cuda"
    """Hardware device."""
    seed: int = 42
    """Random seed."""
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 24.0
    """Slurm job length in hours."""
    mem_gb: int = 128
    """Node memory in GB."""
    log_to: str = pathlib.Path("./logs")
    """Where to log Slurm job stdout/stderr."""
    dump_to: str = pathlib.Path("./results")
    """Where to write checkpoints."""


@beartype.beartype
def cli(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")], sweep: pathlib.Path | None = None
) -> int:
    return 1
