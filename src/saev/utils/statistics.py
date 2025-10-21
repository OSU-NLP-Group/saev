import math
from collections.abc import Sequence

import beartype
import numpy as np
import torch
from jaxtyping import Int, jaxtyped
from torch import Tensor


@beartype.beartype
class PercentileEstimator:
    def __init__(
        self,
        percentile: float | int,
        total: int,
        lr: float = 1e-3,
        shape: tuple[int, ...] = (),
    ):
        self.percentile = percentile
        self.total = total
        self.lr = lr

        self._estimate = torch.zeros(shape)
        self._step = 0

    def update(self, x):
        """
        Update the estimator with a new value.

        This method maintains the marker positions using the P2 algorithm rules. When a new value arrives, it's placed in the appropriate position relative to existing markers, and marker positions are adjusted to maintain their desired percentile positions.

        Arguments:
            x: The new value to incorporate into the estimation
        """
        self._step += 1

        step_size = self.lr * (self.total - self._step) / self.total

        # Is a no-op if it's already on the same device.
        if isinstance(x, Tensor):
            self._estimate = self._estimate.to(x.device)

        self._estimate += step_size * (
            torch.sign(x - self._estimate) + 2 * self.percentile / 100 - 1.0
        )

    @property
    def estimate(self):
        return self._estimate


IndexLike = Tensor | np.ndarray | Sequence[int]


@beartype.beartype
def calc_batch_entropy(
    example_idx: IndexLike,
    token_idx: IndexLike,
    n_examples: int,
    content_tokens_per_example: int,
) -> dict[str, float]:
    """
    Compute entropy and coverage metrics for a batch of shuffled indices.

    The returned mapping includes raw entropy (natural log units), normalized entropy, and coverage ratios for both the example indices and the token indices.
    """
    example_idx_t = _to_tensor(example_idx)
    token_idx_t = _to_tensor(token_idx)
    if n_examples <= 0:
        raise ValueError("n_examples must be positive.")
    if content_tokens_per_example <= 0:
        raise ValueError("content_tokens_per_example must be positive.")

    if example_idx_t.ndim != 1:
        raise ValueError("example_idx must be 1D.")
    if token_idx_t.ndim != 1:
        raise ValueError("token_idx must be 1D.")
    if example_idx_t.numel() == 0:
        raise ValueError("example_idx must contain at least one element.")

    _assert_batch_dim(example_idx_t, token_idx_t)

    example_metrics = _add_prefix(
        "loader/example", _entropy_metrics(example_idx_t, n_examples)
    )
    token_metrics = _add_prefix(
        "loader/token", _entropy_metrics(token_idx_t, content_tokens_per_example)
    )

    return {**example_metrics, **token_metrics}


def _to_tensor(values: IndexLike) -> Tensor:
    if isinstance(values, Tensor):
        return values.to(torch.int64)
    if isinstance(values, np.ndarray):
        return torch.from_numpy(values).to(torch.int64)
    return torch.as_tensor(list(values), dtype=torch.int64)


@jaxtyped(typechecker=beartype.beartype)
def _entropy_metrics(indices: Int[Tensor, " batch"], support: int) -> dict[str, float]:
    _, counts = torch.unique(indices, return_counts=True)
    counts = counts.to(torch.float64)
    if counts.numel() == 0:
        return {
            "entropy": 0.0,
            "entropy_normalized": 0.0,
            "coverage": 0.0,
        }

    probs = counts / counts.sum()
    entropy = float(-(probs * probs.log()).sum().item())
    coverage = float(counts.numel() / support)
    normalized = 0.0 if support <= 1 else float(entropy / math.log(support))

    return {
        "entropy": entropy,
        "entropy_normalized": normalized,
        "coverage": coverage,
    }


def _add_prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


@jaxtyped(typechecker=beartype.beartype)
def _assert_batch_dim(
    example_idx: Int[Tensor, " batch"], token_idx: Int[Tensor, " batch"]
) -> None:
    del example_idx, token_idx
