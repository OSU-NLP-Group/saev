import dataclasses
import typing as tp

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class NoSparsity:
    """No explicit sparsity penalty (e.g. for TopK/BatchTopK where k controls sparsity)."""

    kind: tp.Literal["no-sparsity"] = "no-sparsity"

    def loss(self, f_x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, ""]:
        return torch.tensor(0.0)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class L1Sparsity:
    kind: tp.Literal["l1-sparsity"] = "l1-sparsity"
    coeff: float = 1e-4

    def loss(self, f_x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, ""]:
        l1 = f_x.abs().sum(axis=1).mean(axis=0)
        return l1 * self.coeff


Sparsity = NoSparsity | L1Sparsity


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Relu:
    """Vanilla ReLU"""

    kind: tp.Literal["relu"] = "relu"
    sparsity: Sparsity = L1Sparsity(coeff=4e-4)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TopK:
    kind: tp.Literal["top-k"] = "top-k"
    top_k: int = 32
    """How many values are allowed to be non-zero."""
    sparsity: Sparsity = NoSparsity()

    def __post_init__(self):
        assert self.top_k > 0, "top_k must be a positive integer."


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class BatchTopK:
    kind: tp.Literal["batch-top-k"] = "batch-top-k"
    top_k: int = 32
    """How many values are allowed to be non-zero per sample in the batch."""
    sparsity: Sparsity = NoSparsity()
    momentum: float = 0.1

    def __post_init__(self):
        assert self.top_k > 0, "top_k must be a positive integer."


Config = Relu | TopK | BatchTopK


@jaxtyped(typechecker=beartype.beartype)
class ReluActivation(torch.nn.Module):
    def __init__(self, cfg: Relu):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
        return torch.nn.functional.relu(x)


@jaxtyped(typechecker=beartype.beartype)
class TopKActivation(torch.nn.Module):
    """
    Top-K activation function. For use as activation function of sparse encoder.
    """

    def __init__(self, cfg: TopK):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
        """
        Apply top-k activation to the input tensor.
        """

        bsz, d_sae = x.shape
        k = min(self.cfg.top_k, d_sae)
        _, idxs = torch.topk(x, k, dim=-1, sorted=False)
        mask = torch.zeros_like(x).scatter(-1, idxs, 1.0)

        return torch.mul(mask, x)


@jaxtyped(typechecker=beartype.beartype)
class BatchTopKActivation(torch.nn.Module):
    """
    BatchTopK activation and inference-time threshold for sparse autoencoders.

    This module implements a BatchTopK nonlinearity that enforces a fixed sparsity budget across a batch, together with an inference-time approximation that replaces the batch-coupled operation with a simple elementwise threshold.

    Training mode (model.train()):
        Given pre-activation codes x with shape [batch, d_sae], the BatchTopK activation flattens the batch to shape [batch * d_sae], selects the largest (batch * top_k) entries by value, and sets all other entries to zero. This enforces an average of exactly `top_k` active features per example while allowing the "activation budget" to move between examples in the batch.

        During training, we also estimate an inference threshold theta that approximates the effective cutoff induced by BatchTopK. For each batch, we compute the minimum positive activation that survives the BatchTopK mask and update an exponential moving average of this quantity. This running estimate plays the same role as BatchNorm running statistics: it is updated only in training mode and treated as fixed at inference.

    Eval mode (model.eval()):
        At inference time we do not apply a batch-coupled top-k, since that would make each example depend on the rest of the eval batch. Instead, we use the stored running threshold theta to define a JumpReLU nonlinearity:

            y = x if x > theta else 0

        applied elementwise and independently to each example. This preserves the approximate sparsity level learned during training, but makes the layer deterministic and sample-wise independent for evaluation, probing, and downstream use.

    Inputs:
        x: Tensor of shape [batch, d_sae] containing pre-activation codes.

    Outputs:
        Tensor of shape [batch, d_sae] with the same dtype and device as x, where either:
            - in training mode: exactly (batch * top_k) entries are non-zero across the batch due to the BatchTopK mask, or
            - in eval mode: entries are zeroed by an elementwise JumpReLU with the learned threshold theta.
    """

    def __init__(self, cfg: BatchTopK):
        super().__init__()
        self.cfg = cfg

        self.register_buffer("threshold", torch.tensor(0.0))

    def forward(self, x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
        """
        Apply top-k activation to each sample in the batch.
        """

        if not self.training:
            # Fallback: if θ is still 0 (e.g. never trained), just do ReLU.
            if self.threshold <= 0:
                return torch.where(x > 0, x, torch.zeros_like(x))

            return torch.where(x > self.threshold, x, torch.zeros_like(x))

        bsz, d_sae = x.shape
        x_flat = x.flatten()

        bsz, d_sae = x.shape
        k = min(self.cfg.top_k * bsz, d_sae * bsz)
        _, idxs = torch.topk(x_flat, k, sorted=False)
        mask = torch.zeros_like(x_flat).scatter(-1, idxs, 1.0).reshape(x.shape)

        x = torch.mul(mask, x)

        with torch.no_grad():
            # smallest positive activation in this batch (i.e. the effective threshold)
            pos = x[x > 0]
            if pos.numel() >= 0:
                # EMA update, like BatchNorm
                self.threshold.mul_(1 - self.cfg.momentum).add_(
                    self.cfg.momentum * pos.min()
                )

        return x


@beartype.beartype
def get_activation(cfg: Config) -> torch.nn.Module:
    if isinstance(cfg, Relu):
        return ReluActivation(cfg)
    elif isinstance(cfg, TopK):
        return TopKActivation(cfg)
    elif isinstance(cfg, BatchTopK):
        return BatchTopKActivation(cfg)
    else:
        tp.assert_never(cfg)
