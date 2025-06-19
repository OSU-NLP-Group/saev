import dataclasses
import typing

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Vanilla:
    sparsity_coeff: float = 4e-4
    """How much to weight sparsity loss term."""


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Matryoshka:
    """
    Config for the Matryoshka loss for another arbitrary SAE class.

    Reference code is here: https://github.com/noanabeshima/matryoshka-saes and the original reading is https://sparselatents.com/matryoshka.html and https://arxiv.org/pdf/2503.17547.
    """

    n_prefixes: int = 10
    """Number of random length prefixes to use for loss calculation."""


ObjectiveConfig = Vanilla | Matryoshka


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True, slots=True)
class Loss:
    """The loss term for an autoencoder training batch."""

    @property
    def loss(self) -> Float[Tensor, ""]:
        """Total loss."""
        raise NotImplementedError()

    def metrics(self) -> dict[str, object]:
        raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
class Objective(torch.nn.Module):
    def forward(
        self,
        x: Float[Tensor, "batch d_model"],
        f_x: Float[Tensor, "batch d_sae"],
        x_hat: Float[Tensor, "batch d_model"],
    ) -> Loss:
        raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True, slots=True)
class VanillaLoss(Loss):
    """The vanilla loss terms for an training batch."""

    mse: Float[Tensor, ""]
    """Reconstruction loss (mean squared error)."""
    sparsity: Float[Tensor, ""]
    """Sparsity loss, typically lambda * L1."""
    l0: Float[Tensor, ""]
    """L0 magnitude of hidden activations."""
    l1: Float[Tensor, ""]
    """L1 magnitude of hidden activations."""

    @property
    def loss(self) -> Float[Tensor, ""]:
        """Total loss."""
        return self.mse + self.sparsity

    def metrics(self) -> dict[str, object]:
        return {
            "loss": self.loss.item(),
            "mse": self.mse.item(),
            "l0": self.l0.item(),
            "l1": self.l1.item(),
            "sparsity": self.sparsity.item(),
        }


@jaxtyped(typechecker=beartype.beartype)
class VanillaObjective(Objective):
    def __init__(self, cfg: Vanilla):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        x: Float[Tensor, "batch d_model"],
        f_x: Float[Tensor, "batch d_sae"],
        x_hat: Float[Tensor, "batch d_model"],
    ) -> VanillaLoss:
        # Some values of x and x_hat can be very large. We can calculate a safe MSE
        mse_loss = mean_squared_err(x_hat, x)

        mse_loss = mse_loss.mean()
        l0 = (f_x > 0).float().sum(axis=1).mean(axis=0)
        l1 = f_x.sum(axis=1).mean(axis=0)
        sparsity_loss = self.cfg.sparsity_coeff * l1

        return VanillaLoss(mse_loss, sparsity_loss, l0, l1)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True, slots=True)
class MatryoshkaLoss(Loss):
    """The composite loss terms for an training batch."""

    @property
    def loss(self) -> Float[Tensor, ""]:
        raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
class MatryoshkaObjective(Objective):
    """Torch module for calculating the matryoshka loss for an SAE."""

    def __init__(self, cfg: Matryoshka):
        super().__init__()
        self.cfg = cfg

    def forward(self) -> "MatryoshkaLoss.Loss":
        raise NotImplementedError()


@beartype.beartype
def get_objective(cfg: ObjectiveConfig) -> Objective:
    if isinstance(cfg, Vanilla):
        return VanillaObjective(cfg)
    elif isinstance(cfg, Matryoshka):
        return MatryoshkaObjective(cfg)
    else:
        typing.assert_never(cfg)


@jaxtyped(typechecker=beartype.beartype)
def ref_mean_squared_err(
    x_hat: Float[Tensor, "*d"], x: Float[Tensor, "*d"], norm: bool = False
) -> Float[Tensor, "*d"]:
    mse_loss = torch.pow((x_hat - x.float()), 2)

    if norm:
        mse_loss /= (x**2).sum(dim=-1, keepdim=True).sqrt()
    return mse_loss


@jaxtyped(typechecker=beartype.beartype)
def mean_squared_err(
    x_hat: Float[Tensor, "*batch d"], x: Float[Tensor, "*batch d"], norm: bool = False
) -> Float[Tensor, "*batch d"]:
    upper = x.abs().max()
    x = x / upper
    x_hat = x_hat / upper

    mse = (x_hat - x) ** 2
    # (sam): I am now realizing that we normalize by the L2 norm of x.
    if norm:
        mse /= torch.linalg.norm(x, axis=-1, keepdim=True) + 1e-12
        return mse * upper

    return mse * upper * upper
