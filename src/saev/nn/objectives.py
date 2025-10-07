import dataclasses
import typing

import beartype
import einops
import torch
from jaxtyping import Float, Int64, jaxtyped
from torch import Tensor

from . import modeling


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

    Reference code is here: https://github.com/noanabeshima/matryoshka-saes and the original reading is https://sparselatents.com/matryoshka.html and https://arxiv.org/pdf/2503.17547
    """

    sparsity_coeff: float = 4e-4
    """How much to weight sparsity loss term (if not using TopK/BatchTopK)."""
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
        self, sae: modeling.SparseAutoencoder, x: Float[Tensor, "batch d_model"]
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
        # Keep sparsity_coeff as mutable attribute for scheduler compatibility
        self.sparsity_coeff = cfg.sparsity_coeff

    def forward(
        self, sae: modeling.SparseAutoencoder, x: Float[Tensor, "batch d_model"]
    ) -> VanillaLoss:
        f_x = sae.encode(x)
        x_hat = einops.rearrange(sae.decode(f_x), "batch () d_model -> batch d_model")

        # Some values of x and x_hat can be very large. We can calculate a safe MSE
        mse_loss = mean_squared_err(x_hat, x)

        mse_loss = mse_loss.mean()
        l0 = (f_x > 0).float().sum(dim=1).mean(dim=0)
        l1 = f_x.abs().sum(dim=1).mean(dim=0)
        sparsity_loss = self.sparsity_coeff * l1

        return VanillaLoss(mse_loss, sparsity_loss, l0, l1)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True, slots=True)
class MatryoshkaLoss(Loss):
    """The composite loss terms for an training batch."""

    mse: Float[Tensor, ""]
    """Average of reconstruction loss (mean squared error) for all prefix lengths."""
    sparsity: Float[Tensor, ""]
    """Sparsity loss, typically lambda * L1."""
    l0: Float[Tensor, ""]
    """Sum of L0 magnitudes of hidden activations for all prefix lengths."""
    l1: Float[Tensor, ""]
    """Sum of L1 magnitudes of hidden activations for all prefix lengths."""

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
class MatryoshkaObjective(Objective):
    """Torch module for calculating the matryoshka loss for an SAE."""

    def __init__(self, cfg: Matryoshka):
        super().__init__()
        self.cfg = cfg
        # Keep sparsity_coeff as mutable attribute for scheduler compatibility
        self.sparsity_coeff = cfg.sparsity_coeff

    def forward(
        self, sae: modeling.SparseAutoencoder, x: Float[Tensor, "batch d_model"]
    ) -> MatryoshkaLoss:
        f_x = sae.encode(x)  # shape: (batch, d_sae)
        b, d_sae = f_x.shape

        # Sample prefix cuts
        prefixes = sample_prefixes(d_sae, self.cfg.n_prefixes)

        # Use the new decode API with prefixes
        x_hats = sae.decode(f_x, prefixes=prefixes)

        # Calculate losses
        mse_loss = mean_squared_err(
            x_hats,
            einops.repeat(
                x, "b d_model -> b prefixes d_model", prefixes=self.cfg.n_prefixes
            ),
        ).mean()

        # Calculate sparsity metrics on full encoding
        l0 = (f_x > 0).float().sum(dim=1).mean(dim=0)
        l1 = f_x.abs().sum(dim=1).mean(dim=0)
        sparsity_loss = self.sparsity_coeff * l1

        return MatryoshkaLoss(mse_loss, sparsity_loss, l0, l1)


@torch.no_grad()
@jaxtyped(typechecker=beartype.beartype)
def sample_prefixes(
    d_sae: int, n_prefixes: int, min_prefix_length: int = 1, pareto_power: float = 0.5
) -> Int64[Tensor, " n_prefixes"]:
    """
    Samples prefix lengths using a Pareto distribution. Derived from "Learning Multi-Level Features with
    Matryoshka Sparse Autoencoders" (https://doi.org/10.48550/arXiv.2503.17547)

    Args:
        d_sae: Total number of latent dimensions
        n_prefixes: Number of prefixes to sample
        min_prefix_length: Minimum length of any prefix
        pareto_power: Power parameter for Pareto distribution (lower = more uniform)

    Returns:
        torch.Tensor: Sorted prefix lengths
    """
    if n_prefixes <= 1:
        return torch.tensor([d_sae], dtype=torch.int64)

    assert n_prefixes <= d_sae

    # Calculate probability distribution favoring shorter prefixes
    lengths = torch.arange(1, d_sae)
    pareto_cdf = 1 - ((min_prefix_length / lengths.float()) ** pareto_power)
    pareto_pdf = torch.cat([pareto_cdf[:1], pareto_cdf[1:] - pareto_cdf[:-1]])
    probability_dist = pareto_pdf / pareto_pdf.sum()

    # Sample and sort prefix lengths
    sampled_indices = torch.multinomial(
        probability_dist, num_samples=n_prefixes - 1, replacement=False
    )

    # Convert indices to actual prefix lengths
    prefixes = lengths[sampled_indices]

    # Add n_latents as the final prefix
    prefixes = torch.cat((prefixes.detach().clone(), torch.tensor([d_sae])))

    prefixes, _ = torch.sort(prefixes, descending=False)

    return prefixes.to(torch.int64)


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
    upper = x.abs().max().clamp(min=1e-12)
    x = x / upper
    x_hat = x_hat / upper

    mse = (x_hat - x) ** 2
    # (sam): I am now realizing that we normalize by the L2 norm of x.
    if norm:
        mse /= torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-12
        return mse * upper

    return mse * upper * upper
