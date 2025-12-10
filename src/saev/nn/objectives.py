import dataclasses
import typing as tp

import beartype
import einops
import torch
from jaxtyping import Float, Int, Int64, jaxtyped
from torch import Tensor

from . import modeling


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Matryoshka:
    """
    Config for the Matryoshka loss for another arbitrary SAE class.

    Reference code is here: https://github.com/noanabeshima/matryoshka-saes and the original reading is https://sparselatents.com/matryoshka.html and https://arxiv.org/pdf/2503.17547
    """

    n_prefixes: int = 10
    """Number of random length prefixes to use for loss calculation."""
    dead_threshold_tokens: int = 10_000_000
    """Tokens without activation before a latent is considered dead."""


ObjectiveConfig = Matryoshka


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
        sae: modeling.SparseAutoencoder,
        x: Float[Tensor, "batch d_model"],
        *,
        enc: modeling.SparseAutoencoder.EncodeOut | None = None,
    ) -> Loss:
        raise NotImplementedError()


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
    aux: Float[Tensor, ""]
    """Auxiliary loss term (e.g., AuxK)."""
    n_dead: Int[Tensor, ""]
    """Number of dead latents (per aux loss threshold)."""

    @property
    def loss(self) -> Float[Tensor, ""]:
        """Total loss."""
        return self.mse + self.sparsity + self.aux

    def metrics(self) -> dict[str, object]:
        return {
            "loss": self.loss.item(),
            "mse": self.mse.item(),
            "l0": self.l0.item(),
            "l1": self.l1.item(),
            "sparsity": self.sparsity.item(),
            "aux": self.aux.item(),
            "n_dead": self.n_dead,
        }


@jaxtyped(typechecker=beartype.beartype)
class MatryoshkaObjective(Objective):
    """Torch module for calculating the matryoshka loss for an SAE."""

    def __init__(self, cfg: Matryoshka):
        super().__init__()
        self.cfg = cfg
        self.toks_since_active: Tensor | None = None

    def forward(
        self, sae: modeling.SparseAutoencoder, x: Float[Tensor, "batch d_model"]
    ) -> tuple[MatryoshkaLoss, modeling.SparseAutoencoder.Output]:
        enc = sae.encode(x)
        bsz, d_sae = enc.f_x.shape

        if self.training:
            if self.toks_since_active is None:
                self.toks_since_active = torch.zeros(
                    d_sae, device=enc.f_x.device, dtype=torch.int64
                )

            assert self.toks_since_active.shape == (d_sae,)
            with torch.no_grad():
                active_mask = (enc.f_x.abs() > 0).any(dim=0)
                msg = f"Active mask shape {active_mask.shape} != {(d_sae,)}"
                assert active_mask.shape == (d_sae,), msg
                self.toks_since_active += bsz
                self.toks_since_active[active_mask] = 0
                dead_mask = self.toks_since_active >= self.cfg.dead_threshold_tokens
        else:
            dead_mask = None

        # Sample prefix cuts
        prefixes = sample_prefixes(d_sae, self.cfg.n_prefixes)

        x_hats = sae.decode(enc.f_x, prefixes=prefixes)
        sae_out = modeling.SparseAutoencoder.Output(
            h_x=enc.h_x, f_x=enc.f_x, x_hats=x_hats
        )

        # Calculate losses
        mse_loss = mean_squared_err(
            x_hats,
            einops.repeat(
                x, "b d_model -> b prefixes d_model", prefixes=self.cfg.n_prefixes
            ),
        ).mean()

        aux_loss = sae.cfg.activation.aux.loss(
            sae=sae, x=x, out=sae_out, dead_mask=dead_mask
        )
        n_dead = dead_mask.sum() if dead_mask is not None else torch.tensor(0)

        # Calculate sparsity metrics on full encoding
        return (
            MatryoshkaLoss(
                mse=mse_loss,
                sparsity=sae.activation.cfg.sparsity.loss(enc.f_x),
                l0=(enc.f_x != 0).float().sum(axis=1).mean(axis=0),
                l1=enc.f_x.abs().sum(axis=1).mean(axis=0),
                aux=aux_loss,
                n_dead=n_dead,
            ),
            sae_out,
        )


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
    if isinstance(cfg, Matryoshka):
        return MatryoshkaObjective(cfg)
    else:
        tp.assert_never(cfg)


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
