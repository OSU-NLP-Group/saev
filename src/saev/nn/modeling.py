"""
Neural network architectures for sparse autoencoders.
"""

import dataclasses
import io
import json
import logging
import os
import typing

import beartype
import einops
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from .. import __version__, helpers


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Relu:
    """Vanilla ReLU"""

    pass


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TopK:
    top_k: int = 32
    """How many values are allowed to be non-zero."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class BatchTopK:
    top_k: int = 32
    """How many values are allowed to be non-zero per sample in the batch."""


ActivationConfig = Relu | TopK | BatchTopK


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class AuxiliaryConfig:
    top_k: int = 512
    """How many dead latents to consider for auxiliary loss."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SparseAutoencoderConfig:
    d_vit: int = 1024
    exp_factor: int = 16
    """Expansion factor for SAE."""
    n_reinit_samples: int = 1024 * 16 * 32
    """Number of samples to use for SAE re-init. Anthropic proposes initializing b_dec to the geometric median of the dataset here: https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-bias. We use the regular mean."""
    remove_parallel_grads: bool = True
    """Whether to remove gradients parallel to W_dec columns (which will be ignored because we force the columns to have unit norm). See https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-optimization for the original discussion from Anthropic."""
    normalize_w_dec: bool = True
    """Whether to make sure W_dec has unit norm columns. See https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder for original citation."""
    seed: int = 0
    """Random seed."""
    activation: ActivationConfig = Relu()

    @property
    def d_sae(self) -> int:
        return self.d_vit * self.exp_factor


@jaxtyped(typechecker=beartype.beartype)
class SparseAutoencoder(torch.nn.Module):
    """
    Sparse auto-encoder (SAE) using L1 sparsity penalty.
    """

    def __init__(self, cfg: SparseAutoencoderConfig):
        super().__init__()

        self.cfg = cfg
        self.logger = logging.getLogger(f"sae(seed={cfg.seed})")

        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_vit, cfg.d_sae))
        )
        self.b_enc = torch.nn.Parameter(torch.zeros(cfg.d_sae))

        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_sae, cfg.d_vit))
        )
        self.b_dec = torch.nn.Parameter(torch.zeros(cfg.d_vit))

        self.normalize_w_dec()

        self.activation = get_activation(cfg.activation)

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> tuple[Float[Tensor, "batch d_model"], Float[Tensor, "batch d_sae"]]:
        """
        Given x, calculates the reconstructed x_hat and the intermediate activations f_x.

        Arguments:
            x: a batch of ViT activations.
        """
        f_x = self.encode(x)
        x_hat = self.decode(f_x)

        return x_hat, f_x

    def encode(self, x: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch d_sae"]:
        h_pre = (
            einops.einsum(x, self.W_enc, "... d_vit, d_vit d_sae -> ... d_sae")
            + self.b_enc
        )
        f_x = self.activation(h_pre)
        return f_x

    def decode(
        self, f_x: Float[Tensor, "batch d_sae"]
    ) -> Float[Tensor, "batch d_model"]:
        x_hat = (
            einops.einsum(f_x, self.W_dec, "... d_sae, d_sae d_vit -> ... d_vit")
            + self.b_dec
        )
        return x_hat

    @torch.no_grad()
    def normalize_w_dec(self):
        """
        Set W_dec to unit-norm columns.
        """
        if self.cfg.normalize_w_dec:
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_parallel_grads(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_vit) shape
        """
        if not self.cfg.remove_parallel_grads:
            return

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_vit, d_sae d_vit -> d_sae",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_vit -> d_sae d_vit",
        )


@jaxtyped(typechecker=beartype.beartype)
class MatryoshkaSparseAutoencoder(SparseAutoencoder):
    """
    Subclass of SparseAutoencoder for use with the Matryoshka objective function.
    Needed since the matryoshka objective requires access to the weights of the decoder in order to calculate the
    reconstructions from prefixes of the sparse encoding.

    Still uses L1 for sparsity penalty, though when using BatchTopK as activation (recommended), this is not relevant.
    """

    def __init__(self, cfg: SparseAutoencoderConfig):
        super().__init__(cfg)

    def matryoshka_forward(
        self, x: Float[Tensor, "batch d_model"], n_prefixes: int
    ) -> tuple[Float[Tensor, "batch d_model"], Float[Tensor, "batch d_sae"]]:
        """
        Given x, calculates the reconstructed x_hat from the prefixes of encoded intermediate activations f_x.

        Arguments:
            x: a batch of ViT activations.
        """

        # Remove encoder bias as per Anthropic
        h_pre = (
            einops.einsum(
                x - self.b_dec, self.W_enc, "... d_vit, d_vit d_sae -> ... d_sae"
            )
            + self.b_enc
        )
        f_x = self.activation(h_pre)

        prefixes = self.sample_prefixes(len(f_x), n_prefixes).to(self.b_dec.device)

        block_indices = torch.torch.cat((
            torch.tensor([0]).to(self.b_dec.device),
            prefixes,
        ))
        block_bounds = list(zip(block_indices[:-1], block_indices[1:]))

        block_preds = [self.block_decode(f_x, block) for block in block_bounds]

        prefix_preds = torch.cumsum(torch.stack(block_preds), dim=0)

        return prefix_preds, f_x

    def block_decode(
        self, f_x: Float[Tensor, "batch d_sae"], block: tuple[int]
    ) -> Float[Tensor, "batch d_model"]:
        """Decodes sparse encoding using only the given interval of indices.

        Arguments:
            f_x: Sparse encoding"""

        # Can't use einsum here because the block lengths can change
        x_hat = (
            torch.matmul(f_x[:, block[0] : block[1]], self.W_dec[block[0] : block[1]])
            + self.b_dec
        )

        # x_hat = (
        #    einops.einsum(f_x[block[0]:block[1]], self.W_dec[block[0]:block[1]], "... block, block d_vit -> ... d_vit")
        #    + self.b_dec[block[0]:block[1]]
        # )

        return x_hat

    @torch.no_grad()
    def sample_prefixes(
        self,
        sae_dim: int,
        n_prefixes: int,
        min_prefix_length: int = 1,
        pareto_power: float = 0.5,
        replacement: bool = False,
    ) -> torch.Tensor:
        """
        Samples prefix lengths using a Pareto distribution. Derived from "Learning Multi-Level Features with
        Matryoshka Sparse Autoencoders" (https://doi.org/10.48550/arXiv.2503.17547)

        Args:
            sae_dim: Total number of latent dimensions
            n_prefixes: Number of prefixes to sample
            min_prefix_length: Minimum length of any prefix
            pareto_power: Power parameter for Pareto distribution (lower = more uniform)

        Returns:
            torch.Tensor: Sorted prefix lengths
        """
        if n_prefixes <= 1:
            return torch.tensor([sae_dim])

        # Calculate probability distribution favoring shorter prefixes
        lengths = torch.arange(1, sae_dim)
        pareto_cdf = 1 - ((min_prefix_length / lengths.float()) ** pareto_power)
        pareto_pdf = torch.cat([pareto_cdf[:1], pareto_cdf[1:] - pareto_cdf[:-1]])
        probability_dist = pareto_pdf / pareto_pdf.sum()

        # Sample and sort prefix lengths
        prefixes = torch.multinomial(
            probability_dist, num_samples=n_prefixes - 1, replacement=replacement
        )

        # Add n_latents as the final prefix
        prefixes = torch.cat((prefixes.detach().clone(), torch.tensor([sae_dim])))

        prefixes, _ = torch.sort(prefixes, descending=False)

        return prefixes


@jaxtyped(typechecker=beartype.beartype)
class TopKActivation(torch.nn.Module):
    """
    Top-K activation function. For use as activation function of sparse encoder.
    """

    def __init__(self, cfg: TopK = TopK()):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.top_k

    def forward(self, x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
        """
        Apply top-k activation to the input tensor.
        """
        if self.k <= 0:
            raise ValueError("k must be a positive integer.")

        k_vals, k_inds = torch.topk(x, self.k, dim=-1, sorted=False)
        mask = torch.zeros_like(x).scatter_(
            dim=-1, index=k_inds, src=torch.ones_like(x)
        )

        return torch.mul(mask, x)


@jaxtyped(typechecker=beartype.beartype)
class BatchTopKActivation(torch.nn.Module):
    """
    Batch Top-K activation function. For use as activation function of sparse encoder.
    Applies top-k selection per sample in the batch.
    """

    def __init__(self, cfg: BatchTopK = BatchTopK()):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.top_k

    def forward(self, x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
        """
        Apply top-k activation to each sample in the batch.
        """
        if self.k <= 0:
            raise ValueError("k must be a positive integer.")

        # Handle case where k exceeds number of elements per sample
        k = min(self.k, x.shape[-1])

        # Apply top-k per sample (along the last dimension)
        k_vals, k_inds = torch.topk(x, k, dim=-1, sorted=False)
        mask = torch.zeros_like(x).scatter_(
            dim=-1, index=k_inds, src=torch.ones_like(x)
        )

        return torch.mul(mask, x)


class AuxiliaryLossActivation(torch.nn.Module):
    """
    Auxiliary loss activation function. Used to take the top-k dead latents before calculating the auxiliary loss.
    """

    def __init__(self, cfg: AuxiliaryConfig = AuxiliaryConfig()):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        f_x: Float[Tensor, "batch d_sae"],
        dead_latents: Float[Tensor, "batch d_sae"],
    ) -> Float[Tensor, "batch d_sae"]:
        """
        Apply auxiliary loss activation (top-k of dead latents) to the input tensor.
        """

        # First, mask out all but dead latents
        f_x = f_x * dead_latents

        masked_dead_top_k = torch.zeros_like(f_x)

        # Now, populate top k of the dead latents
        if self.cfg.top_k > 0 and dead_latents.sum() > 0:
            # First, mask out dead latents
            masked_dead_latents = f_x * dead_latents

            # Find top k of dead latents
            k_vals, k_inds = torch.topk(
                masked_dead_latents, self.cfg.top_k, dim=1, sorted=False
            )
            top_k_mask = torch.zeros_like(masked_dead_latents).scatter_(
                dim=-1, index=k_inds, src=torch.ones_like(masked_dead_latents)
            )

            # Mask out all but top k dead latents
            masked_dead_top_k = torch.mul(top_k_mask, f_x)

        return masked_dead_top_k


@beartype.beartype
def get_activation(cfg: ActivationConfig) -> torch.nn.Module:
    if isinstance(cfg, Relu):
        return torch.nn.ReLU()
    elif isinstance(cfg, TopK):
        return TopKActivation(cfg)
    elif isinstance(cfg, BatchTopK):
        return BatchTopKActivation(cfg)
    else:
        typing.assert_never(cfg)


@beartype.beartype
def dump(fpath: str, sae: SparseAutoencoder):
    """
    Save an SAE checkpoint to disk along with configuration, using the [trick from equinox](https://docs.kidger.site/equinox/examples/serialisation).

    Arguments:
        fpath: filepath to save checkpoint to.
        sae: sparse autoencoder checkpoint to save.
    """
    # Custom serialization to handle activation object
    cfg_dict = dataclasses.asdict(sae.cfg)
    # Replace activation dict with custom format
    activation = sae.cfg.activation
    cfg_dict["activation"] = {
        "cls": activation.__class__.__name__,
        "params": dataclasses.asdict(activation),
    }

    header = {
        "schema": 2,
        "cfg": cfg_dict,
        "commit": helpers.current_git_commit() or "unknown",
        "lib": __version__,
    }

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as fd:
        header_str = json.dumps(header)
        fd.write((header_str + "\n").encode("utf-8"))
        torch.save(sae.state_dict(), fd)


@beartype.beartype
def load(fpath: str, *, device="cpu") -> SparseAutoencoder:
    """
    Loads a sparse autoencoder from disk.
    """
    with open(fpath, "rb") as fd:
        header = json.loads(fd.readline())
        buffer = io.BytesIO(fd.read())

    if "schema" not in header:
        # Original, pre-schema format: just raw config parameters
        # Remove old parameters that no longer exist
        for keyword in ("sparsity_coeff", "ghost_grads", "l1_coeff", "use_ghost_grads"):
            header.pop(keyword, None)
        # Legacy format - create SparseAutoencoderConfig with Relu activation
        cfg = SparseAutoencoderConfig(**header, activation=Relu())
    elif header["schema"] == 1:
        # Schema version 1: A cautionary tale of poor version management
        #
        # This schema version unfortunately has TWO incompatible formats because we made breaking changes without incrementing the schema version. This is exactly what schema versioning is supposed to prevent!
        #
        # Format 1A (original): cls field contains activation type ("Relu", "TopK", etc.)
        # Format 1B (later): cls field is "SparseAutoencoderConfig" and activation is a dict
        #
        # The complex logic below exists to handle both formats. This should have been avoided by incrementing to schema version 2 when we changed the format.
        #
        # Apologies from Sam for this mess - proper schema versioning discipline would have prevented this confusing situation. Every breaking change should increment the version number!

        cls_name = header.get("cls", "SparseAutoencoderConfig")
        cfg_dict = header["cfg"]

        if cls_name in ["Relu", "TopK", "BatchTopK"]:
            # Format 1A: Old format where cls indicates the activation type
            activation_cls = globals()[cls_name]
            if cls_name in ["TopK", "BatchTopK"]:
                activation = activation_cls(top_k=cfg_dict.get("top_k", 32))
            else:
                activation = activation_cls()
            cfg = SparseAutoencoderConfig(**cfg_dict, activation=activation)
        else:
            # Format 1B: Newer format with activation as dict
            if "activation" in cfg_dict:
                activation_info = cfg_dict["activation"]
                activation_cls = globals()[activation_info["cls"]]
                activation = activation_cls(**activation_info["params"])
                cfg_dict["activation"] = activation
            cfg = SparseAutoencoderConfig(**cfg_dict)
    elif header["schema"] == 2:
        # Schema version 2: cleaner format with activation serialization
        cfg_dict = header["cfg"]
        activation_info = cfg_dict["activation"]
        activation_cls = globals()[activation_info["cls"]]
        activation = activation_cls(**activation_info["params"])
        cfg_dict["activation"] = activation
        cfg = SparseAutoencoderConfig(**cfg_dict)
    else:
        raise ValueError(f"Unknown schema version: {header['schema']}")

    model = SparseAutoencoder(cfg)
    model.load_state_dict(torch.load(buffer, weights_only=True, map_location=device))
    return model
