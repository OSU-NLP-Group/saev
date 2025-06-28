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
@dataclasses.dataclass(frozen=True, slots=True)
class Relu:
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

    @property
    def d_sae(self) -> int:
        return self.d_vit * self.exp_factor


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class JumpRelu:
    """Implementation of the JumpReLU activation function for SAEs. Not implemented."""

    pass


SparseAutoencoderConfig = Relu | JumpRelu


@jaxtyped(typechecker=beartype.beartype)
class SparseAutoencoder(torch.nn.Module):
    """
    Sparse auto-encoder (SAE) using L1 sparsity penalty.
    """

    def __init__(self, cfg: Relu):
        super().__init__()

        self.cfg = cfg

        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_vit, cfg.d_sae))
        )
        self.b_enc = torch.nn.Parameter(torch.zeros(cfg.d_sae))

        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_sae, cfg.d_vit))
        )
        self.b_dec = torch.nn.Parameter(torch.zeros(cfg.d_vit))

        self.activation = get_activation(cfg)

        self.logger = logging.getLogger(f"sae(seed={cfg.seed})")

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> tuple[Float[Tensor, "batch d_model"], Float[Tensor, "batch d_sae"]]:
        """
        Given x, calculates the reconstructed x_hat and the intermediate activations f_x.

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
        x_hat = self.decode(f_x)

        return x_hat, f_x

    def decode(
        self, f_x: Float[Tensor, "batch d_sae"]
    ) -> Float[Tensor, "batch d_model"]:
        x_hat = (
            einops.einsum(f_x, self.W_dec, "... d_sae, d_sae d_vit -> ... d_vit")
            + self.b_dec
        )
        return x_hat

    @torch.no_grad()
    def init_b_dec(self, vit_acts: Float[Tensor, "n d_vit"]):
        if self.cfg.n_reinit_samples <= 0:
            self.logger.info("Skipping init_b_dec.")
            return
        previous_b_dec = self.b_dec.clone().cpu()
        vit_acts = vit_acts[: self.cfg.n_reinit_samples]
        assert len(vit_acts) == self.cfg.n_reinit_samples
        mean = vit_acts.mean(axis=0)
        previous_distances = torch.norm(vit_acts - previous_b_dec, dim=-1)
        distances = torch.norm(vit_acts - mean, dim=-1)
        self.logger.info(
            "Prev dist: %.3f; new dist: %.3f",
            previous_distances.median(axis=0).values.mean().item(),
            distances.median(axis=0).values.mean().item(),
        )
        self.b_dec.data = mean.to(self.b_dec.dtype).to(self.b_dec.device)

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


@beartype.beartype
def get_activation(cfg: SparseAutoencoderConfig) -> torch.nn.Module:
    if isinstance(cfg, Relu):
        return torch.nn.ReLU()
    elif isinstance(cfg, JumpRelu):
        raise NotImplementedError()
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
    header = {
        "schema": 1,
        "cfg": dataclasses.asdict(sae.cfg),
        "cls": sae.cfg.__class__.__name__,
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
        # Original, pre-schema stuff.
        for keyword in ("sparsity_coeff", "ghost_grads"):
            header.pop(keyword)
        cfg = Relu(**header)
    elif header["schema"] == 1:
        cls = globals()[header["cls"]]  # default for v0
        cfg = cls(**header["cfg"])
    else:
        raise ValueError(f"Unknown schema version: {header['schema']}")

    model = SparseAutoencoder(cfg)
    model.load_state_dict(torch.load(buffer, weights_only=True, map_location=device))
    return model
