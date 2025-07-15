"""
Neural network architectures for sparse autoencoders.
"""

import dataclasses
import io
import json
import logging
import os
import pathlib
import subprocess
import typing

import beartype
import einops
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from .. import __version__, config


@jaxtyped(typechecker=beartype.beartype)
class SparseAutoencoder(torch.nn.Module):
    """
    Sparse auto-encoder (SAE) using L1 sparsity penalty.
    """

    def __init__(self, cfg: config.SparseAutoencoder):
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


@jaxtyped(typechecker=beartype.beartype)
class MatryoshkaSparseAutoencoder(SparseAutoencoder):
    """
    Subclass of SparseAutoencoder for use with the Matryoshka objective function.
    Needed since the matryoshka objective requires access to the weights of the decoder in order to calculate the
    reconstructions from prefixes of the sparse encoding.

    Still uses L1 for sparsity penalty, though when using BatchTopK as activation (recommended), this is not relevant.
    """

    def __init__(self, cfg: config.SparseAutoencoder):
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

        block_indices = torch.torch.cat((torch.tensor([0]).to(self.b_dec.device), prefixes))
        block_bounds = list(zip(block_indices[:-1], block_indices[1:]))

        block_preds = [
            self.block_decode(f_x, block) for block in block_bounds
        ]

        prefix_preds = torch.cumsum(torch.stack(block_preds), dim=0)

        return prefix_preds, f_x

    def block_decode(
        self, f_x: Float[Tensor, "batch d_sae"], block: tuple[int]
    ) -> Float[Tensor, "batch d_model"]:
        """Decodes sparse encoding using only the given interval of indices.
        
        Arguments:
            f_x: Sparse encoding """
        
        # Can't use einsum here because the block lengths can change
        x_hat = torch.matmul(f_x[:, block[0]:block[1]], self.W_dec[block[0]:block[1]]) + self.b_dec
        
        #x_hat = (
        #    einops.einsum(f_x[block[0]:block[1]], self.W_dec[block[0]:block[1]], "... block, block d_vit -> ... d_vit")
        #    + self.b_dec[block[0]:block[1]]
        #)

        return x_hat

    @torch.no_grad()
    def sample_prefixes(
        self,
        sae_dim: int,
        n_prefixes: int,
        min_prefix_length: int = 1,
        pareto_power: float = 0.5,
        replacement: bool = False
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
        prefixes = torch.multinomial(probability_dist, num_samples=n_prefixes - 1, replacement=replacement)

        # Add n_latents as the final prefix
        prefixes = torch.cat((prefixes.detach().clone(), torch.tensor([sae_dim])))

        prefixes, _ = torch.sort(prefixes, descending=False)

        return prefixes


@jaxtyped(typechecker=beartype.beartype)
class TopK(torch.nn.Module):
    """
    Top-K activation function. For use as activation function of sparse encoder.
    """

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k

    def forward(self, x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
        """
        Apply top-k activation to the input tensor.
        """
        if self.k <= 0:
            raise ValueError("k must be a positive integer.")
        
        k_vals, k_inds = torch.topk(x, self.k, dim=-1, sorted=False)
        mask = torch.zeros_like(x).scatter_(dim=-1, index=k_inds, src=torch.ones_like(x))

        return torch.mul(mask, x)


@jaxtyped(typechecker=beartype.beartype)
class BatchTopK(torch.nn.Module):
    """
    Batch Top-K activation function. For use as activation function of sparse encoder.
    """

    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k

    def forward(self, x: Float[Tensor, "batch d_sae"]) -> Float[Tensor, "batch d_sae"]:
        """
        Apply top-k activation to the input tensor.
        """
        if self.k <= 0:
            raise ValueError("k must be a positive integer.")
        
        orig_shape = x.shape
        x = x.flatten()
        k_vals, k_inds = torch.topk(x, self.k, dim=-1, sorted=False)
        mask = torch.zeros_like(x).scatter_(dim=-1, index=k_inds, src=torch.ones_like(x))

        return torch.mul(mask, x).reshape(orig_shape)


@beartype.beartype
def get_activation(cfg: config.SparseAutoencoder) -> torch.nn.Module:
    if isinstance(cfg, config.Relu):
        return torch.nn.ReLU()
    elif isinstance(cfg, config.JumpRelu):
        raise NotImplementedError()
    elif isinstance(cfg, config.TopK):
        return TopK(k=cfg.top_k)
    elif isinstance(cfg, config.BatchTopK):
        return BatchTopK(k=cfg.top_k)
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
        "commit": current_git_commit() or "unknown",
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
        cfg = config.Relu(**header)
    elif header["schema"] == 1:
        cls = getattr(config, header["cls"])  # default for v0
        cfg = cls(**header["cfg"])
    else:
        raise ValueError(f"Unknown schema version: {header['schema']}")

    model = SparseAutoencoder(cfg)
    model.load_state_dict(torch.load(buffer, weights_only=True, map_location=device))
    return model


@beartype.beartype
def current_git_commit() -> str | None:
    """
    Best-effort short SHA of the repo containing *this* file.

    Returns `None` when
    * `git` executable is missing,
    * we’re not inside a git repo (e.g. installed wheel),
    * or any git call errors out.
    """
    try:
        # Walk up until we either hit a .git dir or the FS root
        here = pathlib.Path(__file__).resolve()
        for parent in (here, *here.parents):
            if (parent / ".git").exists():
                break
        else:  # no .git found
            return None

        result = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
