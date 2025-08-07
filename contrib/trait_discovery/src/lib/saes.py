"""Scorer wrapper for pre-trained Sparse Autoencoders."""

import beartype
import torch
from jaxtyping import Float, jaxtyped
from lib.baselines import Scorer
from torch import Tensor

import saev.nn.modeling


@jaxtyped(typechecker=beartype.beartype)
class SparseAutoencoderScorer(Scorer):
    """Wraps a pre-trained SAE to provide the Scorer interface."""

    def __init__(self, ckpt_fpath: str):
        super().__init__()
        self.ckpt_fpath = ckpt_fpath
        self.sae = saev.nn.modeling.load(ckpt_fpath, device="cpu")
        self._trained = True

    @property
    def n_prototypes(self) -> int:
        return self.sae.cfg.d_sae

    @property
    def kwargs(self) -> dict[str, object]:
        """Not applicable for pre-trained SAEs."""
        return {"ckpt_fpath": self.ckpt_fpath}

    def train(self, dataloader):
        """Pre-trained SAEs don't need training."""
        pass

    def forward(self, activations: Float[Tensor, "B D"]) -> Float[Tensor, "B K"]:
        """
        Score activations using SAE latent activations.

        Args:
            activations: ViT activations of shape (batch, d_vit)

        Returns:
            Latent activations of shape (batch, d_sae)
        """
        with torch.no_grad():
            latents = self.sae.encode(activations)
        return latents
