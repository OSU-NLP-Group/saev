"""Fake CLIP model for testing with tiny-open-clip-model.

This module provides a test-only vision transformer that works with
the tiny-open-clip-model from HuggingFace, which uses 8x8 images
and 2x2 patches instead of the standard 224x224 images with 16x16 patches.
"""

from collections.abc import Callable

import beartype
import open_clip
import torch
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor

from .. import helpers
from . import models


@jaxtyped(typechecker=beartype.beartype)
class Vit(models.VisionTransformer, torch.nn.Module):
    family: str = "fake-clip"

    def __init__(self, ckpt: str):
        super().__init__()

        # Only support the tiny test model
        assert ckpt == "hf-hub:hf-internal-testing/tiny-open-clip-model", (
            f"FakeClip only supports tiny-open-clip-model, got {ckpt}"
        )

        clip, _ = open_clip.create_model_from_pretrained(
            ckpt, cache_dir=helpers.get_cache_dir()
        )
        self._ckpt = ckpt
        model = clip.visual
        model.proj = None
        model.output_tokens = True  # type: ignore
        self.model = model.eval()

    @property
    def ckpt(self) -> str:
        return self._ckpt

    @property
    def patch_size(self) -> int:
        """Tiny model uses 2x2 patches."""
        return 2

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.transformer.resblocks

    @staticmethod
    def make_transforms(
        ckpt: str, n_patches_per_img: int
    ) -> tuple[Callable, Callable | None]:
        """Create transforms for preprocessing: (img_transform, sample_transform | None)."""
        _, img_transform = open_clip.create_model_from_pretrained(
            ckpt, cache_dir=helpers.get_cache_dir()
        )
        return img_transform, None

    @staticmethod
    def make_resize(
        ckpt: str,
        n_patches_per_img: int = -1,
        *,
        scale: float = 1.0,
        resample: Image.Resampling = Image.LANCZOS,
    ) -> Callable[[Image.Image], Image.Image]:
        """Create resize transform for tiny model (8x8 images)."""

        def resize(img: Image.Image) -> Image.Image:
            # Tiny model uses 8x8 images
            size_px = (int(8 * scale), int(8 * scale))
            return img.resize(size_px, resample=resample)

        return resize

    def get_token_i(self, content_tokens_per_example: int) -> slice:
        return slice(None, None, None)

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        return self.model(batch)
