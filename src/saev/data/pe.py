"""Perception Encoder (PE) models from Meta (Bolya et al., 2025).

PE-Core: CLIP-style model for language alignment.
PE-Spatial: Dense prediction model distilled from SAM 2.1.

Both are available via timm.
"""

import logging
from collections.abc import Callable

import beartype
import timm
import timm.data
import torch
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor

from . import models


@jaxtyped(typechecker=beartype.beartype)
class _Base(torch.nn.Module, models.Transformer):
    """Base class for PE models with shared functionality."""

    family: str  # Set by subclass

    def __init__(self, ckpt: str):
        super().__init__()
        self._ckpt = ckpt
        self.logger = logging.getLogger(f"{self.family}/{ckpt}")

        # Load model without classifier head, outputting patch features
        self.model = timm.create_model(ckpt, pretrained=True, num_classes=0)
        self.model.eval()

        # Get data config for transforms
        self._data_config = timm.data.resolve_model_data_config(self.model)

    @property
    def ckpt(self) -> str:
        return self._ckpt

    @property
    def patch_size(self) -> int:
        """Get patch size from model's patch embedding layer."""
        patch_embed = self.model.patch_embed
        if hasattr(patch_embed, "patch_size"):
            ps = patch_embed.patch_size
            if isinstance(ps, tuple):
                assert ps[0] == ps[1]
                return ps[0]
            return ps
        if hasattr(patch_embed, "proj") and isinstance(
            patch_embed.proj, torch.nn.Conv2d
        ):
            w, h = patch_embed.proj.kernel_size
            assert w == h
            return w
        raise ValueError(f"Cannot determine patch size for {self._ckpt}")

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.blocks

    def get_token_i(self, content_tokens_per_example: int) -> slice:
        # PE models have CLS token at position 0, then patch tokens
        # Return all tokens (CLS + patches)
        return slice(None, None, None)

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        # timm ViT with num_classes=0 returns (batch, n_tokens, dim) when using forward_features
        # But forward() with num_classes=0 may pool. Use forward_features instead.
        features = self.model.forward_features(batch)
        return features

    @staticmethod
    def _make_transforms_impl(
        ckpt: str, n_patches_per_img: int
    ) -> tuple[Callable, Callable | None]:
        """Create transforms using timm's data config."""
        model = timm.create_model(ckpt, pretrained=False)
        data_config = timm.data.resolve_model_data_config(model)
        img_transform = timm.data.create_transform(**data_config, is_training=False)
        del model
        return img_transform, None

    @staticmethod
    def _make_resize_impl(
        ckpt: str,
        n_patches_per_img: int,
        *,
        scale: float,
        resample: Image.Resampling,
    ) -> Callable[[Image.Image], Image.Image]:
        """Create resize transform for visualization."""
        model = timm.create_model(ckpt, pretrained=False)
        data_config = timm.data.resolve_model_data_config(model)
        input_size = data_config["input_size"]
        _, h, w = input_size
        assert h == w, f"Expected square input, got {input_size}"
        del model

        def resize(img: Image.Image) -> Image.Image:
            size = int(h * scale)
            return img.resize((size, size), resample=resample)

        return resize


@jaxtyped(typechecker=beartype.beartype)
class Core(_Base):
    """PE-Core: CLIP-style model for language alignment.

    Available checkpoints:
    - vit_pe_core_large_patch14_336.fb (L/14, 336px)
    - vit_pe_core_base_patch16_224.fb (B/16, 224px)
    """

    family: str = "pe-core"

    @staticmethod
    def make_transforms(
        ckpt: str, n_patches_per_img: int
    ) -> tuple[Callable, Callable | None]:
        return _Base._make_transforms_impl(ckpt, n_patches_per_img)

    @staticmethod
    def make_resize(
        ckpt: str,
        n_patches_per_img: int = -1,
        *,
        scale: float = 1.0,
        resample: Image.Resampling = Image.LANCZOS,
    ) -> Callable[[Image.Image], Image.Image]:
        return _Base._make_resize_impl(
            ckpt, n_patches_per_img, scale=scale, resample=resample
        )


@jaxtyped(typechecker=beartype.beartype)
class Spatial(_Base):
    """PE-Spatial: Dense prediction model distilled from SAM 2.1.

    Available checkpoints:
    - vit_pe_spatial_large_patch14_448.fb (L/14, 448px)
    - vit_pe_spatial_base_patch16_512.fb (B/16, 512px)
    """

    family: str = "pe-spatial"

    @staticmethod
    def make_transforms(
        ckpt: str, n_patches_per_img: int
    ) -> tuple[Callable, Callable | None]:
        return _Base._make_transforms_impl(ckpt, n_patches_per_img)

    @staticmethod
    def make_resize(
        ckpt: str,
        n_patches_per_img: int = -1,
        *,
        scale: float = 1.0,
        resample: Image.Resampling = Image.LANCZOS,
    ) -> Callable[[Image.Image], Image.Image]:
        return _Base._make_resize_impl(
            ckpt, n_patches_per_img, scale=scale, resample=resample
        )
