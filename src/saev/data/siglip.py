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
class Vit(torch.nn.Module, models.VisionTransformer):
    family = "siglip"
    patch_size = 16

    @staticmethod
    def make_transforms(
        ckpt: str, n_patches_per_img: int
    ) -> tuple[Callable, Callable | None]:
        """Create transforms for preprocessing: (img_transform, sample_transform | None)."""
        if ckpt.startswith("hf-hub:"):
            _, img_transform = open_clip.create_model_from_pretrained(
                ckpt, cache_dir=helpers.get_cache_dir()
            )
        else:
            arch, ckpt = ckpt.split("/")
            _, img_transform = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )
        return img_transform, None

    def __init__(self, ckpt: str):
        super().__init__()

        if ckpt.startswith("hf-hub:"):
            clip, _ = open_clip.create_model_from_pretrained(
                ckpt, cache_dir=helpers.get_cache_dir()
            )
        else:
            arch, ckpt = ckpt.split("/")
            clip, _ = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )
        self._ckpt = ckpt

        model = clip.visual
        model.proj = None
        model.output_tokens = True  # type: ignore
        self.model = model

        assert isinstance(self.model, open_clip.timm_model.TimmModel)

    @property
    def ckpt(self) -> str:
        return self._ckpt

    @staticmethod
    def make_resize(
        ckpt: str,
        n_patches_per_img: int = -1,
        *,
        scale: float = 1.0,
        resample: Image.Resampling = Image.LANCZOS,
    ) -> Callable[[Image.Image], Image.Image]:
        """Create resize transform for visualization. Use resample=Image.NEAREST for segmentation masks."""
        from PIL import Image

        def resize(img: Image.Image) -> Image.Image:
            # SigLIP typically uses 224x224 or 384x384 images
            # We'll assume 224x224 for simplicity
            resize_size_px = (int(224 * scale), int(224 * scale))
            return img.resize(resize_size_px, resample=resample)

        return resize

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.trunk.blocks

    def get_patches(self, n_patches_per_img: int) -> slice:
        return slice(None, None, None)

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        result = self.model(batch)
        return result
