from collections.abc import Callable

import beartype
import torch
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor

from . import models


@jaxtyped(typechecker=beartype.beartype)
class Vit(models.VisionTransformer, torch.nn.Module):
    family: str = "clip"

    def __init__(self, ckpt: str):
        super().__init__()

        import open_clip

        from .. import helpers

        if ckpt.startswith("hf-hub:"):
            clip, _ = open_clip.create_model_from_pretrained(
                ckpt, cache_dir=helpers.get_cache_dir()
            )
            _, ckpt = ckpt.split("hf-hub:")
        else:
            arch, ckpt = ckpt.split("/")
            clip, _ = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )
        self._ckpt = ckpt
        model = clip.visual
        model.proj = None
        model.output_tokens = True  # type: ignore
        self.model = model.eval()

        assert not isinstance(self.model, open_clip.timm_model.TimmModel)

    @property
    def ckpt(self) -> str:
        return self._ckpt

    @property
    def patch_size(self) -> int:
        """Get patch size for CLIP models."""
        # Standard CLIP models all use 16x16 patches
        # The tiny test model uses 2x2 but that's not a real model
        return 16

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.transformer.resblocks

    def get_patches(self, n_patches_per_img: int) -> slice:
        return slice(None, None, None)

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        return self.model(batch)

    @staticmethod
    def make_transforms(
        ckpt: str, n_patches_per_img: int
    ) -> tuple[Callable, Callable | None]:
        """Create transforms for preprocessing: (img_transform, sample_transform | None)."""
        import open_clip

        from .. import helpers

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

    @staticmethod
    def make_resize(
        ckpt: str,
        n_patches_per_img: int = -1,
        *,
        scale: float = 2.0,
        resample: Image.Resampling = Image.LANCZOS,
    ) -> Callable[[Image.Image], Image.Image]:
        def resize(img: Image.Image) -> Image.Image:
            resize_size_px = (int(256 * scale), int(256 * scale))
            crop_size_px = (int(224 * scale), int(224 * scale))

            resize_w_px, resize_h_px = resize_size_px
            crop_w_px, crop_h_px = crop_size_px
            crop_coords_px = (
                (resize_w_px - crop_w_px) // 2,
                (resize_h_px - crop_h_px) // 2,
                (resize_w_px + crop_w_px) // 2,
                (resize_h_px + crop_h_px) // 2,
            )
            return img.resize(resize_size_px, resample=resample).crop(crop_coords_px)

        return resize
