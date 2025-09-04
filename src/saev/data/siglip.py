from collections.abc import Callable

import beartype
import open_clip
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from .. import helpers
from . import models


@jaxtyped(typechecker=beartype.beartype)
class Vit(torch.nn.Module, models.VisionTransformer):
    family = "siglip"

    @staticmethod
    def make_transforms(ckpt: str) -> tuple[Callable, Callable | None]:
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
        self.ckpt = ckpt

        model = clip.visual
        model.proj = None
        model.output_tokens = True  # type: ignore
        self.model = model

        assert isinstance(self.model, open_clip.timm_model.TimmModel)

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.trunk.blocks

    def get_patches(self, n_patches_per_img: int) -> slice:
        return slice(None, None, None)

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        result = self.model(batch)
        return result
