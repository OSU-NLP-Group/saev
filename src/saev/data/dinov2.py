from collections.abc import Callable

import beartype
import torch
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

from . import models


@jaxtyped(typechecker=beartype.beartype)
class Vit(torch.nn.Module, models.Transformer):
    family: str = "dinov2"
    patch_size: int = 14

    @staticmethod
    def make_transforms(
        ckpt: str, n_patches_per_img: int
    ) -> tuple[Callable, Callable | None]:
        img_transform = v2.Compose([
            v2.Resize(size=(256, 256)),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])
        return img_transform, None

    def __init__(self, ckpt: str):
        super().__init__()
        self._ckpt = ckpt
        self.model = torch.hub.load("facebookresearch/dinov2", ckpt)

    @property
    def ckpt(self) -> str:
        return self._ckpt

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.blocks

    def get_patches(self, n_patches_per_img: int) -> slice:
        n_reg = self.model.num_register_tokens
        patches = torch.cat((
            torch.tensor([0]),  # CLS token
            torch.arange(n_reg + 1, n_reg + 1 + n_patches_per_img),  # patches
        ))
        return patches

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"], **kwargs
    ) -> Float[Tensor, "batch patches dim"]:
        dct = self.model.forward_features(batch)

        features = torch.cat(
            (dct["x_norm_clstoken"][:, None, :], dct["x_norm_patchtokens"]), axis=1
        )
        return features

    @staticmethod
    def make_resize(
        ckpt: str,
        n_patches_per_img: int = -1,
        *,
        scale: float = 1.0,
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
