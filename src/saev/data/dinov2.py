from collections.abc import Callable

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torchvision.transforms import v2

from . import models


@jaxtyped(typechecker=beartype.beartype)
class Vit(torch.nn.Module, models.VisionTransformer):
    family: str = "dinov2"

    @staticmethod
    def make_transforms(ckpt: str) -> tuple[Callable, Callable | None]:
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
        self.ckpt = ckpt
        self.model = torch.hub.load("facebookresearch/dinov2", ckpt)

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
