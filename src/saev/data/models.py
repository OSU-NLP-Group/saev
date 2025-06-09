import logging
import typing
from collections.abc import Callable

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import config, helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


@jaxtyped(typechecker=beartype.beartype)
class DinoV2(torch.nn.Module):
    def __init__(self, vit_ckpt: str):
        super().__init__()

        self.model = torch.hub.load("facebookresearch/dinov2", vit_ckpt)
        self.name = f"dinov2/{vit_ckpt}"

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
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        dct = self.model.forward_features(batch)

        features = torch.cat(
            (dct["x_norm_clstoken"][:, None, :], dct["x_norm_patchtokens"]), axis=1
        )
        return features


@jaxtyped(typechecker=beartype.beartype)
class Clip(torch.nn.Module):
    def __init__(self, vit_ckpt: str):
        super().__init__()

        import open_clip

        if vit_ckpt.startswith("hf-hub:"):
            clip, _ = open_clip.create_model_from_pretrained(
                vit_ckpt, cache_dir=helpers.get_cache_dir()
            )
        else:
            arch, ckpt = vit_ckpt.split("/")
            clip, _ = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )

        model = clip.visual
        model.proj = None
        model.output_tokens = True  # type: ignore
        self.model = model.eval()

        assert not isinstance(self.model, open_clip.timm_model.TimmModel)

        self.name = f"clip/{vit_ckpt}"

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.transformer.resblocks

    def get_patches(self, cfg: config.Activations) -> slice:
        return slice(None, None, None)

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        return self.model(batch)


@beartype.beartype
def make_vit(vit_family: str, vit_ckpt: str):
    if vit_family == "clip":
        return Clip(vit_ckpt)
    elif vit_family == "dinov2":
        return DinoV2(vit_ckpt)
    else:
        typing.assert_never(vit_family)


@beartype.beartype
def make_img_transform(vit_family: str, vit_ckpt: str) -> Callable:
    if vit_family == "clip":
        import open_clip

        if vit_ckpt.startswith("hf-hub:"):
            _, img_transform = open_clip.create_model_from_pretrained(
                vit_ckpt, cache_dir=helpers.get_cache_dir()
            )
        else:
            arch, ckpt = vit_ckpt.split("/")
            _, img_transform = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )
        return img_transform

    elif vit_family == "dinov2":
        from torchvision.transforms import v2

        return v2.Compose([
            v2.Resize(size=(256, 256)),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])

    else:
        typing.assert_never(vit_family)
