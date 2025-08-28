import dataclasses
import logging
import pathlib
import typing as tp
from collections.abc import Callable

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from saev import helpers

from . import dinov3

logger = logging.getLogger(__name__)


@tp.runtime_checkable
class VisionTransformer(tp.Protocol):
    """Protocol defining the interface for all Vision Transformer models."""

    name: str

    def get_residuals(self) -> list[torch.nn.Module]:
        """Return the list of residual blocks/layers for hook registration."""

    def get_patches(self, n_patches_per_img: int) -> slice | torch.Tensor:
        """Return indices for selecting relevant patches from activations."""

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        """Run forward pass on batch of images."""


@jaxtyped(typechecker=beartype.beartype)
class DinoV3(torch.nn.Module):
    def __init__(self, vit_ckpt: str):
        super().__init__()
        name = self._parse_name(vit_ckpt)
        self.name = f"dinov3/{name}"
        self.model = dinov3.load(name, vit_ckpt)

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.blocks

    def get_patches(self, n_patches_per_img: int) -> slice:
        n_reg = self.model.cfg.n_storage_tokens
        patches = torch.cat((
            torch.tensor([0]),  # CLS token
            torch.arange(n_reg + 1, n_reg + 1 + n_patches_per_img),  # patches
        ))
        return patches

    @staticmethod
    def _parse_name(dinov3_ckpt: str) -> str:
        name_ds, sha = pathlib.Path(dinov3_ckpt).stem.split("-")
        *name, pretrain, ds = name_ds.split("_")
        assert pretrain == "pretrain"
        return "_".join(name)

    def forward(
        self, batch: Float[Tensor, "batch n kernel"], **kwargs
    ) -> Float[Tensor, "batch patches dim"]:
        grid = kwargs.pop("grid")
        if kwargs:
            logger.info("Unused kwargs: %s", kwargs)
        dct = self.model(batch, grid=grid)

        features = torch.cat((dct["cls"][:, None, :], dct["patches"]), axis=1)
        return features


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
        self, batch: Float[Tensor, "batch 3 width height"], **kwargs
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

    def get_patches(self, n_patches_per_img: int) -> slice:
        return slice(None, None, None)

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        return self.model(batch)


@jaxtyped(typechecker=beartype.beartype)
class Siglip(torch.nn.Module):
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
        self.model = model

        assert isinstance(self.model, open_clip.timm_model.TimmModel)

        self.name = f"siglip/{vit_ckpt}"

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.trunk.blocks

    def get_patches(self, n_patches_per_img: int) -> slice:
        return slice(None, None, None)

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        result = self.model(batch)
        return result


@beartype.beartype
def make_vit(vit_family: str, vit_ckpt: str) -> VisionTransformer:
    if vit_family == "clip":
        return Clip(vit_ckpt)
    if vit_family == "siglip":
        return Siglip(vit_ckpt)
    elif vit_family == "dinov2":
        return DinoV2(vit_ckpt)
    elif vit_family == "dinov3":
        return DinoV3(vit_ckpt)
    else:
        tp.assert_never(vit_family)


@beartype.beartype
def make_transforms(vit_family: str, vit_ckpt: str) -> tuple[Callable, Callable | None]:
    if vit_family == "clip" or vit_family == "siglip":
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
        return img_transform, None

    elif vit_family == "dinov2":
        from torchvision.transforms import v2

        img_transform = v2.Compose([
            v2.Resize(size=(256, 256)),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])
        return img_transform, None

    elif vit_family == "dinov3":
        from torchvision.transforms import v2

        from . import transforms

        img_transform = v2.Compose([
            transforms.FlexResize(patch_size=16, n_patches=640),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])
        sample_transform = transforms.Unfold(patch_size=16, n_patches=640)
        return img_transform, sample_transform
    else:
        tp.assert_never(vit_family)
