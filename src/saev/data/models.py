import abc
import logging
from collections.abc import Callable

import beartype
import torch
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor

logger = logging.getLogger(__name__)


@jaxtyped(typechecker=beartype.beartype)
class VisionTransformer(abc.ABC):
    """Protocol defining the interface for all Vision Transformer models."""

    @property
    @abc.abstractmethod
    def family(self) -> str: ...

    @property
    @abc.abstractmethod
    def ckpt(self) -> str: ...

    @property
    @abc.abstractmethod
    def patch_size(self) -> int:
        """Patch size in pixels (e.g., 14 or 16)."""

    @property
    def name(self) -> str:
        return f"{self.family}/{self.ckpt}"

    @staticmethod
    @abc.abstractmethod
    def make_transforms(
        ckpt: str, n_patches_per_img: int
    ) -> tuple[Callable, Callable | None]:
        """Create transforms for preprocessing: (img_transform, sample_transform | None)."""

    @staticmethod
    @abc.abstractmethod
    def make_resize(
        ckpt: str,
        n_patches_per_img: int,
        *,
        scale: float = 1.0,
        resample: Image.Resampling = Image.LANCZOS,
    ) -> Callable[[Image.Image], Image.Image]:
        """Create resize transform for visualization. Use resample=Image.NEAREST for segmentation masks."""

    @abc.abstractmethod
    def get_residuals(self) -> list[torch.nn.Module]:
        """Return the list of residual blocks/layers for hook registration."""

    @abc.abstractmethod
    def get_patches(self, n_patches_per_img: int) -> slice | torch.Tensor:
        """Return indices for selecting relevant patches from activations."""

    @abc.abstractmethod
    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        """Run forward pass on batch of images."""


_global_vit_registry: dict[str, type[VisionTransformer]] = {}


@beartype.beartype
def load_vit_cls(family: str) -> type[VisionTransformer]:
    """Load a ViT family class."""
    if family not in _global_vit_registry:
        raise ValueError(f"Family '{family}' not found.")

    return _global_vit_registry[family]


@beartype.beartype
def register_family(cls: type[VisionTransformer]):
    """Register a new ViT family class."""
    if cls.family in _global_vit_registry:
        logger.warning("Overwriting key '%s' in registry.", cls.family)
    _global_vit_registry[cls.family] = cls


def list_families() -> list[str]:
    """List all ViT family names."""
    return list(_global_vit_registry.keys())
