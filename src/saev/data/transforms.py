# src/saev/data/transforms.py
import math
import typing as tp

import beartype
import einops
import torch
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import v2


@beartype.beartype
class FlexResize(v2.Transform):
    def __init__(
        self,
        patch_size: int,
        n_patches: int,
        resample: Image.Resampling | int = Image.LANCZOS,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.resample = resample

    def transform(self, inpt: tp.Any, params: dict[str, tp.Any]):
        if isinstance(inpt, Image.Image):
            return _resize_to_patch_grid(
                inpt, p=self.patch_size, n=self.n_patches, resample=self.resample
            )
        else:
            raise TypeError(type(inpt))


@beartype.beartype
def _resize_to_patch_grid(
    img: Image.Image,
    *,
    p: int,
    n: int,
    resample: Image.Resampling | int = Image.LANCZOS,
) -> Image.Image:
    """
    Resize image to (w, h) so that:
      - w % p == 0, h % p == 0
      - (h/p) * (w/p) == N
      - Minimizes change in aspect ratio.
    """
    if p <= 0 or n <= 0:
        raise ValueError("p and n must be positive integers")

    w0, h0 = img.size
    a0 = w0 / h0

    # Find the aspect ratio closest to a0
    best_c = 0
    best_dist = float("inf")
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i != 0:
            continue

        for d in (i, n // i):
            c, r = d, n // d
            aspect = c / r
            dist = abs(aspect - a0)

            if dist < best_dist:
                best_c = d
                best_dist = dist

    c = best_c
    r = n // c
    w, h = c * p, r * p
    return img.resize((w, h), resample=resample)


@beartype.beartype
class Patchify(nn.Module):
    def __init__(self, patch_size: int, n_patches: int, key: str = "image"):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.key = key

    def forward(self, sample: dict[str, object]) -> dict[str, object]:
        assert self.key in sample
        img = sample[self.key]
        c, h, w = img.shape
        p = self.patch_size
        assert (h % p == 0) and (w % p == 0), f"Got {h}x{w}, patch={p}"

        patches_nd = einops.rearrange(
            img, "c (hp p1) (wp p2) -> (hp wp) (c p1 p2)", p1=p, p2=p
        )
        n, d = patches_nd.shape
        assert n == self.n_patches, f"Expected n={self.n_patches}, got {n}"
        assert d == c * p * p, f"d mismatch: {d} != {c}*{p}*{p}"

        sample[self.key] = patches_nd.contiguous()
        sample["grid"] = torch.tensor([h // p, w // p], dtype=torch.int16)
        return sample


@jaxtyped(typechecker=beartype.beartype)
def unfolded_conv2d(
    x_bchw: Float[Tensor, "b c h w"], conv: nn.Conv2d
) -> Float[Tensor, "b n d"]:
    """
    Returns tokens shaped (B, L, D), where L = (H/k)*(W/k), D = conv.out_channels.
    Requires: stride == kernel_size, padding == 0, groups == 1, dilation == 1.
    """
    k = conv.kernel_size[0]

    assert conv.kernel_size == (k, k)
    assert conv.stride == (k, k)
    assert conv.padding == (0, 0)
    assert conv.groups == 1
    assert conv.dilation == (1, 1)

    *b, c, h, w = x_bchw.shape

    assert h % k == 0 and w % k == 0

    tokens_bnd = einops.rearrange(
        x_bchw, "b c (hp p1) (wp p2) -> b (hp wp) (c p1 p2)", p1=k, p2=k
    ).contiguous()
    w_dp = conv.weight.reshape(conv.out_channels, c * k * k)
    tokens_bnd = tokens_bnd @ w_dp.T
    if conv.bias is not None:
        tokens_bnd = tokens_bnd + conv.bias[None, None, :]
    return tokens_bnd


@jaxtyped(typechecker=beartype.beartype)
def conv2d_to_tokens(
    x_bchw: Float[Tensor, "b c h w"], conv: nn.Conv2d
) -> Float[Tensor, "b n d"]:
    """Conv2d then flatten spatial to L, return (B, L, D)."""
    y_bdhw = conv(x_bchw)
    return einops.rearrange(y_bdhw, "b d h w -> b (h w) d")
