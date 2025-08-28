# src/saev/data/transforms.py
import math
import typing as tp

import beartype
import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import v2


@beartype.beartype
class FlexResize(v2.Transform):
    def __init__(self, patch_size: int, n_patches: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches

    def transform(self, inpt: tp.Any, params: dict[str, tp.Any]):
        if isinstance(inpt, Image.Image):
            return _resize_to_patch_grid(inpt, p=self.patch_size, n=self.n_patches)
        else:
            raise TypeError(type(inpt))


@beartype.beartype
def _resize_to_patch_grid(
    img: Image.Image, *, p: int, n: int, resample=Image.LANCZOS
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
class Unfold(nn.Module):
    def __init__(self, patch_size: int, n_patches: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches

    def forward(self, sample: dict[str, object]) -> dict[str, object]:
        assert "image" in sample
        img = sample["image"]
        c, h, w = img.shape
        p = self.patch_size
        assert (h % p == 0) and (w % p == 0), f"Got {h}x{w}, patch={p}"

        cols_bpn = F.unfold(img[None, ...], kernel_size=p, stride=p)
        _, k, n = cols_bpn.shape
        assert n == self.n_patches, f"Expected n={self.n_patches}, got {n}"
        assert k == c * p * p, f"k mismatch: {k} != {c}*{p}*{p}"

        sample["image"] = einops.rearrange(cols_bpn, "() k n -> n k").contiguous()
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

    b, c, h, w = x_bchw.shape

    assert h % k == 0 and w % k == 0

    cols_bpn = F.unfold(x_bchw, kernel_size=k, stride=k)  # (B, C x P x P, D)
    w_dp = conv.weight.reshape(conv.out_channels, c * k * k)
    tokens_bnd = einops.rearrange(cols_bpn, "b px n -> b n px") @ w_dp.T
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
