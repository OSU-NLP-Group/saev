import dataclasses
import functools
import itertools
import logging
import os.path
import typing as tp
from collections.abc import Callable, Iterable

import beartype
import numpy as np
import requests
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor

from saev import helpers
from saev.data import models

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    img_size_x: int = 512
    img_size_y: int = 128
    patch_size: int = 16
    in_chans: int = 1
    embed_dim: int = 768
    depth: int = 12
    n_heads: int = 12
    mlp_ratio: float = 4.0
    pos_trainable: bool = False
    qkv_bias: bool = True
    qk_norm: bool = False
    init_values: float | None = None
    drop_rate: float = 0.0
    norm_layer_eps: float = 1e-6
    global_pool: tp.Literal["mean", "cls"] = "mean"  # "mean", "cls", or None

    @property
    def n_patches_x(self):
        return self.img_size_x // self.patch_size

    @property
    def n_patches_y(self):
        return self.img_size_y // self.patch_size

    @property
    def n_patches(self):
        return self.n_patches_x * self.n_patches_y

    @property
    def n_tokens(self):
        return self.n_patches + 1


# --- positional encodings -----------------------------------------------------


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    # pos: array of positions, shape (M,)
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed_flexible(
    embed_dim: int,
    grid_size: tuple[int, int],
    cls_token: bool = False,
) -> np.ndarray:
    # grid_size: (H, W) of the patch grid
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w, h
    grid = np.stack(grid, axis=0)  # 2, H, W
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# --- timm-ish helpers ---------------------------------------------------------


def _ntuple(n: int):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(itertools.repeat(x, n))

    return parse


@jaxtyped(typechecker=beartype.beartype)
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


@jaxtyped(typechecker=beartype.beartype)
class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias: bool = True,
        drop: float = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _ntuple(2)(bias)
        drop_probs = _ntuple(2)(drop)
        linear_layer = (
            functools.partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        )

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


@jaxtyped(typechecker=beartype.beartype)
class Attention(nn.Module):
    fused_attn: bool = True

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim should be divisible by n_heads"
        if qk_norm or scale_norm:
            assert norm_layer is not None, (
                "norm_layer must be provided if qk_norm or scale_norm is True"
            )
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn_weights = attn.softmax(dim=-1)
            x = self.attn_drop(attn_weights) @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


@jaxtyped(typechecker=beartype.beartype)
class LayerScale(nn.Module):
    def __init__(
        self, dim: int, init_values: float = 1e-5, inplace: bool = False
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.mul_(self.gamma)
        return x * self.gamma


@jaxtyped(typechecker=beartype.beartype)
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x_skip = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.ls1(x)
        x = self.drop_path1(x)
        x = x + x_skip
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


@jaxtyped(typechecker=beartype.beartype)
class PatchEmbed(nn.Module):
    """Image (time x mel) to patch embeddings."""

    def __init__(
        self,
        img_size: tuple[int, int] = (512, 128),
        patch_size: tuple[int, int] = (16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = _ntuple(2)(img_size)
        patch_size = _ntuple(2)(patch_size)
        n_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = n_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2)  # [B, D, H'*W']
        x = x.transpose(1, 2)  # [B, H'*W', D]
        return x


@jaxtyped(typechecker=beartype.beartype)
class Encoder(nn.Module):
    """Pure PyTorch Bird-MAE backbone (no HF)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(
            img_size=(cfg.img_size_x, cfg.img_size_y),
            patch_size=(cfg.patch_size, cfg.patch_size),
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, cfg.n_patches + 1, cfg.embed_dim),
            requires_grad=cfg.pos_trainable,
        )

        if self.pos_embed.data.shape[1] == cfg.n_tokens:
            pos_embed_np = get_2d_sincos_pos_embed_flexible(
                self.pos_embed.shape[-1],
                self.patch_embed.patch_hw,
                cls_token=True,
            )
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed_np).float().unsqueeze(0)
            )
        else:
            logger.warning(
                "Positional embedding shape mismatch. Will not initialize sin-cos pos embed."
            )

        dpr = [x.item() for x in torch.linspace(0, cfg.drop_rate, cfg.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=cfg.embed_dim,
                n_heads=cfg.n_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=cfg.qk_norm,
                init_values=cfg.init_values,
                proj_drop=cfg.drop_rate,
                attn_drop=cfg.drop_rate,
                drop_path=dpr[i],
                norm_layer=functools.partial(nn.LayerNorm, eps=cfg.norm_layer_eps),
            )
            for i in range(cfg.depth)
        ])

        self.pos_drop = nn.Dropout(p=cfg.drop_rate)
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_layer_eps)
        self.fc_norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_layer_eps)
        self.global_pool = cfg.global_pool

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d):
            w = module.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(
        self, input_values: Float[Tensor, "batch 1 512 128"]
    ) -> dict[str, Tensor]:
        if input_values.ndim == 3:
            input_values = input_values.unsqueeze(0)

        B, C, X, Y = input_values.shape
        assert X == self.cfg.img_size_x
        assert Y == self.cfg.img_size_y

        x = self.patch_embed(input_values)  # [B, N_patches, D]

        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N_patches, D]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        tokens = x  # [B, 1+N, D]

        if self.global_pool == "mean":
            pooled = tokens[:, 1:, :].mean(dim=1)
            pooled = self.fc_norm(pooled)
        elif self.global_pool == "cls":
            x_norm = self.norm(tokens)
            pooled = x_norm[:, 0]
        else:
            raise ValueError(f"Invalid global_pool: {self.global_pool}")

        return dict(pooled=pooled, tokens=tokens[:, 1:, :])


_PRETRAINED_CFGS = {
    "Bird-MAE-Base": Config(depth=12, embed_dim=768, n_heads=12),
    "Bird-MAE-Large": Config(depth=24, embed_dim=1024, n_heads=16),
    "Bird-MAE-Huge": Config(depth=32, embed_dim=1280, n_heads=16),
}


@beartype.beartype
def load(ckpt: str, device="cpu") -> Encoder:
    if ckpt not in _PRETRAINED_CFGS:
        raise ValueError(f"Checkpoint '{ckpt}' not in {list(_PRETRAINED_CFGS)}.")
    cfg = _PRETRAINED_CFGS[ckpt]

    fpath = download_hf_file(ckpt)
    state_dict = safetensors.torch.load_file(fpath)

    model = Encoder(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=True, assign=True)
    assert not missing, missing
    assert not unexpected, unexpected

    model = model.to(device)
    return model


@beartype.beartype
def download_hf_file(ckpt: str, *, force: bool = False) -> str:
    # Construct the URL
    url = f"https://huggingface.co/DBD-research-group/{ckpt}/resolve/main/model.safetensors"

    # Create the local path
    cache_dir = helpers.get_cache_dir()
    local_dir = os.path.join(cache_dir, "hf", ckpt)
    local_path = os.path.join(local_dir, "model.safetensors")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Check if the file exists
    if os.path.exists(local_path) and not force:
        return local_path

    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path


@jaxtyped(typechecker=beartype.beartype)
class Transformer(nn.Module, models.Transformer):
    family: str = "bird-mae"
    patch_size: int = 16

    def __init__(self, ckpt: str):
        super().__init__()
        self.model = load(ckpt)

        self._ckpt = ckpt
        self.logger = logging.getLogger(ckpt.lower())

    @property
    def ckpt(self) -> str:
        return self._ckpt

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.blocks

    def get_token_i(self, content_tokens_per_example: int) -> slice:
        n_reg = self.model.cfg.n_storage_tokens
        patches = torch.cat((
            torch.tensor([0]),  # CLS token
            torch.arange(n_reg + 1, n_reg + 1 + content_tokens_per_example),  # patches
        ))
        return patches

    def forward(
        self, batch: Float[Tensor, "batch n kernel"], **kwargs
    ) -> Float[Tensor, "batch patches dim"]:
        if kwargs:
            self.logger.info("Unused kwargs: %s", kwargs)
        dct = self.model(batch)

        features = torch.cat((dct["pooled"][:, None, :], dct["tokens"]), axis=1)
        return features

    @staticmethod
    def make_transforms(
        ckpt: str, n_patches_per_img: int
    ) -> tuple[Callable, Callable | None]:
        """Create transforms for preprocessing: (img_transform, sample_transform | None)."""
        img_transform = v2.Compose([
            transforms.FlexResize(patch_size=16, n_patches=n_patches_per_img),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])
        sample_transform = transforms.Patchify(
            patch_size=16, n_patches=n_patches_per_img
        )
        return img_transform, sample_transform

    @staticmethod
    def make_resize(
        ckpt: str,
        n_patches_per_img: int,
        *,
        scale: float = 1.0,
        resample: Image.Resampling = Image.LANCZOS,
    ) -> Callable[[Image.Image], Image.Image]:
        """Create resize transform for visualization. Use resample=Image.NEAREST for segmentation masks."""
        import functools

        return functools.partial(
            transforms.resize_to_patch_grid,
            p=int(16 * scale),
            n=n_patches_per_img,
            resample=resample,
        )
