import json
import os

from hypothesis import given
from hypothesis import strategies as st

from saev.data import images
from saev.data.writers import Config, IndexLookup, Metadata, ShardWriter


def test_first_patch():
    md = Metadata(
        n_imgs=10_000,
        n_patches_per_img=256,
        layers=(11,),
        max_patches_per_shard=200_000_000,
        vit_family="clip",
        vit_ckpt="ckpt",
        cls_token=True,
        d_vit=512,
        data="test",
    )
    il = IndexLookup(md, "cls", 11)
    assert il.map_global(0) == (0, (0, 0, 0))


def test_second_img_with_cls():
    meta = Metadata(
        n_imgs=10_000,
        n_patches_per_img=196,
        layers=(-1,),
        max_patches_per_shard=200_000_000,
        vit_family="clip",
        vit_ckpt="ckpt",
        cls_token=True,
        d_vit=512,
        data="test",
    )
    il = IndexLookup(meta, "image", -1)
    sh, (img_i, layer_i, token_i) = il.map_global(196)
    assert sh == 0
    assert img_i == 1
    assert layer_i == 0
    assert token_i == 0


def test_second_img_without_cls():
    meta = Metadata(
        n_imgs=10_000,
        n_patches_per_img=196,
        layers=(-1,),
        max_patches_per_shard=200_000_000,
        vit_family="clip",
        vit_ckpt="ckpt",
        cls_token=False,
        d_vit=512,
        data="test",
    )
    il = IndexLookup(meta, "image", -1)
    sh, (img_i, layer_i, token_i) = il.map_global(196)
    assert sh == 0
    assert img_i == 1
    assert layer_i == 0
    assert token_i == 0


def test_third_image_with_cls():
    meta = Metadata(
        n_imgs=10_000,
        n_patches_per_img=64,
        layers=(0, 1, 2, 3),
        max_patches_per_shard=200_000_000,
        vit_family="clip",
        vit_ckpt="ckpt",
        cls_token=True,
        d_vit=1024,
        data="test",
    )
    il = IndexLookup(meta, "all", 2)
    sh, (img_i, layer_i, token_i) = il.map_global(130)
    assert sh == 0
    assert img_i == 2
    assert layer_i == 2
    assert token_i == 0


def test_image_with_layers():
    md = Metadata(
        n_imgs=10_000,
        n_patches_per_img=64,
        layers=(0, 1, 2, 3),
        max_patches_per_shard=200_000_000,
        vit_family="clip",
        vit_ckpt="ckpt",
        cls_token=False,
        d_vit=1024,
        data="test",
    )
    il = IndexLookup(md, "all", "all")
    sh, (img_i, layer_i, token_i) = il.map_global(128)
    assert sh == 0
    assert img_i == 0
    assert layer_i == 2
    assert token_i == 0


def test_metadata_json_has_required_keys(tmp_path, example_cfg):
    outdir = tmp_path / Metadata.from_cfg(example_cfg).hash
    os.makedirs(outdir)
    # Write metadata.json
    Metadata.from_cfg(example_cfg).dump(outdir / "metadata.json")

    md = json.loads((outdir / "metadata.json").read_text())
    # required keys from the protocol
    expected = {
        "vit_family",
        "vit_ckpt",
        "layers",
        "n_patches_per_img",
        "cls_token",
        "d_vit",
        "n_imgs",
        "max_patches_per_shard",
        "data",
        "dtype",
        "protocol",
    }
    assert set(md) == expected, (
        f"metadata.json keys must exactly match spec, got {set(md)}"
    )

    # dtype & protocol must be fixed strings
    assert md["dtype"] == "float32"
    assert md["protocol"] == "1.0.0"

    # data must be a dict with a __class__ key
    assert isinstance(md["data"], dict)
    assert "__class__" in md["data"]


@given(
    max_patches=st.integers(min_value=1, max_value=10_000_000),
    n_patches=st.integers(min_value=1, max_value=1000),
    n_layers=st.integers(min_value=1, max_value=50),
    cls_token=st.booleans(),
)
def test_shard_size_consistency(max_patches, n_patches, n_layers, cls_token):
    md = Metadata(
        vit_family="clip",
        vit_ckpt="ckpt",
        layers=tuple(range(n_layers)),
        n_patches_per_img=n_patches,
        cls_token=cls_token,
        d_vit=1,
        n_imgs=1,
        max_patches_per_shard=max_patches,
        data="dummy",
    )
    # compute _spec_ value
    T = n_patches + (1 if cls_token else 0)
    spec_nv = max_patches // (T * n_layers)
    # via Metadata property
    assert md.n_imgs_per_shard == spec_nv
    # via ShardWriter logic
    cfg = Config(
        data=images.Imagenet(),
        dump_to=".",
        vit_layers=list(range(n_layers)),
        n_patches_per_img=n_patches,
        cls_token=cls_token,
        max_patches_per_shard=max_patches,
        vit_family="clip",
        vit_ckpt="ckpt",
    )
    sw = ShardWriter(cfg)
    assert sw.n_imgs_per_shard == spec_nv
