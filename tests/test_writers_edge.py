import pytest

from saev.data.writers import IndexLookup, Metadata


def test_singleton_dataset():
    meta = Metadata(
        n_imgs=1,
        n_patches_per_img=196,
        layers=(0,),
        max_patches_per_shard=10_000,
        vit_family="clip",
        vit_ckpt="ckpt",
        cls_token=True,
        d_vit=512,
        data="test",
    )
    il = IndexLookup(meta, "cls", 0)
    assert il.length() == 1
    shard_i, (img_i, layer_i, token_i) = il.map_global(0)
    assert shard_i == 0
    assert img_i == 0
    assert layer_i == 0
    assert token_i == 0

    with pytest.raises(IndexError):
        il.map_global(1)
