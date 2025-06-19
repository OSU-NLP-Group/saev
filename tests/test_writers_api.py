from saev.data.writers import IndexLookup, Metadata


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
        data={"__class__": "Fake"},
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
        data={"__class__": "Fake"},
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
        data={"__class__": "Fake"},
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
        data={"__class__": "Fake"},
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
        data={"__class__": "Fake"},
    )
    il = IndexLookup(md, "all", "all")
    sh, (img_i, layer_i, token_i) = il.map_global(128)
    assert sh == 0
    assert img_i == 0
    assert layer_i == 2
    assert token_i == 0
