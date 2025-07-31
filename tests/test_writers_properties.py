"""
On OSC, with the fish shell:

for shards in /fs/scratch/PAS2136/samuelstevens/cache/saev/*; uv run pytest tests/test_writers.py --shards $shards; end
"""

import json
import os
import pathlib
import tempfile

import pytest
import torch
from hypothesis import given, reject, settings
from hypothesis import strategies as st

from saev.data import images, models
from saev.data.torch import Config as TorchConfig
from saev.data.torch import Dataset as TorchDataset
from saev.data.writers import (
    Config,
    IndexLookup,
    Metadata,
    RecordedVisionTransformer,
    ShardWriter,
    get_acts_dir,
    get_dataloader,
    worker_fn,
)


@st.composite
def metadatas(draw) -> Metadata:
    try:
        return Metadata(
            vit_family="clip",
            vit_ckpt="ckpt",
            layers=tuple(
                sorted(
                    draw(
                        st.sets(
                            st.integers(min_value=0, max_value=24),
                            min_size=1,
                            max_size=24,
                        )
                    )
                )
            ),
            n_patches_per_img=draw(st.integers(min_value=1, max_value=512)),
            cls_token=draw(st.booleans()),
            d_vit=512,
            n_imgs=draw(st.integers(min_value=1, max_value=10_000_000)),
            max_patches_per_shard=draw(st.integers(min_value=1, max_value=200_000_000)),
            data={"__class__": "Fake"},
        )
    except AssertionError:
        reject()


@st.composite
def writers_configs(draw) -> Config:
    return Config(vit_family=draw(st.sampled_from(["clip", "siglip", "dinov2"])))


def patches():
    return st.sampled_from(["cls", "image", "all"])


def layers():
    return st.one_of(st.just("all"), st.integers())


@pytest.mark.slow
def test_dataloader_batches(tmp_path):
    cfg = Config(
        data=images.Imagenet(split="validation"),
        vit_ckpt="ViT-B-32/openai",
        d_vit=768,
        vit_layers=[-2, -1],
        n_patches_per_img=49,
        vit_batch_size=8,
        dump_to=str(tmp_path),
    )
    dataloader = get_dataloader(
        cfg, img_transform=models.make_img_transform(cfg.vit_family, cfg.vit_ckpt)
    )
    batch = next(iter(dataloader))

    assert isinstance(batch, dict)
    assert "image" in batch
    assert "index" in batch

    torch.testing.assert_close(batch["index"], torch.arange(8))
    assert batch["image"].shape == (8, 3, 224, 224)


@pytest.mark.slow
def test_shard_writer_and_dataset_e2e(tmp_path):
    cfg = Config(
        data=images.Imagenet(split="validation"),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=128,
        n_patches_per_img=16,
        vit_layers=[-2, -1],
        vit_batch_size=8,
        n_workers=8,
        dump_to=str(tmp_path),
    )
    vit = models.make_vit(cfg.vit_family, cfg.vit_ckpt)
    vit = RecordedVisionTransformer(
        vit, cfg.n_patches_per_img, cfg.cls_token, cfg.vit_layers
    )
    dataloader = get_dataloader(
        cfg,
        img_transform=models.make_img_transform(cfg.vit_family, cfg.vit_ckpt),
    )
    writer = ShardWriter(cfg)
    dataset = TorchDataset(
        TorchConfig(
            shard_root=get_acts_dir(cfg),
            patches="cls",
            layer=-1,
            scale_mean=False,
            scale_norm=False,
        )
    )

    i = 0
    for b, batch in zip(range(4), dataloader):
        # Don't care about the forward pass.
        out, cache = vit(batch["image"])
        del out

        writer[i : i + len(cache)] = cache
        i += len(cache)
        assert cache.shape == (cfg.vit_batch_size, len(cfg.vit_layers), 17, cfg.d_vit)

        acts = [dataset[i.item()]["act"] for i in batch["index"]]
        from_dataset = torch.stack(acts)
        torch.testing.assert_close(cache[:, -1, 0], from_dataset)
        print(f"Batch {b} matched.")


@given(metadatas(), st.data())
def test_api_surface(metadata, data):
    if metadata.cls_token:
        patches = data.draw(st.sampled_from(["cls", "all", "image"]))
    else:
        patches = data.draw(st.sampled_from(["all", "image"]))
    layer = data.draw(st.sampled_from([*metadata.layers, "all"]))

    il = IndexLookup(metadata, patches, layer)

    assert hasattr(il, "map_global")
    assert hasattr(il, "map_img")
    assert hasattr(il, "length")


@given(metadatas(), st.data())
def test_roundtrip(metadata, data):
    if metadata.cls_token:
        patches = data.draw(st.sampled_from(["cls", "all", "image"]))
    else:
        patches = data.draw(st.sampled_from(["all", "image"]))

    layer = data.draw(st.sampled_from([*metadata.layers, "all"]))

    il = IndexLookup(metadata, patches, layer)

    length = il.length()
    assert 1 <= length

    i = data.draw(st.integers(min_value=0, max_value=length - 1))

    sh_i, (img_i_in_sh, layer_i, token_i) = il.map_global(i)

    # Basic invariants
    assert 0 <= sh_i < metadata.n_shards
    assert 0 <= img_i_in_sh < metadata.n_imgs_per_shard


@given(metadatas(), st.data())
def test_length_always_nonnegative(metadata, data):
    if metadata.cls_token:
        patches = data.draw(st.sampled_from(["cls", "all", "image"]))
    else:
        patches = data.draw(st.sampled_from(["all", "image"]))
    layer = data.draw(st.sampled_from([*metadata.layers, "all"]))

    il = IndexLookup(metadata, patches, layer)

    assert il.length() >= 0


@given(metadatas(), st.data())
def test_negative_index_raises(metadata, data):
    if metadata.cls_token:
        patches = data.draw(st.sampled_from(["cls", "all", "image"]))
    else:
        patches = data.draw(st.sampled_from(["all", "image"]))
    layer = data.draw(st.sampled_from([*metadata.layers, "all"]))

    il = IndexLookup(metadata, patches, layer)

    with pytest.raises(IndexError):
        il.map_global(-1)

    with pytest.raises(IndexError):
        il.map_img(-1)


@given(metadatas(), st.data())
def test_index_equal_length_raises(metadata, data):
    if metadata.cls_token:
        patches = data.draw(st.sampled_from(["cls", "all", "image"]))
    else:
        patches = data.draw(st.sampled_from(["all", "image"]))
    layer = data.draw(st.sampled_from([*metadata.layers, "all"]))

    il = IndexLookup(metadata, patches, layer)
    length = il.length()

    with pytest.raises(IndexError):
        il.map_global(length)


@given(
    layers=st.sets(st.integers(min_value=0, max_value=24), min_size=1, max_size=24),
    n_patches_per_img=st.integers(min_value=1, max_value=256),
    data=st.data(),
)
def test_missing_layer(layers, n_patches_per_img, data):
    missing_layer = data.draw(
        st.sampled_from([i for i in range(25) if i not in layers]),
        label="missing_layer",
    )

    layers = tuple(sorted(layers))

    md = Metadata(
        n_imgs=10_000,
        n_patches_per_img=n_patches_per_img,
        layers=layers,
        max_patches_per_shard=200_000_000,
        vit_family="clip",
        vit_ckpt="ckpt",
        cls_token=False,
        d_vit=512,
        data={"__class__": "Fake"},
    )

    with pytest.raises(Exception):
        # IndexLookup should complain
        IndexLookup(md, "cls", missing_layer)


@given(md=metadatas(), patches=patches())
def test_missing_cls_token(md, patches):
    if not md.cls_token and patches == "cls":
        with pytest.raises(Exception):
            IndexLookup(md, patches, md.layers[0])
    else:
        IndexLookup(md, patches, md.layers[0])


def test_shards_json_is_emitted(tmp_path):
    cfg = Config(
        data=images.Fake(n_imgs=10),
        dump_to=str(tmp_path),
        vit_layers=[0],
        n_patches_per_img=16,
        cls_token=True,
        max_patches_per_shard=256,
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=128,
        vit_batch_size=12,
    )
    worker_fn(cfg)

    outdir = tmp_path / Metadata.from_cfg(cfg).hash
    shards_json = outdir / "shards.json"

    # Assert that file exists and has correct contents
    assert shards_json.exists(), "protocol.md requires shards.json in the output dir"

    arr = json.loads(shards_json.read_text())
    # Should be a list of one entry per bin file
    expected_n_shards = Metadata.from_cfg(cfg).n_shards
    assert isinstance(arr, list) and len(arr) == expected_n_shards

    # Each entry has `name` and `n_imgs`
    for idx, entry in enumerate(arr):
        assert entry["name"] == f"acts{idx:06d}.bin"
        # last shard may be smaller
        assert entry["n_imgs"] > 0 and isinstance(entry["n_imgs"], int)


@given(cfg=writers_configs())
def test_metadata_json_has_required_keys(cfg):
    # We cannot use the tmp_path fixture because of Hypothesis.
    # See https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck.function_scoped_fixture
    with tempfile.TemporaryDirectory() as tmp_path:
        outdir = pathlib.Path(tmp_path, Metadata.from_cfg(cfg).hash)
        os.makedirs(outdir)
        # Write metadata.json
        Metadata.from_cfg(cfg).dump(str(outdir))

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


@settings(deadline=None, max_examples=20)
@given(
    max_patches=st.integers(min_value=1, max_value=10_000_000),
    n_patches=st.integers(min_value=1, max_value=1000),
    n_layers=st.integers(min_value=1, max_value=50),
    cls_token=st.booleans(),
)
def test_shard_size_consistency(max_patches, n_patches, n_layers, cls_token):
    # We cannot use the tmp_path fixture because of Hypothesis.
    # See https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck.function_scoped_fixture
    with tempfile.TemporaryDirectory() as tmp_path:
        md = Metadata(
            vit_family="clip",
            vit_ckpt="ckpt",
            layers=tuple(range(n_layers)),
            n_patches_per_img=n_patches,
            cls_token=cls_token,
            d_vit=1,
            n_imgs=1,
            max_patches_per_shard=max_patches,
            data={"__class__": "Fake"},
        )
        # compute _spec_ value
        T = n_patches + (1 if cls_token else 0)
        spec_nv = max_patches // (T * n_layers)
        # via Metadata property
        assert md.n_imgs_per_shard == spec_nv
        # via ShardWriter logic
        cfg = Config(
            data=images.Imagenet(),
            dump_to=str(tmp_path),
            vit_layers=list(range(n_layers)),
            n_patches_per_img=n_patches,
            cls_token=cls_token,
            max_patches_per_shard=max_patches,
            vit_family="clip",
            vit_ckpt="ckpt",
        )
        sw = ShardWriter(cfg)
        assert sw.n_imgs_per_shard == spec_nv
