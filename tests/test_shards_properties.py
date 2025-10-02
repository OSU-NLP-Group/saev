"""
On OSC, with the fish shell:

for shards in /fs/scratch/PAS2136/samuelstevens/cache/saev/*; uv run pytest tests/test_shards*.py --shards $shards; end
"""

import dataclasses
import json
import os
import pathlib
import tempfile

import pytest
import torch
from hypothesis import assume, given, reject, settings
from hypothesis import strategies as st

from saev.data import Dataset, IndexedConfig, datasets, models
from saev.data.shards import (
    Metadata,
    RecordedTransformer,
    ShardWriter,
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
            patches_per_ex=draw(st.integers(min_value=1, max_value=512)),
            cls_token=draw(st.booleans()),
            d_vit=512,
            n_ex=draw(st.integers(min_value=1, max_value=10_000_000)),
            max_patches_per_shard=draw(st.integers(min_value=1, max_value=200_000_000)),
            data={"__class__": "Fake"},
        )
    except AssertionError:
        reject()


@st.composite
def worker_fn_kwargs(draw) -> dict[str, object]:
    return dict(
        data=datasets.Fake(n_ex=1024),
        vit_family=draw(st.sampled_from(["clip", "siglip", "dinov2"])),
    )


def patches():
    return st.sampled_from(["cls", "image", "all"])


def layers():
    return st.one_of(st.just("all"), st.integers())


@pytest.mark.slow
def test_dataloader_batches(tmp_path):
    vit_cls = models.load_vit_cls("clip")
    img_tr, sample_tr = vit_cls.make_transforms("ViT-B-32/openai", 49)
    dataloader = get_dataloader(
        datasets.Fake(n_ex=10),
        batch_size=8,
        n_workers=0,
        img_tr=img_tr,
        sample_tr=sample_tr,
    )
    batch = next(iter(dataloader))

    assert isinstance(batch, dict)
    assert "image" in batch
    assert "index" in batch

    torch.testing.assert_close(batch["index"], torch.arange(8))
    assert batch["image"].shape == (8, 3, 224, 224)


@pytest.mark.slow
def test_shard_writer_and_dataset_e2e(tmp_path):
    data_cfg = datasets.Fake(n_ex=128)
    md = Metadata(
        family="clip",
        ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        layers=(-2, -1),
        patches_per_ex=16,
        cls_token=True,
        d_model=128,
        n_ex=data_cfg.n_ex,
        patches_per_shard=128,
        data={**dataclasses.asdict(data_cfg), "__class__": data_cfg.__class__.__name__},
    )
    vit_cls = models.load_vit_cls(md.family)
    img_tr, sample_tr = vit_cls.make_transforms(md.ckpt, md.patches_per_ex)
    vit = RecordedTransformer(
        vit_cls(md.ckpt), md.patches_per_ex, md.cls_token, md.layers
    )
    dataloader = get_dataloader(
        data_cfg,
        batch_size=8,
        n_workers=8,
        img_tr=img_tr,
        sample_tr=sample_tr,
    )
    writer = ShardWriter(tmp_path / md.hash, md)
    dataset = Dataset(
        IndexedConfig(shard_root=get_acts_dir(cfg), patches="cls", layer=-1)
    )

    i = 0
    for b, batch in zip(range(4), dataloader):
        # Don't care about the forward pass.
        out, cache = vit(batch["image"])
        del out

        writer.write_batch(cache, i)
        i += len(cache)
        assert cache.shape == (cfg.vit_batch_size, len(cfg.vit_layers), 17, cfg.d_vit)

        acts = [dataset[i.item()]["act"] for i in batch["index"]]
        from_dataset = torch.stack(acts)
        torch.testing.assert_close(cache[:, -1, 0], from_dataset)
        print(f"Batch {b} matched.")


def test_shards_json_is_emitted(tmp_path):
    cfg = Config(
        data=datasets.Fake(n_ex=10),
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


@given(kwargs=worker_fn_kwargs())
def test_metadata_json_has_required_keys(kwargs):
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
            "pixel_agg",
            "dtype",
            "protocol",
        }
        assert set(md) == expected, (
            f"metadata.json keys must exactly match spec, got {set(md)}"
        )

        # dtype & protocol must be fixed strings
        assert md["dtype"] == "float32"
        assert md["protocol"] == "1.1"

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
    assume(max_patches >= n_patches)
    # We cannot use the tmp_path fixture because of Hypothesis.
    # See https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck.function_scoped_fixture
    with tempfile.TemporaryDirectory() as tmp_path:
        md = Metadata(
            family="clip",
            ckpt="ckpt",
            layers=tuple(range(n_layers)),
            patches_per_ex=n_patches,
            cls_token=cls_token,
            d_model=1,
            n_ex=1,
            patches_per_shard=max_patches,
            data={"__class__": "Fake"},
        )
        # compute _spec_ value
        n_tokens = n_patches + (1 if cls_token else 0)
        spec_nv = max_patches // (n_tokens * n_layers)
        # via Metadata property
        assert md.n_imgs_per_shard == spec_nv
        # via ShardWriter logic
        cfg = Config(
            data=datasets.Fake(n_ex=128),
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
