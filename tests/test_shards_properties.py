"""
On OSC, with the fish shell:

for shards in /fs/scratch/PAS2136/samuelstevens/cache/saev/*; uv run pytest tests/test_shards*.py --shards $shards; end
"""

import contextlib
import dataclasses
import json
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
            family="clip",
            ckpt="ckpt",
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
            d_model=512,
            n_examples=draw(st.integers(min_value=1, max_value=10_000_000)),
            patches_per_shard=draw(st.integers(min_value=1, max_value=200_000_000)),
            data={"__class__": "Fake"},
            dataset=pathlib.Path("/fake/dataset/path"),
        )
    except AssertionError:
        reject()


@contextlib.contextmanager
def tmp_shards_root():
    """Create a temporary shard root directory."""
    # We cannot use the tmp_path fixture because of Hypothesis.
    # See https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck.function_scoped_fixture
    with tempfile.TemporaryDirectory() as tmp_path:
        shards_root = pathlib.Path(tmp_path) / "saev" / "shards"
        shards_root.mkdir(parents=True)
        yield shards_root


@pytest.mark.slow
def test_dataloader_batches():
    vit_cls = models.load_model_cls("clip")
    img_tr, sample_tr = vit_cls.make_transforms("ViT-B-32/openai", 49)
    dataloader = get_dataloader(
        datasets.Fake(n_examples=10),
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
def test_shard_writer_and_dataset_e2e():
    with tmp_shards_root() as shards_root:
        data_cfg = datasets.Fake(n_examples=128)

        md = Metadata(
            family="clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            layers=(-2, -1),
            patches_per_ex=16,
            cls_token=True,
            d_model=128,
            n_examples=data_cfg.n_examples,
            patches_per_shard=128,
            data={
                **dataclasses.asdict(data_cfg),
                "__class__": data_cfg.__class__.__name__,
            },
            dataset=pathlib.Path("/fake/dataset/path"),
        )
        md.dump(shards_root)

        vit_cls = models.load_model_cls(md.family)
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
        writer = ShardWriter(shards_root, md)
        dataset = Dataset(
            IndexedConfig(shards=shards_root / md.hash, patches="cls", layer=-1)
        )

        i = 0
        for b, batch in zip(range(4), dataloader):
            # Don't care about the forward pass.
            out, cache = vit(batch["image"])
            del out

            writer.write_batch(cache, i)
            i += len(cache)
            assert cache.shape == (
                dataloader.batch_size,
                len(md.layers),
                17,
                md.d_model,
            )

            acts = [dataset[i.item()]["act"] for i in batch["index"]]
            from_dataset = torch.stack(acts)
            torch.testing.assert_close(cache[:, -1, 0], from_dataset)
            print(f"Batch {b} matched.")


def test_shards_json_is_emitted():
    with tmp_shards_root() as shards_root:
        shards = worker_fn(
            family="clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            patches_per_ex=16,
            cls_token=True,
            d_model=128,
            layers=[0],
            data=datasets.Fake(n_examples=10),
            batch_size=12,
            n_workers=4,
            patches_per_shard=256,
            shards_root=shards_root,
            device="cpu",
        )

        shards_json = shards / "shards.json"

        # Assert that file exists and has correct contents
        assert shards_json.exists(), (
            "protocol.md requires shards.json in the output dir"
        )

        arr = json.loads(shards_json.read_text())
        # Should be a list of one entry per bin file
        md = Metadata.load(shards)
        expected_n_shards = md.n_shards
        assert isinstance(arr, list) and len(arr) == expected_n_shards

        # Each entry has `name` and `n_examples`
        for idx, entry in enumerate(arr):
            assert entry["name"] == f"acts{idx:06d}.bin"
            # last shard may be smaller
            assert entry["n_examples"] > 0 and isinstance(entry["n_examples"], int)


@given(md=metadatas())
def test_metadata_json_has_required_keys(md):
    """Check, per the protocol.md document, that every key is present on disk."""

    with tmp_shards_root() as shards_root:
        # Write metadata.json
        md.dump(shards_root)

        md = json.loads((shards_root / md.hash / "metadata.json").read_text())
        # required keys from the protocol
        expected = {
            "family",
            "ckpt",
            "layers",
            "patches_per_ex",
            "cls_token",
            "d_model",
            "n_examples",
            "patches_per_shard",
            "data",
            "dataset",
            "pixel_agg",
            "dtype",
            "protocol",
        }
        assert set(md) == expected, (
            f"metadata.json keys must exactly match spec, got {set(md)}"
        )

        # dtype & protocol must be fixed strings
        assert md["dtype"] == "float32"
        assert md["protocol"] == "2.0"

        # data must be a dict with a __class__ key
        assert isinstance(md["data"], dict)
        assert "__class__" in md["data"]


@settings(deadline=None, max_examples=20)
@given(
    patches_per_shard=st.integers(min_value=1, max_value=10_000_000),
    patches_per_ex=st.integers(min_value=1, max_value=1000),
    n_layers=st.integers(min_value=1, max_value=50),
    cls_token=st.booleans(),
)
def test_shard_size_consistency(patches_per_shard, patches_per_ex, n_layers, cls_token):
    assume(patches_per_shard >= (patches_per_ex + 2) * n_layers)
    md = Metadata(
        family="clip",
        ckpt="ckpt",
        layers=tuple(range(n_layers)),
        patches_per_ex=patches_per_ex,
        cls_token=cls_token,
        d_model=1,
        n_examples=1,
        patches_per_shard=patches_per_shard,
        data={"__class__": "Fake"},
        dataset=pathlib.Path("/fake/dataset/path"),
    )
    # compute spec value
    n_tokens = patches_per_ex + (1 if cls_token else 0)
    spec_nv = patches_per_shard // (n_tokens * n_layers)
    # via Metadata property
    assert md.ex_per_shard == spec_nv

    with tmp_shards_root() as shards_root:
        sw = ShardWriter(shards_root, md)
        assert sw.ex_per_shard == spec_nv


@pytest.mark.parametrize(
    "patches_per_ex,cls_token,expected",
    [
        (49, False, 49),
        (49, True, 50),
        (16, False, 16),
        (16, True, 17),
        (1, False, 1),
        (1, True, 2),
    ],
)
def test_tokens_per_ex(patches_per_ex, cls_token, expected):
    md = Metadata(
        family="clip",
        ckpt="ckpt",
        layers=(0,),
        patches_per_ex=patches_per_ex,
        cls_token=cls_token,
        d_model=512,
        n_examples=1,
        patches_per_shard=1000,
        data={"__class__": "Fake"},
        dataset=pathlib.Path("/fake/dataset/path"),
    )
    assert md.tokens_per_ex == expected


@given(md=metadatas())
def test_metadata_smoke(md):
    """Test that all derived properties and methods on Metadata don't throw exceptions."""
    md.hash
    md.tokens_per_ex
    md.n_shards
    md.ex_per_shard
    md.shard_shape
