import base64
import contextlib
import dataclasses
import json
import pathlib
import pickle
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
from hypothesis import assume, given, reject, settings
from hypothesis import strategies as st

from saev.data import IndexedConfig, IndexedDataset, datasets, models
from saev.data.shards import (
    Metadata,
    RecordedTransformer,
    ShardWriter,
    get_dataloader,
    worker_fn,
)

mp.set_start_method("spawn", force=True)


@st.composite
def metadatas(draw) -> Metadata:
    try:
        n_examples = draw(st.integers(min_value=1, max_value=10_000_000))
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
            content_tokens_per_example=draw(st.integers(min_value=1, max_value=512)),
            cls_token=draw(st.booleans()),
            d_model=512,
            n_examples=n_examples,
            max_tokens_per_shard=draw(st.integers(min_value=1, max_value=200_000_000)),
            data=base64.b64encode(
                pickle.dumps(datasets.FakeImg(n_examples=n_examples))
            ).decode("utf8"),
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
        datasets.FakeImg(n_examples=10),
        batch_size=8,
        n_workers=0,
        img_tr=img_tr,
        sample_tr=sample_tr,
    )
    batch = next(iter(dataloader))

    assert isinstance(batch, dict)
    assert "data" in batch
    assert "index" in batch

    torch.testing.assert_close(batch["index"], torch.arange(8))
    assert batch["data"].shape == (8, 3, 224, 224)


@pytest.mark.slow
@pytest.mark.xfail(reason="Don't know how to test end-to-end yet.")
def test_shard_writer_and_dataset_e2e():
    with tmp_shards_root() as shards_root:
        data_cfg = datasets.FakeImg(n_examples=128)

        md = Metadata(
            family="clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            layers=(-2, -1),
            content_tokens_per_example=16,
            cls_token=True,
            d_model=128,
            n_examples=data_cfg.n_examples,
            max_tokens_per_shard=128,
            data={
                **dataclasses.asdict(data_cfg),
                "__class__": data_cfg.__class__.__name__,
            },
            dataset=pathlib.Path("/fake/dataset/path"),
        )
        md.dump(shards_root)

        vit_cls = models.load_model_cls(md.family)
        img_tr, sample_tr = vit_cls.make_transforms(
            md.ckpt, md.content_tokens_per_example
        )
        vit = RecordedTransformer(
            vit_cls(md.ckpt), md.content_tokens_per_example, md.cls_token, md.layers
        )
        dataloader = get_dataloader(
            data_cfg,
            batch_size=8,
            n_workers=8,
            img_tr=img_tr,
            sample_tr=sample_tr,
        )
        writer = ShardWriter(shards_root, md)
        dataset = IndexedDataset(
            IndexedConfig(shards=shards_root / md.hash, patches="cls", layer=-1)
        )

        i = 0
        for b, batch in zip(range(4), dataloader):
            # Don't care about the forward pass.
            out, cache = vit(batch["data"])
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
            content_tokens_per_example=16,
            cls_token=True,
            d_model=128,
            layers=[0],
            data=datasets.FakeImg(n_examples=10),
            batch_size=12,
            n_workers=4,
            max_tokens_per_shard=256,
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
            "content_tokens_per_example",
            "cls_token",
            "d_model",
            "n_examples",
            "max_tokens_per_shard",
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
        assert md["protocol"] == "2.1"

        # data must be base64-encoded pickle object.
        data_bytes = base64.b64decode(md["data"].encode("utf8"))
        pickle.loads(data_bytes)


@settings(deadline=None, max_examples=20)
@given(
    max_tokens_per_shard=st.integers(min_value=1, max_value=10_000_000),
    content_tokens_per_example=st.integers(min_value=1, max_value=1000),
    n_layers=st.integers(min_value=1, max_value=50),
    cls_token=st.booleans(),
)
def test_shard_size_consistency(
    max_tokens_per_shard, content_tokens_per_example, n_layers, cls_token
):
    assume(max_tokens_per_shard >= (content_tokens_per_example + 2) * n_layers)
    md = Metadata(
        family="clip",
        ckpt="ckpt",
        layers=tuple(range(n_layers)),
        content_tokens_per_example=content_tokens_per_example,
        cls_token=cls_token,
        d_model=1,
        n_examples=1,
        max_tokens_per_shard=max_tokens_per_shard,
        data=base64.b64encode(pickle.dumps(datasets.FakeImg(n_examples=1))).decode(
            "utf8"
        ),
        dataset=pathlib.Path("/fake/dataset/path"),
    )
    # compute spec value
    n_tokens = content_tokens_per_example + (1 if cls_token else 0)
    spec_nv = max_tokens_per_shard // (n_tokens * n_layers)
    # via Metadata property
    assert md.examples_per_shard == spec_nv


@pytest.mark.parametrize(
    "content_tokens_per_example,cls_token,expected",
    [
        (49, False, 49),
        (49, True, 50),
        (16, False, 16),
        (16, True, 17),
        (1, False, 1),
        (1, True, 2),
    ],
)
def test_tokens_per_ex(content_tokens_per_example, cls_token, expected):
    md = Metadata(
        family="clip",
        ckpt="ckpt",
        layers=(0,),
        content_tokens_per_example=content_tokens_per_example,
        cls_token=cls_token,
        d_model=512,
        n_examples=1,
        max_tokens_per_shard=1000,
        data=base64.b64encode(pickle.dumps(datasets.FakeImg(n_examples=1))).decode(
            "utf8"
        ),
        dataset=pathlib.Path("/fake/dataset/path"),
    )
    assert md.tokens_per_example == expected


@given(md=metadatas())
def test_metadata_smoke(md):
    """Test that all derived properties and methods on Metadata don't throw exceptions."""
    md.hash
    md.tokens_per_example
    md.n_shards
    md.examples_per_shard
    md.shard_shape
