import pytest

from saev.data import datasets, shards


@pytest.fixture(scope="session")
def custom_shards_dir(request):
    n_examples = 8
    layers = [0, 1]
    max_tokens_per_shard = 128
    with pytest.helpers.tmp_shards_root() as shards_root:
        yield shards.worker_fn(
            data=datasets.FakeImg(n_examples=n_examples),
            family="clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            d_model=128,
            layers=layers,
            content_tokens_per_example=16,
            cls_token=True,  # This model has a [CLS] token
            max_tokens_per_shard=max_tokens_per_shard,
            batch_size=4,
            n_workers=0,
            device="cpu",
            shards_root=shards_root,
        )


def test_content_tokens_with_cls_token(custom_shards_dir):
    md = shards.Metadata.load(custom_shards_dir)
    index_map = shards.IndexMap(md, "content", 0)

    idx = 0
    index = index_map.from_global(idx)
    assert index.idx == idx
    assert index.example_idx == 0
    assert index.content_token_idx == 0
    assert index.shard_idx == 0
    assert index.example_idx_in_shard == 0
    assert index.layer_idx_in_shard == 0
    assert index.token_idx_in_shard == 1

    idx = 15
    index = index_map.from_global(idx)
    assert index.idx == idx
    assert index.example_idx == 0
    assert index.content_token_idx == 15
    assert index.shard_idx == 0
    assert index.example_idx_in_shard == 0
    assert index.layer_idx_in_shard == 0
    assert index.token_idx_in_shard == 16

    idx = 16
    index = index_map.from_global(idx)
    assert index.idx == idx
    assert index.example_idx == 1
    assert index.content_token_idx == 0
    assert index.shard_idx == 0
    assert index.example_idx_in_shard == 1
    assert index.layer_idx_in_shard == 0
    assert index.token_idx_in_shard == 1

    idx = 31
    index = index_map.from_global(idx)
    assert index.idx == idx
    assert index.example_idx == 1
    assert index.content_token_idx == 15
    assert index.shard_idx == 0
    assert index.example_idx_in_shard == 1
    assert index.layer_idx_in_shard == 0
    assert index.token_idx_in_shard == 16

    idx = 48
    index = index_map.from_global(idx)
    assert index.idx == idx
    assert index.example_idx == 3
    assert index.content_token_idx == 0
    assert index.shard_idx == 1
    assert index.example_idx_in_shard == 0
    assert index.layer_idx_in_shard == 0
    assert index.token_idx_in_shard == 1


def test_special_tokens_with_cls_token(custom_shards_dir):
    md = shards.Metadata.load(custom_shards_dir)
    index_map = shards.IndexMap(md, "special", 0)

    idx = 0
    index = index_map.from_global(idx)
    assert index.idx == idx
    assert index.example_idx == 0
    assert index.content_token_idx == -1
    assert index.shard_idx == 0
    assert index.example_idx_in_shard == 0
    assert index.layer_idx_in_shard == 0
    assert index.token_idx_in_shard == 0
