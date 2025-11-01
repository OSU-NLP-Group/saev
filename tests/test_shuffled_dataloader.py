# tests/test_iterable_dataloader.py
import dataclasses
import gc
import json
import os
import time

import beartype
import psutil
import pytest
import torch
import torch.multiprocessing as mp

import saev.data
from saev.data import ShuffledConfig, ShuffledDataLoader, datasets, shards

mp.set_start_method("spawn", force=True)


@pytest.fixture(scope="session")
def cfg(shards_dir):
    metadata = saev.data.Metadata.load(shards_dir)
    layer = metadata.layers[0]
    cfg = ShuffledConfig(
        shards=shards_dir, tokens="content", layer=layer, debug=True, log_every_s=1.0
    )

    return cfg


@beartype.beartype
def _global_index(example_idx: int, token_idx: int, n_patches: int) -> int:
    """Map (example_idx, token_idx) to linear index used by Dataset when cfg.patches == "patches" and cfg.layer is fixed."""
    return example_idx * n_patches + token_idx


def test_init_smoke(cfg):
    ShuffledDataLoader(cfg)


def test_len_smoke(cfg):
    dl = ShuffledDataLoader(cfg)
    assert isinstance(len(dl), int)


def test_iter_smoke(cfg):
    dl = ShuffledDataLoader(cfg)
    # simply iterating one element should succeed
    batch = next(iter(dl))
    assert "act" in batch
    assert "example_idx" in batch
    assert "token_idx" in batch


def test_batches(cfg):
    dl = ShuffledDataLoader(cfg)
    it = iter(dl)
    for _ in range(8):
        batch = next(it)
        assert "act" in batch
        assert "example_idx" in batch
        assert "token_idx" in batch


@pytest.mark.parametrize("bs", [4, 8, 16, 24])
def test_batch_size_matches(cfg, bs):
    cfg = dataclasses.replace(cfg, batch_size=bs)
    dl = ShuffledDataLoader(cfg)
    it = iter(dl)
    for _ in range(4):
        batch = next(it)
        assert batch["act"].shape[0] == bs
        assert batch["example_idx"].shape[0] == bs
        assert batch["token_idx"].shape[0] == bs


def peak_children():
    """Return set(pid) of live child processes."""
    return {p.pid: p.name() for p in psutil.Process().children(recursive=True)}


@pytest.mark.xfail()
def test_no_child_leak(cfg):
    """Loader must clean up its workers after iteration terminates."""
    before = peak_children()

    dl = ShuffledDataLoader(cfg)

    for _ in range(2):  # minimal work
        next(iter(dl))

    if hasattr(dl, "shutdown"):
        dl.shutdown()  # explicit close
    del dl
    gc.collect()
    time.sleep(5.0)  # give OS a tick
    gc.collect()

    after = peak_children()
    assert set(after.keys()).issubset(set(before.keys()))  # no new zombies


@pytest.mark.slow
@pytest.mark.xfail(reason="Not implemented.")
def test_missing_shard_file_not_detected_at_init(tmp_path):
    """Test that missing shard files are NOT detected at initialization. This exposes the validation gap."""
    with pytest.helpers.tmp_shards_root() as shards_root:
        # Create a small dataset with multiple shards
        n_examples = 10
        d_model = 128
        content_tokens_per_example = 16
        layers = [0]

        # Use small max_tokens_per_shard to force multiple shards
        # Each image has 17 tokens (16 patches + 1 CLS), so with 2 images per shard we get 34 patches per shard
        max_tokens_per_shard = 34  # This will create ~5 shards for 10 images

        # Generate the activation shards
        shards_dir = shards.worker_fn(
            family="clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=content_tokens_per_example,
            cls_token=True,
            shards_root=shards_root,
            d_model=d_model,
            layers=layers,
            max_tokens_per_shard=max_tokens_per_shard,
            batch_size=2,
            data=datasets.FakeImg(n_examples=n_examples),
            n_workers=0,
            device="cpu",
        )

        # Get the actual shard directory
        metadata = shards.Metadata.load(shards_dir)
        shard_root = os.path.join(str(tmp_path), metadata.hash)

        # Verify we have multiple shards
        shard_files = [f for f in os.listdir(shard_root) if f.endswith(".bin")]
        assert len(shard_files) > 1, f"Expected multiple shards, got {len(shard_files)}"

        # Delete one of the middle shard files (not the first one)
        missing_shard = "acts000001.bin"
        missing_file_path = os.path.join(shard_root, missing_shard)
        assert os.path.exists(missing_file_path), (
            f"Shard file {missing_shard} should exist before deletion"
        )
        os.remove(missing_file_path)
        assert not os.path.exists(missing_file_path), (
            f"Shard file {missing_shard} should be deleted"
        )

        # Verify shards.json still lists the deleted file
        with open(os.path.join(shard_root, "shards.json")) as fd:
            shards_data = json.load(fd)
        shard_names = [s["name"] for s in shards_data]
        assert missing_shard in shard_names, (
            f"shards.json should still list {missing_shard}"
        )

        # Create shuffled dataloader. This should raise an error at initialization because missing files should be detected early
        with pytest.raises(FileNotFoundError):
            cfg = ShuffledConfig(
                shard_root=shard_root, tokens="content", layer=layers[0]
            )
            ShuffledDataLoader(cfg)


def _token_coverage(batch, content_tokens_per_example: int) -> float:
    unique_tokens = torch.unique(batch["token_idx"]).numel()
    return unique_tokens / content_tokens_per_example


@pytest.mark.slow
def test_min_buffer_fill_default_allows_low_coverage(cfg):
    dl = ShuffledDataLoader(cfg)
    try:
        batch = next(iter(dl))
        coverage = _token_coverage(batch, dl.metadata.content_tokens_per_example)
        last_fill = dl._last_reservoir_fill
    finally:
        dl.shutdown()

    assert coverage < 0.15
    assert last_fill is None


@pytest.mark.slow
def test_min_buffer_fill_warmup_improves_coverage(cfg):
    cfg = dataclasses.replace(cfg, min_buffer_fill=0.2)
    dl = ShuffledDataLoader(cfg)
    try:
        batch = next(iter(dl))
        coverage = _token_coverage(batch, dl.metadata.content_tokens_per_example)
    finally:
        dl.shutdown()

    assert coverage >= 0.8
