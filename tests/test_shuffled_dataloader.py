# tests/test_iterable_dataloader.py
import contextlib
import dataclasses
import gc
import json
import os
import pathlib
import tempfile
import time

import beartype
import psutil
import pytest
import torch.multiprocessing as mp

import saev.data
from saev.data import ShuffledConfig, ShuffledDataLoader, datasets, shards

mp.set_start_method("spawn", force=True)


@pytest.fixture(scope="session")
def cfg(pytestconfig):
    shards = pytestconfig.getoption("--shards")
    if shards is None:
        pytest.skip("--shards not supplied")

    shards = pathlib.Path(shards)

    metadata = saev.data.Metadata.load(shards)
    layer = metadata.layers[0]
    cfg = ShuffledConfig(
        shards=shards, patches="image", layer=layer, debug=True, log_every_s=1.0
    )

    return cfg


@contextlib.contextmanager
def tmp_shards_root():
    """Create a temporary shard root directory."""
    # We cannot use the tmp_path fixture because of Hypothesis.
    # See https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck.function_scoped_fixture
    with tempfile.TemporaryDirectory() as tmp_path:
        shards_root = pathlib.Path(tmp_path) / "saev" / "shards"
        shards_root.mkdir(parents=True)
        yield shards_root


@beartype.beartype
def _global_index(ex_i: int, patch_i: int, n_patches: int) -> int:
    """Map (ex_i, patch_i) to linear index used by Dataset when cfg.patches == "patches" and cfg.layer is fixed."""
    return ex_i * n_patches + patch_i


def test_init_smoke(cfg):
    ShuffledDataLoader(cfg)


def test_len_smoke(cfg):
    dl = ShuffledDataLoader(cfg)
    assert isinstance(len(dl), int)


def test_iter_smoke(cfg):
    dl = ShuffledDataLoader(cfg)
    # simply iterating one element should succeed
    batch = next(iter(dl))
    assert "act" in batch and "ex_i" in batch and "patch_i" in batch


def test_batches(cfg):
    dl = ShuffledDataLoader(cfg)
    it = iter(dl)
    for _ in range(8):
        batch = next(it)
        assert "act" in batch and "ex_i" in batch and "patch_i" in batch


@pytest.mark.parametrize("bs", [4, 8, 16, 24])
def test_batch_size_matches(cfg, bs):
    cfg = dataclasses.replace(cfg, batch_size=bs)
    dl = ShuffledDataLoader(cfg)
    it = iter(dl)
    for _ in range(4):
        batch = next(it)
        assert batch["act"].shape[0] == bs
        assert batch["ex_i"].shape[0] == bs
        assert batch["patch_i"].shape[0] == bs


def peak_children():
    """Return set(pid) of live child processes."""
    return {p.pid: p.name() for p in psutil.Process().children(recursive=True)}


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
def test_missing_shard_file_not_detected_at_init(tmp_path):
    """Test that missing shard files are NOT detected at initialization - exposes the validation gap."""
    with tmp_shards_root() as shards_root:
        # Create a small dataset with multiple shards
        n_ex = 10
        d_model = 128
        n_patches = 16
        layers = [0]

        # Use small max_patches_per_shard to force multiple shards
        # Each image has 17 tokens (16 patches + 1 CLS), so with 2 images per shard we get 34 patches per shard
        patches_per_shard = 34  # This will create ~5 shards for 10 images

        # Generate the activation shards
        shards_dir = shards.worker_fn(
            family="clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            patches_per_ex=n_patches,
            cls_token=True,
            shards_root=shards_root,
            d_model=d_model,
            layers=layers,
            patches_per_shard=patches_per_shard,
            batch_size=2,
            data=datasets.Fake(n_ex=n_ex),
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
                shard_root=shard_root, patches="image", layer=layers[0]
            )
            ShuffledDataLoader(cfg)
