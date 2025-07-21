# tests/test_iterable_dataloader.py
import dataclasses
import gc
import time

import psutil
import pytest
import torch.multiprocessing as mp

import saev.data
from saev.data.shuffled import Config as IterableConfig
from saev.data.shuffled import DataLoader

mp.set_start_method("spawn", force=True)

N_SAMPLES = 25_000  # quick but representative
BATCH_SIZE = 4_096


@pytest.fixture(scope="session")
def iterable_cfg(pytestconfig):
    shards = pytestconfig.getoption("--shards")
    if shards is None:
        pytest.skip("--shards not supplied")
    metadata = saev.data.Metadata.load(shards)
    layer = metadata.layers[0]
    cfg = IterableConfig(shard_root=shards, patches="image", layer=layer)
    return cfg


def _global_index(img_i: int, patch_i: int, n_patches: int) -> int:
    """
    Map (image_i, patch_i) to linear index used by Dataset when
    cfg.patches == "patches" and cfg.layer is fixed.
    """
    return img_i * n_patches + patch_i


def test_init_smoke(iterable_cfg):
    DataLoader(iterable_cfg)


def test_len_smoke(iterable_cfg):
    dl = DataLoader(iterable_cfg)
    assert isinstance(len(dl), int)


def test_iter_smoke(iterable_cfg):
    dl = DataLoader(iterable_cfg)
    # simply iterating one element should succeed
    batch = next(iter(dl))
    assert "act" in batch and "image_i" in batch and "patch_i" in batch


def test_batches(iterable_cfg):
    dl = DataLoader(iterable_cfg)
    it = iter(dl)
    for _ in range(8):
        batch = next(it)
        assert "act" in batch and "image_i" in batch and "patch_i" in batch


@pytest.mark.parametrize("bs", [8, 32, 128, 512, 2048])
def test_batch_size_matches(iterable_cfg, bs):
    cfg = dataclasses.replace(iterable_cfg, batch_size=bs)
    dl = DataLoader(cfg)
    it = iter(dl)
    for _ in range(4):
        batch = next(it)
        assert batch["act"].shape[0] == bs
        assert batch["image_i"].shape[0] == bs
        assert batch["patch_i"].shape[0] == bs


def peak_children():
    """Return set(pid) of live child processes."""
    return {p.pid: p.name() for p in psutil.Process().children(recursive=True)}


def test_no_child_leak(iterable_cfg):
    """Loader must clean up its workers after iteration terminates."""
    before = peak_children()

    dl = DataLoader(iterable_cfg)

    for _ in range(2):  # minimal work
        next(iter(dl))

    if hasattr(dl, "shutdown"):
        dl.shutdown()  # explicit close
    del dl
    gc.collect()
    time.sleep(5.0)  # give OS a tick
    gc.collect()

    after = peak_children()
    assert after <= before  # no new zombies
