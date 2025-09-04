# tests/test_iterable_dataloader.py
import dataclasses
import gc
import json
import os
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
    assert set(after.keys()).issubset(set(before.keys()))  # no new zombies


@pytest.mark.slow
def test_missing_shard_file_not_detected_at_init(tmp_path):
    """Test that missing shard files are NOT detected at initialization - exposes the validation gap."""
    from saev.data import datasets, writers

    # Create a small dataset with multiple shards
    n_imgs = 10
    d_vit = 128
    n_patches = 16
    layers = [0]

    # Use small max_patches_per_shard to force multiple shards
    # Each image has 17 tokens (16 patches + 1 CLS), so with 2 images per shard we get 34 patches per shard
    max_patches_per_shard = 34  # This will create ~5 shards for 10 images

    # Create activation shards
    cfg = writers.Config(
        data=datasets.Fake(n_imgs=n_imgs),
        dump_to=str(tmp_path),
        vit_family="clip",
        vit_ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
        d_vit=d_vit,
        vit_layers=layers,
        n_patches_per_img=n_patches,
        cls_token=True,
        max_patches_per_shard=max_patches_per_shard,
        vit_batch_size=2,
        n_workers=0,
        device="cpu",
    )

    # Generate the activation shards
    writers.worker_fn(cfg)

    # Get the actual shard directory
    metadata = writers.Metadata.from_cfg(cfg)
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
        cfg = IterableConfig(shard_root=shard_root, patches="image", layer=layers[0])
        DataLoader(cfg)
