# tests/test_iterable_dataloader.py
import contextlib
import dataclasses
import gc
import json
import os
import pathlib
import queue
import tempfile
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


@pytest.mark.slow
def test_min_buffer_fill_handles_small_dataset():
    with tmp_shards_root() as shards_root:
        data_cfg = datasets.FakeImg(n_examples=2)
        shards_dir = saev.data.shards.worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-2],
            data=data_cfg,
            batch_size=2,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
        )

        md = saev.data.Metadata.load(shards_dir)
        cfg = ShuffledConfig(
            shards=shards_dir,
            tokens="content",
            layer=md.layers[0],
            debug=True,
            log_every_s=1.0,
            batch_size=16,
            min_buffer_fill=0.5,
        )

        dl = ShuffledDataLoader(cfg)
        seen = 0
        for batch in dl:
            seen += len(batch["example_idx"])
            assert batch["act"].shape[0] == cfg.batch_size
            break

        assert seen == cfg.batch_size


@pytest.mark.slow
def test_min_buffer_fill_with_batch_limiter():
    import saev.utils.scheduling

    with tmp_shards_root() as shards_root:
        data_cfg = datasets.FakeImg(n_examples=4)
        shards_dir = saev.data.shards.worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-2],
            data=data_cfg,
            batch_size=2,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
        )

        md = saev.data.Metadata.load(shards_dir)
        cfg = ShuffledConfig(
            shards=shards_dir,
            tokens="content",
            layer=md.layers[0],
            debug=True,
            log_every_s=1.0,
            batch_size=16,
            min_buffer_fill=0.2,
        )

        dl = ShuffledDataLoader(cfg)
        dl = saev.utils.scheduling.BatchLimiter(dl, 80)
        for batch in dl:
            assert batch is not None


@pytest.mark.slow
def test_min_buffer_fill_allows_epoch_restart_with_batch_limiter():
    import saev.utils.scheduling

    with tmp_shards_root() as shards_root:
        data_cfg = datasets.FakeImg(n_examples=12)
        shards_dir = saev.data.shards.worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-2],
            data=data_cfg,
            batch_size=4,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
        )

        md = saev.data.Metadata.load(shards_dir)
        cfg = ShuffledConfig(
            shards=shards_dir,
            tokens="content",
            layer=md.layers[0],
            debug=True,
            log_every_s=1.0,
            batch_size=32,
            buffer_size=8,
            min_buffer_fill=0.5,
        )

        dl = ShuffledDataLoader(cfg)
        limit = 2 * data_cfg.n_examples * md.content_tokens_per_example
        dl = saev.utils.scheduling.BatchLimiter(dl, limit)

        seen = 0
        for batch in dl:
            seen += len(batch["example_idx"])

        dataset_tokens = data_cfg.n_examples * md.content_tokens_per_example
        assert seen >= limit
        assert seen > dataset_tokens
        assert seen < limit * 3


@pytest.mark.slow
def test_min_buffer_fill_handles_manager_shutdown_with_buffered_tokens():
    import saev.utils.scheduling

    with tmp_shards_root() as shards_root:
        data_cfg = datasets.FakeImg(n_examples=8)
        shards_dir = saev.data.shards.worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-2],
            data=data_cfg,
            batch_size=4,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
        )

        md = saev.data.Metadata.load(shards_dir)
        cfg = ShuffledConfig(
            shards=shards_dir,
            tokens="content",
            layer=md.layers[0],
            debug=True,
            log_every_s=1.0,
            batch_size=32,
            buffer_size=64,
            min_buffer_fill=0.5,
        )

        dl = ShuffledDataLoader(cfg)
        limit = data_cfg.n_examples * md.content_tokens_per_example + 64
        dl = saev.utils.scheduling.BatchLimiter(dl, limit)

        seen = 0
        for batch in dl:
            seen += len(batch["example_idx"])

        assert seen >= limit


class _DeadProc:
    def is_alive(self) -> bool:  # pragma: no cover - trivial shim
        return False


class _FakeReservoir:
    def __init__(self, *, capacity: int, qsize: int):
        self.capacity = capacity
        self._qsize = qsize

    def qsize(self) -> int:
        return self._qsize

    def close(self) -> None:  # pragma: no cover - used in __del__
        pass


@pytest.mark.slow
def test_min_buffer_fill_waits_even_when_manager_finished():
    with tmp_shards_root() as shards_root:
        data_cfg = datasets.FakeImg(n_examples=2)
        shards_dir = saev.data.shards.worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-2],
            data=data_cfg,
            batch_size=2,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
        )

        md = saev.data.Metadata.load(shards_dir)
        cfg = ShuffledConfig(
            shards=shards_dir,
            tokens="content",
            layer=md.layers[0],
            debug=True,
            log_every_s=1.0,
            batch_size=32,
            buffer_size=64,
            min_buffer_fill=0.5,
        )

        dl = ShuffledDataLoader(cfg)
        dl.reservoir = _FakeReservoir(
            capacity=cfg.buffer_size * cfg.batch_size, qsize=cfg.batch_size
        )
        dl.manager_proc = _DeadProc()
        dl.err_queue = queue.Queue()

        dl._wait_for_min_buffer_fill(cfg.batch_size)


@pytest.mark.slow
def test_min_buffer_fill_allows_epoch_restart_without_runtime_error():
    import saev.utils.scheduling

    with tmp_shards_root() as shards_root:
        data_cfg = datasets.FakeImg(n_examples=8)
        shards_dir = saev.data.shards.worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-2],
            data=data_cfg,
            batch_size=4,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
        )

        md = saev.data.Metadata.load(shards_dir)
        cfg = ShuffledConfig(
            shards=shards_dir,
            tokens="content",
            layer=md.layers[0],
            debug=True,
            log_every_s=1.0,
            batch_size=32,
            buffer_size=64,
            min_buffer_fill=0.5,
        )

        dl = ShuffledDataLoader(cfg)
        limit = data_cfg.n_examples * md.content_tokens_per_example + 64
        dl = saev.utils.scheduling.BatchLimiter(dl, limit)

        seen = 0
        for batch in dl:
            seen += len(batch["example_idx"])

        assert seen >= limit


@pytest.mark.slow
def test_min_buffer_fill_manager_finishes_with_backlog():
    with tmp_shards_root() as shards_root:
        data_cfg = datasets.FakeImg(n_examples=16)
        shards_dir = saev.data.shards.worker_fn(
            family="fake-clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            content_tokens_per_example=16,
            cls_token=False,
            d_model=128,
            layers=[-2],
            data=data_cfg,
            batch_size=4,
            n_workers=0,
            max_tokens_per_shard=256,
            shards_root=shards_root,
            device="cpu",
        )

        md = saev.data.Metadata.load(shards_dir)
        cfg = ShuffledConfig(
            shards=shards_dir,
            tokens="content",
            layer=md.layers[0],
            debug=True,
            log_every_s=0.1,
            batch_size=32,
            buffer_size=64,
            min_buffer_fill=0.5,
        )

        dl = ShuffledDataLoader(cfg)
        try:
            dl._start_manager()

            target_fill = cfg.batch_size
            timeout = time.time() + 10.0
            while dl.reservoir.qsize() < target_fill:
                if not dl.manager_proc.is_alive():
                    break
                assert time.time() < timeout, "Reservoir never filled"
                time.sleep(0.01)

            assert dl.reservoir.qsize() > 0, "Reservoir stayed empty"

            dl.stop_event.set()
            dl.manager_proc.join(timeout=5.0)

            dl._wait_for_min_buffer_fill(cfg.batch_size)
        finally:
            dl.shutdown()
