import contextlib
import pathlib
import tempfile
import uuid

import hypothesis.strategies as st
import torch
from hypothesis import HealthCheck, given, settings
from tdiscovery.baselines import (
    BaselineRun,
    MiniBatchKMeans,
    TrainConfig,
    dump,
    load,
)

import saev.data
import saev.helpers


def _make_train_config(
    root: pathlib.Path,
) -> tuple[TrainConfig, pathlib.Path, pathlib.Path]:
    train_shards = root / "train-shards"
    val_shards = root / "val-shards"
    train_shards.mkdir(parents=True, exist_ok=True)
    val_shards.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(
        method="kmeans",
        k=4,
        device="cpu",
        collapse_tol=0.25,
        runs_root=root / "runs",
        train_data=saev.data.ShuffledConfig(shards=train_shards),
        val_data=saev.data.ShuffledConfig(shards=val_shards),
    )
    return cfg, train_shards, val_shards


def _make_model(cfg: TrainConfig) -> MiniBatchKMeans:
    model = MiniBatchKMeans(k=cfg.k, device="cpu", collapse_tol=cfg.collapse_tol)
    centers = torch.randn(cfg.k, 3)
    counts = torch.arange(1, cfg.k + 1, dtype=torch.float32)
    model.cluster_centers_ = centers
    model.cluster_counts_ = counts
    model.n_features_in_ = centers.shape[1]
    model.n_steps_ = 7
    return model


@contextlib.contextmanager
def _tmp_baseline_root():
    """Temporary directory helper suitable for Hypothesis (cf. tmp_shards_root)."""
    with tempfile.TemporaryDirectory() as tmp:
        yield pathlib.Path(tmp)


def test_dump_and_load_round_trip(tmp_path):
    cfg, train_shards, val_shards = _make_train_config(tmp_path)
    run = BaselineRun.new(
        "round-trip",
        train_shards_dir=train_shards,
        val_shards_dir=val_shards,
        runs_root=cfg.runs_root,
    )
    model = _make_model(cfg)

    ckpt_path = dump(run.ckpt_dir, model=model)
    assert ckpt_path.exists()

    loaded = load(run, device="cpu")
    assert isinstance(loaded, MiniBatchKMeans)
    assert loaded.cluster_counts_ is not None
    assert loaded.cluster_centers_ is not None
    assert torch.equal(loaded.cluster_counts_, model.cluster_counts_)
    assert torch.equal(loaded.cluster_centers_, model.cluster_centers_)
    assert loaded.collapse_tol == cfg.collapse_tol


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    k=st.integers(min_value=1, max_value=8),
    d_model=st.integers(min_value=1, max_value=6),
    steps=st.integers(min_value=0, max_value=64),
    collapse=st.floats(
        min_value=0.05, max_value=1.5, allow_nan=False, allow_infinity=False
    ),
)
def test_dump_and_load_round_trip_property(k, d_model, steps, collapse):
    with _tmp_baseline_root() as root:
        cfg, train_shards, val_shards = _make_train_config(root)
        run = BaselineRun.new(
            f"hypothesis-{uuid.uuid4().hex}",
            train_shards_dir=train_shards,
            val_shards_dir=val_shards,
            runs_root=cfg.runs_root,
        )

        model = MiniBatchKMeans(k=k, device="cpu", collapse_tol=collapse)
        centers = torch.randn(k, d_model)
        counts = torch.rand(k) + 1e-3
        model.cluster_centers_ = centers
        model.cluster_counts_ = counts
        model.n_features_in_ = d_model
        model.n_steps_ = steps

        dump(run.ckpt_dir, model=model)
        loaded = load(run, device="cpu")

        assert isinstance(loaded, MiniBatchKMeans)
        assert loaded.cluster_counts_ is not None
        assert loaded.cluster_centers_ is not None
        assert loaded.k == k
        assert loaded.n_features_in_ == d_model
        assert loaded.n_steps_ == steps
        assert loaded.collapse_tol == collapse
        assert torch.equal(loaded.cluster_counts_, model.cluster_counts_)
        assert torch.equal(loaded.cluster_centers_, model.cluster_centers_)
