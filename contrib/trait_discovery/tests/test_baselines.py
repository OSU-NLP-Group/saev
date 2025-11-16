import contextlib
import dataclasses
import pathlib
import tempfile
import uuid

import hypothesis.strategies as st
import orjson
import torch
from hypothesis import HealthCheck, given, settings
from tdiscovery.baselines import MiniBatchKMeans, MiniBatchPCA, TrainConfig, dump, load

import saev.data
import saev.disk
import saev.helpers


def _make_train_config(
    root: pathlib.Path,
) -> tuple[TrainConfig, pathlib.Path, pathlib.Path]:
    train_shards = root / "train-shards"
    val_shards = root / "val-shards"
    train_shards.mkdir(parents=True, exist_ok=True)
    val_shards.mkdir(parents=True, exist_ok=True)
    runs_root = root / "saev" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(
        method="kmeans",
        k=4,
        device="cpu",
        collapse_tol=0.25,
        runs_root=runs_root,
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


def _make_pca_model(k: int, d_model: int) -> MiniBatchPCA:
    model = MiniBatchPCA(n_components=k, device="cpu")
    q, _ = torch.linalg.qr(torch.randn(d_model, d_model))
    model.components_ = q[:, :k].T.contiguous()
    model.mean_ = torch.randn(d_model)
    model.scatter_ = torch.eye(d_model)
    model.explained_variance_ = torch.linspace(1.0, 2.0, k)
    model.n_features_in_ = d_model
    model.n_samples_seen_ = 32
    model.n_steps_ = 4
    model.total_variance_ = float(model.explained_variance_.sum().item())
    model.last_batch_recon_error_ = 0.01
    model.last_batch_var_ratio_ = 0.95
    return model


@contextlib.contextmanager
def _tmp_baseline_root():
    """Temporary directory helper suitable for Hypothesis (cf. tmp_shards_root)."""
    with tempfile.TemporaryDirectory() as tmp:
        yield pathlib.Path(tmp)


def test_dump_and_load_round_trip(tmp_path):
    cfg, train_shards, val_shards = _make_train_config(tmp_path)
    run = saev.disk.Run.new(
        "round-trip",
        train_shards_dir=train_shards,
        val_shards_dir=val_shards,
        runs_root=cfg.runs_root,
    )
    model = _make_model(cfg)

    ckpt_path = dump(run, cfg, model=model)
    assert ckpt_path.exists()

    loaded = load(run, device="cpu")
    assert isinstance(loaded, MiniBatchKMeans)
    assert loaded.cluster_counts_ is not None
    assert loaded.cluster_centers_ is not None
    assert torch.equal(loaded.cluster_counts_, model.cluster_counts_)
    assert torch.equal(loaded.cluster_centers_, model.cluster_centers_)
    assert loaded.collapse_tol == cfg.collapse_tol

    cfg_path = run.ckpt.parent / "config.json"
    assert cfg_path.exists()
    stored_cfg = orjson.loads(cfg_path.read_bytes())
    expected_cfg = orjson.loads(saev.helpers.jdumps(dataclasses.asdict(cfg)))
    assert stored_cfg == expected_cfg


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
        run = saev.disk.Run.new(
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

        dump(run, cfg, model=model)
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

        cfg_path = run.ckpt.parent / "config.json"
        assert cfg_path.exists()
        stored_cfg = orjson.loads(cfg_path.read_bytes())
        expected_cfg = orjson.loads(saev.helpers.jdumps(dataclasses.asdict(cfg)))
        assert stored_cfg == expected_cfg


def test_dump_and_load_round_trip_pca(tmp_path):
    cfg, train_shards, val_shards = _make_train_config(tmp_path)
    cfg = dataclasses.replace(cfg, method="pca", k=3)
    run = saev.disk.Run.new(
        "pca-round-trip",
        train_shards_dir=train_shards,
        val_shards_dir=val_shards,
        runs_root=cfg.runs_root,
    )

    model = _make_pca_model(cfg.k, d_model=5)
    dump(run, cfg, model=model)
    loaded = load(run, device="cpu")

    assert isinstance(loaded, MiniBatchPCA)
    assert loaded.components_ is not None
    assert loaded.mean_ is not None
    assert torch.allclose(loaded.components_, model.components_)
    assert torch.allclose(loaded.mean_, model.mean_)
    assert torch.equal(loaded.explained_variance_, model.explained_variance_)
    assert loaded.n_samples_seen_ == model.n_samples_seen_
    assert loaded.n_steps_ == model.n_steps_
