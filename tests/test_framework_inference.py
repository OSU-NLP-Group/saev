import json

import pytest

import saev.data
import saev.data.datasets
import saev.disk
import saev.framework.inference
import saev.framework.train
import saev.metrics
import saev.nn


def test_img_folder_inference():
    """Tests that image folders (non-segmentation) work with inference.py

    1. Save activations to disk (in a temporary location)
    2. Train an SAE for 0 tokens (we don't care about the weights)
    3. Run the inference script.
    """
    with (
        pytest.helpers.tmp_shards_root() as shards_root,
        pytest.helpers.tmp_runs_root() as runs_root,
    ):
        # Save activations to disk
        shards_dir = pytest.helpers.write_shards(
            shards_root, data=saev.data.datasets.FakeImg(n_examples=32)
        )
        md = saev.data.Metadata.load(shards_dir)

        # Train for 0 tokens.
        train_cfg = saev.framework.train.Config(
            train_data=saev.data.ShuffledConfig(shards=shards_dir, layer=0),
            val_data=saev.data.ShuffledConfig(shards=shards_dir, layer=0),
            n_train=0,
            n_val=0,
            runs_root=runs_root,
            device="cpu",
            sae=saev.nn.SparseAutoencoderConfig(
                d_model=md.d_model, d_sae=md.d_model * 8, reinit_blend=0.0
            ),
        )
        (run_id,) = saev.framework.train.worker_fn([train_cfg])

        inference_cfg = saev.framework.inference.Config(
            run=runs_root / run_id,
            data=saev.data.OrderedConfig(shards=shards_dir, layer=0),
            device="cpu",
        )
        saev.framework.inference.worker_fn(inference_cfg)

        run = saev.disk.Run(runs_root / run_id)
        fpaths = saev.framework.inference.Filepaths.from_run(run, md)

        # Assert all files exist and have non-zero size
        for fpath in fpaths:
            assert fpath.exists()
            assert fpath.stat().st_size > 0

        with open(fpaths.metrics) as fd:
            metrics = json.load(fd)
        saev.metrics.Metrics.from_dict(metrics)

        expected = {
            "mse_per_dim",
            "mse_per_token",
            "normalized_mse",
            "baseline_mse_per_dim",
            "baseline_mse_per_token",
            "sse_recon",
            "sse_baseline",
            "n_tokens",
            "d_model",
            "n_elements",
        }
        assert expected.issubset(metrics.keys())
        assert metrics["n_tokens"] > 0
        assert metrics["d_model"] == md.d_model
        assert metrics["n_elements"] == metrics["n_tokens"] * metrics["d_model"]
        assert metrics["sse_recon"] >= 0.0
        assert metrics["sse_baseline"] > 0.0
        assert metrics["mse_per_dim"] == pytest.approx(
            metrics["sse_recon"] / metrics["n_elements"], rel=1e-6
        )
        assert metrics["mse_per_token"] == pytest.approx(
            metrics["sse_recon"] / metrics["n_tokens"], rel=1e-6
        )
        assert metrics["baseline_mse_per_dim"] == pytest.approx(
            metrics["sse_baseline"] / metrics["n_elements"], rel=1e-6
        )
        assert metrics["baseline_mse_per_token"] == pytest.approx(
            metrics["sse_baseline"] / metrics["n_tokens"], rel=1e-6
        )
        assert metrics["normalized_mse"] == pytest.approx(
            metrics["sse_recon"] / metrics["sse_baseline"], rel=1e-6
        )


def test_metrics_roundtrip():
    metrics = saev.metrics.Metrics.from_accumulators(
        sse_recon=42.0,
        sse_baseline=84.0,
        n_tokens=7,
        d_model=3,
    )
    dct = json.loads(json.dumps(metrics.to_dict()))
    loaded = saev.metrics.Metrics.from_dict(dct)
    assert loaded == metrics


def test_metrics_from_accumulators_accepts_perfect_reconstruction():
    metrics = saev.metrics.Metrics.from_accumulators(
        sse_recon=0.0,
        sse_baseline=84.0,
        n_tokens=7,
        d_model=3,
    )
    assert metrics.normalized_mse == 0.0
    assert metrics.mse_per_dim == 0.0
    assert metrics.mse_per_token == 0.0


@pytest.mark.parametrize(
    ("n_tokens", "d_model", "sse_recon", "sse_baseline"),
    (
        (0, 3, 1.0, 2.0),
        (7, 0, 1.0, 2.0),
        (7, 3, -1.0, 2.0),
        (7, 3, 1.0, 0.0),
        (7, 3, 1.0, -1.0),
    ),
)
def test_metrics_from_accumulators_rejects_invalid_inputs(
    n_tokens: int, d_model: int, sse_recon: float, sse_baseline: float
):
    with pytest.raises(AssertionError):
        saev.metrics.Metrics.from_accumulators(
            sse_recon=sse_recon,
            sse_baseline=sse_baseline,
            n_tokens=n_tokens,
            d_model=d_model,
        )


def test_metrics_from_dict_rejects_bool_for_int_fields():
    metrics = saev.metrics.Metrics.from_accumulators(
        sse_recon=42.0,
        sse_baseline=84.0,
        n_tokens=7,
        d_model=3,
    ).to_dict()
    metrics["n_tokens"] = True

    with pytest.raises(AssertionError):
        saev.metrics.Metrics.from_dict(metrics)


def test_metrics_from_dict_rejects_bool_for_float_fields():
    metrics = saev.metrics.Metrics.from_accumulators(
        sse_recon=42.0,
        sse_baseline=84.0,
        n_tokens=7,
        d_model=3,
    ).to_dict()
    metrics["sse_recon"] = True

    with pytest.raises(AssertionError):
        saev.metrics.Metrics.from_dict(metrics)
