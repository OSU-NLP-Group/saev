"""Unit tests for 1D probe training."""

import logging

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from tdiscovery.probe1d import Sparse1DProbe

cuda_available = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires GPU"
)


def test_fit_smoke():
    """Test that optimizer converges on linearly separable data with L2 regularization."""
    torch.manual_seed(42)
    n_samples = 128
    n_latents, n_classes = 5, 3

    # Generate linearly separable data
    x = torch.randn(n_samples, n_latents).to_sparse_csr()
    true_w = torch.randn((n_latents, n_classes))
    true_b = torch.randn((1, n_classes))
    y = ((x @ true_w + true_b) > 0).float()

    # Initialize optimizer
    clf = Sparse1DProbe(
        n_latents=n_latents, n_classes=n_classes, device="cpu", row_batch_size=4
    )

    clf.fit(x, y)
    clf.loss_matrix(x, y)


@pytest.mark.parametrize("seed", range(5))
def test_fit_against_sklearn(seed):
    torch.manual_seed(seed)
    n_samples, n_latents, n_classes = 64, 8, 4
    x = torch.randn(n_samples, n_latents)
    # make it sparse k=3 per sample
    topk = torch.topk(x.abs(), k=3, dim=1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(1, topk.indices, True)
    x[~mask] = 0

    # one "true" latent per class + noise
    true_w = torch.zeros(n_latents, n_classes)
    for c in range(n_classes):
        true_w[c, c] = torch.randn(())
    true_b = torch.randn((1, n_classes))
    logits = (x @ true_w) + true_b
    y = torch.bernoulli(torch.sigmoid(logits))

    loss_ref = np.zeros((n_latents, n_classes))
    for i in range(n_latents):
        xi = x[:, i : i + 1].numpy()
        for c in range(n_classes):
            yc = y[:, c].numpy()

            # Use very large C to effectively disable regularization
            lr = LogisticRegression(
                fit_intercept=True, solver="lbfgs", C=1e10, max_iter=100
            )
            lr.fit(xi, yc)

            # compute mean NLL on (xi, yc)
            z = lr.intercept_[0] + lr.coef_[0, 0] * xi.squeeze()
            mu = 1 / (1 + np.exp(-z))
            loss_ref[i, c] = -(yc * np.log(mu) + (1 - yc) * np.log(1 - mu)).mean()

    # Use very small ridge to effectively disable regularization (matching sklearn)
    probe = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cpu",
        ridge=1e-10,
        row_batch_size=4,
    )
    probe.fit(x.to_sparse_csr(), y)
    loss_sparse = probe.loss_matrix(x.to_sparse_csr(), y)
    torch.testing.assert_close(
        loss_sparse, torch.tensor(loss_ref, dtype=torch.float32), rtol=1e-4, atol=1e-4
    )


@cuda_available
@pytest.mark.parametrize("seed", range(5))
def test_fit_against_sklearn_on_gpu(seed):
    """Test that our GPU implementation matches sklearn's CPU implementation."""
    torch.manual_seed(seed)
    n_samples, n_latents, n_classes = 64, 8, 4
    x = torch.randn(n_samples, n_latents)
    # make it sparse k=3 per sample
    topk = torch.topk(x.abs(), k=3, dim=1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(1, topk.indices, True)
    x[~mask] = 0

    # one "true" latent per class + noise
    true_w = torch.zeros(n_latents, n_classes)
    for c in range(n_classes):
        true_w[c, c] = torch.randn(())
    true_b = torch.randn((1, n_classes))
    logits = (x @ true_w) + true_b
    y = torch.bernoulli(torch.sigmoid(logits))

    loss_ref = np.zeros((n_latents, n_classes))
    for i in range(n_latents):
        xi = x[:, i : i + 1].numpy()
        for c in range(n_classes):
            yc = y[:, c].numpy()

            # Use very large C to effectively disable regularization
            lr = LogisticRegression(
                fit_intercept=True, solver="lbfgs", C=1e10, max_iter=100
            )
            lr.fit(xi, yc)

            # compute mean NLL on (xi, yc)
            z = lr.intercept_[0] + lr.coef_[0, 0] * xi.squeeze()
            mu = 1 / (1 + np.exp(-z))
            loss_ref[i, c] = -(yc * np.log(mu) + (1 - yc) * np.log(1 - mu)).mean()

    # Use very small ridge to effectively disable regularization (matching sklearn)
    # Run on GPU this time
    probe = Sparse1DProbe(
        n_latents=n_latents, n_classes=n_classes, device="cuda:0", ridge=1e-10
    )
    probe.fit(x.to_sparse_csr(), y)

    # Move sparse matrix to GPU for loss computation
    x_gpu = x.to_sparse_csr().to("cuda:0")
    y_gpu = y.to("cuda:0")
    loss_sparse = probe.loss_matrix(x_gpu, y_gpu)

    # Compare results (move back to CPU for comparison)
    torch.testing.assert_close(
        loss_sparse.cpu(),
        torch.tensor(loss_ref, dtype=torch.float32),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("class_slab_size", [2, 4, 6])
def test_chunked_classes_vs_full(seed, class_slab_size):
    """Verify that processing classes in chunks gives same results as processing all at once."""
    torch.manual_seed(seed)
    n_samples, n_latents, n_classes = 32, 8, 6
    x = torch.randn(n_samples, n_latents)

    # Make sparse k=3 per sample
    topk = torch.topk(x.abs(), k=3, dim=1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(1, topk.indices, True)
    x[~mask] = 0
    x_sparse = x.to_sparse_csr()

    # Generate labels
    true_w = torch.randn(n_latents, n_classes) * 0.5
    true_b = torch.randn(1, n_classes)
    logits = (x @ true_w) + true_b
    y = torch.bernoulli(torch.sigmoid(logits))

    # Fit with full batch (all classes at once)
    probe_full = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cpu",
        ridge=1e-8,
        class_slab_size=n_classes,  # Process all classes at once
    )
    probe_full.fit(x_sparse, y)

    # Fit with chunked classes
    probe_chunked = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cpu",
        ridge=1e-8,
        class_slab_size=class_slab_size,
    )
    probe_chunked.fit(x_sparse, y)

    # Parameters should match (with slightly relaxed tolerance due to independent convergence)
    torch.testing.assert_close(
        probe_full.coef_, probe_chunked.coef_, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        probe_full.intercept_, probe_chunked.intercept_, rtol=1e-3, atol=1e-3
    )

    # Loss should match
    loss_full = probe_full.loss_matrix(x_sparse, y)
    loss_chunked = probe_chunked.loss_matrix(x_sparse, y)
    torch.testing.assert_close(loss_full, loss_chunked, rtol=1e-3, atol=1e-3)


def _generate_sparse_dataset(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    n_samples, n_latents, n_classes = 40, 6, 5
    x = torch.randn(n_samples, n_latents)

    topk = torch.topk(x.abs(), k=3, dim=1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(1, topk.indices, True)
    x[~mask] = 0
    x_sparse = x.to_sparse_csr()

    true_w = torch.randn(n_latents, n_classes) * 0.7
    true_b = torch.randn(1, n_classes)
    logits = (x @ true_w) + true_b
    y = torch.bernoulli(torch.sigmoid(logits))

    return x, x_sparse, y


def _sklearn_reference_dense(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    C: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_latents = x.shape
    n_classes = y.shape[1]

    loss_ref = np.zeros((n_latents, n_classes))
    coef_ref = np.zeros_like(loss_ref)
    intercept_ref = np.zeros_like(loss_ref)

    for i in range(n_latents):
        xi = x[:, i : i + 1].numpy()
        for c in range(n_classes):
            yc = y[:, c].numpy()

            lr = LogisticRegression(
                fit_intercept=True,
                solver="lbfgs",
                C=C,
                max_iter=max_iter,
            )
            lr.fit(xi, yc)

            z = lr.intercept_[0] + lr.coef_[0, 0] * xi.squeeze()
            mu = 1.0 / (1.0 + np.exp(-z))
            mu = np.clip(mu, 1e-7, 1 - 1e-7)

            loss_ref[i, c] = -(yc * np.log(mu) + (1 - yc) * np.log(1 - mu)).mean()
            coef_ref[i, c] = lr.coef_[0, 0]
            intercept_ref[i, c] = lr.intercept_[0]

    return loss_ref, coef_ref, intercept_ref


def _reference_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    threshold: float,
    clamp_eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_latents = x.shape
    n_classes = y.shape[1]

    loss_ref = np.zeros((n_latents, n_classes))
    tp_ref = np.zeros((n_latents, n_classes))
    fp_ref = np.zeros((n_latents, n_classes))
    tn_ref = np.zeros((n_latents, n_classes))
    fn_ref = np.zeros((n_latents, n_classes))

    for i in range(n_latents):
        xi = x[:, i : i + 1].numpy()
        for c in range(n_classes):
            yc = y[:, c].numpy()

            lr = LogisticRegression(
                fit_intercept=True,
                solver="lbfgs",
                C=1e10,
                max_iter=400,
            )
            lr.fit(xi, yc)

            z = lr.intercept_[0] + lr.coef_[0, 0] * xi.squeeze()
            mu = 1.0 / (1.0 + np.exp(-z))
            mu = np.clip(mu, clamp_eps, 1 - clamp_eps)
            preds = mu > threshold

            loss_ref[i, c] = -(yc * np.log(mu) + (1 - yc) * np.log(1 - mu)).mean()
            yc_bool = yc.astype(bool)
            tp_ref[i, c] = np.logical_and(preds, yc_bool).sum()
            fp_ref[i, c] = np.logical_and(preds, ~yc_bool).sum()
            fn_ref[i, c] = np.logical_and(~preds, yc_bool).sum()
            tn_ref[i, c] = np.logical_and(~preds, ~yc_bool).sum()

    return loss_ref, tp_ref, fp_ref, tn_ref, fn_ref


def _assert_probe_matches_reference(
    threshold: float,
    clamp_eps: float = 1e-7,
) -> None:
    x, x_sparse, y = _generate_sparse_dataset(seed=0)

    probe = Sparse1DProbe(
        n_latents=x_sparse.shape[1],
        n_classes=y.shape[1],
        device="cpu",
        ridge=1e-8,
        class_slab_size=2,
        row_batch_size=16,
    )
    probe.eps = clamp_eps
    probe.fit(x_sparse, y)

    loss, tp, fp, tn, fn = probe.loss_matrix_with_aux(
        x_sparse, y.bool(), threshold=threshold
    )

    loss_ref, tp_ref, fp_ref, tn_ref, fn_ref = _reference_metrics(
        x, y, threshold, clamp_eps=clamp_eps
    )

    torch.testing.assert_close(
        loss.cpu(),
        torch.tensor(loss_ref, dtype=torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        tp.cpu(),
        torch.tensor(tp_ref, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        fp.cpu(),
        torch.tensor(fp_ref, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        tn.cpu(),
        torch.tensor(tn_ref, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        fn.cpu(),
        torch.tensor(fn_ref, dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_confusion_against_sklearn_threshold_point_five():
    _assert_probe_matches_reference(threshold=0.5)


def test_confusion_against_sklearn_threshold_point_two():
    _assert_probe_matches_reference(threshold=0.2)


def test_confusion_against_sklearn_threshold_point_eight():
    _assert_probe_matches_reference(threshold=0.8)


@pytest.mark.slow
def test_fit_matches_sklearn_on_ill_conditioned_sparse_inputs():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    n_samples, n_latents, n_classes = 32, 4, 2

    exponents = torch.randint(-4, 5, (n_samples, n_latents), generator=g).float()
    scales = torch.pow(torch.tensor(10.0), exponents)
    base = torch.randn((n_samples, n_latents), generator=g)
    x_dense = base * scales

    topk = torch.topk(x_dense.abs(), k=3, dim=1)
    mask = torch.zeros_like(x_dense, dtype=torch.bool)
    mask.scatter_(1, topk.indices, True)
    x_dense = torch.where(mask, x_dense, torch.zeros_like(x_dense))

    nnz_per_latent = mask.sum(dim=0)
    for latent_idx in range(n_latents):
        if nnz_per_latent[latent_idx] > 0:
            continue
        row_idx = latent_idx % n_samples
        scale = torch.pow(
            torch.tensor(10.0),
            torch.randint(-4, 5, (), generator=g).float(),
        )
        x_dense[row_idx, latent_idx] = torch.randn((), generator=g) * scale
        mask[row_idx, latent_idx] = True

    true_w = torch.randn((n_latents, n_classes), generator=g) * 1.8
    true_b = torch.randn((1, n_classes), generator=g)
    logits = x_dense @ true_w + true_b
    y = torch.bernoulli(torch.sigmoid(logits))

    loss_ref, coef_ref, intercept_ref = _sklearn_reference_dense(
        x_dense, y, C=5_000.0, max_iter=400
    )

    x_sparse = x_dense.to_sparse_csr()
    probe = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cpu",
        ridge=1e-6,
        n_iter=40,
        tol=1e-6,
        row_batch_size=32,
        class_slab_size=2,
    )
    probe.logger.setLevel(logging.ERROR)
    probe.fit(x_sparse, y)

    loss = probe.loss_matrix(x_sparse, y)
    torch.testing.assert_close(
        loss.cpu(),
        torch.tensor(loss_ref, dtype=torch.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    torch.testing.assert_close(
        probe.coef_.cpu(),
        torch.tensor(coef_ref, dtype=torch.float32),
        rtol=5e-3,
        atol=5e-3,
    )
    torch.testing.assert_close(
        probe.intercept_.cpu(),
        torch.tensor(intercept_ref, dtype=torch.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_confusion_against_sklearn_threshold_extremes():
    torch.manual_seed(123)
    g = torch.Generator().manual_seed(123)
    n_samples, n_latents, n_classes = 48, 5, 3

    raw = torch.randn((n_samples, n_latents), generator=g)
    exponents = torch.randint(-4, 5, (n_samples, n_latents), generator=g).float()
    scales = torch.pow(torch.tensor(10.0), exponents)
    x_dense = raw * scales

    topk = torch.topk(x_dense.abs(), k=3, dim=1)
    mask = torch.zeros_like(x_dense, dtype=torch.bool)
    mask.scatter_(1, topk.indices, True)
    x_dense = torch.where(mask, x_dense, torch.zeros_like(x_dense))

    nnz_per_latent = mask.sum(dim=0)
    for latent_idx in range(n_latents):
        if nnz_per_latent[latent_idx] > 0:
            continue
        row_idx = latent_idx % n_samples
        scale = torch.pow(
            torch.tensor(10.0),
            torch.randint(-4, 5, (), generator=g).float(),
        )
        x_dense[row_idx, latent_idx] = torch.randn((), generator=g) * scale

    true_w = torch.randn((n_latents, n_classes), generator=g) * 3.2
    true_b = torch.randn((1, n_classes), generator=g) * 1.2
    logits = x_dense @ true_w + true_b
    y = torch.bernoulli(torch.sigmoid(logits))

    x_sparse = x_dense.to_sparse_csr()
    probe = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cpu",
        ridge=1e-8,
        n_iter=60,
        tol=1e-6,
        row_batch_size=16,
        class_slab_size=3,
    )
    clamp_eps = 1e-5
    probe.eps = clamp_eps
    probe.logger.setLevel(logging.ERROR)
    probe.fit(x_sparse, y)

    for threshold in (0.02, 0.98):
        loss, tp, fp, tn, fn = probe.loss_matrix_with_aux(
            x_sparse, y.bool(), threshold=threshold
        )
        (
            loss_ref,
            tp_ref,
            fp_ref,
            tn_ref,
            fn_ref,
        ) = _reference_metrics(x_dense, y, threshold, clamp_eps=clamp_eps)

        torch.testing.assert_close(
            loss.cpu(),
            torch.tensor(loss_ref, dtype=torch.float32),
            rtol=1e-2,
            atol=1e-2,
        )
        for actual, expected in (
            (tp, tp_ref),
            (fp, fp_ref),
            (tn, tn_ref),
            (fn, fn_ref),
        ):
            torch.testing.assert_close(
                actual.cpu(),
                torch.tensor(expected, dtype=torch.float32),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("seed", range(3))
def test_chunked_events_vs_full(seed):
    """Verify that processing events/rows in chunks gives same results as processing all at once."""
    torch.manual_seed(seed)
    n_samples, n_latents, n_classes = 64, 8, 4
    x = torch.randn(n_samples, n_latents)

    # Make sparse k=3 per sample
    topk = torch.topk(x.abs(), k=3, dim=1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(1, topk.indices, True)
    x[~mask] = 0
    x_sparse = x.to_sparse_csr()

    # Generate labels
    true_w = torch.randn(n_latents, n_classes) * 0.5
    true_b = torch.randn(1, n_classes)
    logits = (x @ true_w) + true_b
    y = torch.bernoulli(torch.sigmoid(logits))

    # Fit without event chunking (process all rows at once)
    probe_full = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cpu",
        ridge=1e-8,
        row_batch_size=n_samples,
    )
    probe_full.fit(x_sparse, y)

    # Fit with event chunking (process 16 rows at a time)
    probe_chunked = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cpu",
        ridge=1e-8,
        row_batch_size=16,
    )
    probe_chunked.fit(x_sparse, y)

    # Parameters should match (with slightly relaxed tolerance due to chunking)
    torch.testing.assert_close(
        probe_full.coef_, probe_chunked.coef_, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        probe_full.intercept_, probe_chunked.intercept_, rtol=1e-3, atol=1e-3
    )


@pytest.mark.slow
def test_realistic_scale():
    """Test that chunked implementation can handle realistic dimensions without OOM."""
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)

    torch.manual_seed(42)
    n_samples, n_latents, n_classes = 64_000, 1024, 20
    # Simulate sparse activations with L0=100
    nnz_per_sample = 12

    # Build sparse CSR matrix efficiently
    indices = []
    indptr = [0]
    data = []

    for i in range(n_samples):
        # Random latents activated for this sample
        cols = np.random.choice(n_latents, size=nnz_per_sample, replace=False)
        vals = np.random.randn(nnz_per_sample).astype(np.float32)

        indices.extend(cols)
        data.extend(vals)
        indptr.append(len(indices))

    x_sparse = torch.sparse_csr_tensor(
        torch.tensor(indptr, dtype=torch.int32),
        torch.tensor(indices, dtype=torch.int32),
        torch.tensor(data, dtype=torch.float32),
        size=(n_samples, n_latents),
    )

    # Random labels
    y = torch.zeros((n_samples, n_classes), dtype=torch.float32)
    y[torch.arange(n_samples), torch.randint(0, n_classes, (n_samples,))] = 1.0

    # Should not OOM with chunking
    probe = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cpu",
        ridge=1e-8,
        class_slab_size=8,
        row_batch_size=10_000,
        n_iter=5,  # Fewer iterations for speed
    )
    probe.fit(x_sparse, y)

    # Verify it produced reasonable results
    assert probe.coef_.shape == (n_latents, n_classes)
    assert probe.intercept_.shape == (n_latents, n_classes)
    assert not torch.isnan(probe.coef_).any()
    assert not torch.isnan(probe.intercept_).any()
