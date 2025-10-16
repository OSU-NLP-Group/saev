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


def test_accuracy_matches_dense():
    """Accuracy computed via chunked implementation should match dense reference."""
    torch.manual_seed(0)
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

    probe = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cpu",
        ridge=1e-8,
        class_slab_size=2,
        row_batch_size=16,
    )
    probe.fit(x_sparse, y)

    _, aux = probe.loss_matrix_with_aux(x_sparse, y.bool())
    acc_sparse = aux["accuracy"].cpu()

    with torch.no_grad():
        logits_dense = probe.intercept_.unsqueeze(0) + probe.coef_.unsqueeze(
            0
        ) * x.unsqueeze(2)
        pred_dense = torch.sigmoid(logits_dense) > 0.5
        y_expanded = y.unsqueeze(1).bool()
        acc_dense = (pred_dense == y_expanded).float().mean(dim=0)

    torch.testing.assert_close(acc_sparse, acc_dense, rtol=1e-5, atol=1e-5)


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
