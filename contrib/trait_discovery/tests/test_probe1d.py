"""Unit tests for 1D probe training."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from tdiscovery.probe1d import Sparse1DProbe


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
    clf = Sparse1DProbe(n_latents=n_latents, n_classes=n_classes, device="cpu")

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
        n_latents=n_latents, n_classes=n_classes, device="cpu", ridge=1e-10
    )
    probe.fit(x.to_sparse_csr(), y)
    loss_sparse = probe.loss_matrix(x.to_sparse_csr(), y)
    torch.testing.assert_close(
        loss_sparse, torch.tensor(loss_ref, dtype=torch.float32), rtol=1e-4, atol=1e-4
    )
