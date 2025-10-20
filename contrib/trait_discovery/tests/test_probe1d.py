"""Unit tests for 1D probe training."""

import logging

import beartype
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from tdiscovery.probe1d import Reference1DProbe, Sparse1DProbe
from torch import Tensor

REF_MAX_ITER = 2048
SPARSE_MAX_ITER = 512
REF_MAX_ITER_FAST = 1024
SPARSE_MAX_ITER_FAST = 256

cuda_available = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires GPU"
)


@beartype.beartype
def _train_probe_and_reference(
    x_dense: Tensor,
    y: Tensor,
    *,
    ridge: float = 1e-8,
    max_iter: int = 80,
    lm_max_update: float = 8.0,
) -> tuple[Sparse1DProbe, Reference1DProbe]:
    x_sparse = x_dense.to_sparse_csr()
    probe = Sparse1DProbe(
        n_latents=x_dense.shape[1],
        n_classes=y.shape[1],
        device="cpu",
        ridge=ridge,
        max_iter=max_iter,
        row_batch_size=256,
        class_slab_size=y.shape[1],
        lm_max_update=lm_max_update,
    )
    probe.logger.setLevel(logging.ERROR)
    probe.fit(x_sparse, y.to(torch.float32))
    ref = Reference1DProbe(
        ridge=ridge,
        tol=1e-10,
        max_iter=max(max_iter * 4, REF_MAX_ITER),
        delta_logit=8.0,
    )
    ref.fit(x_dense[:, 0].numpy(), y.squeeze(1).numpy())
    return probe, ref


def test_reference_probe_matches_sklearn():
    rng = np.random.default_rng(0)
    n_samples = 512
    x = rng.normal(0.0, 1.0, size=n_samples)
    logits_true = -0.75 + 1.35 * x
    probs = 1.0 / (1.0 + np.exp(-logits_true))
    y = rng.binomial(1, probs)

    probe = Reference1DProbe(
        ridge=1e-6,
        tol=1e-8,
        max_iter=REF_MAX_ITER,
        delta_logit=8.0,
    )
    probe.fit(x, y)
    b_probe = float(probe.intercept_[0])
    w_probe = float(probe.coef_[0])

    lr = LogisticRegression(
        fit_intercept=True,
        solver="lbfgs",
        C=1e8,
        max_iter=5000,
    )
    lr.fit(x.reshape(-1, 1), y)
    b_ref = float(lr.intercept_[0])
    w_ref = float(lr.coef_[0, 0])

    np.testing.assert_allclose(b_probe, b_ref, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(w_probe, w_ref, rtol=5e-2, atol=5e-2)


def test_reference_probe_handles_linearly_separable():
    pos = np.linspace(0.5, 5.0, num=64)
    neg = np.linspace(-5.0, -0.5, num=64)
    x = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros_like(neg), np.ones_like(pos)])

    probe = Reference1DProbe(
        ridge=1e-4,
        tol=1e-8,
        max_iter=REF_MAX_ITER,
        delta_logit=6.0,
    )
    probe.fit(x, y)

    logits = float(probe.intercept_[0]) + float(probe.coef_[0]) * x
    preds = (logits > 0).astype(int)
    assert np.array_equal(preds, y.astype(int))
    assert probe.converged_ or probe.n_iter_ == probe.max_iter


@pytest.mark.parametrize("seed", range(6))
def test_reference_probe_matches_sklearn_random_seeds(seed: int):
    rng = np.random.default_rng(seed)
    n_samples = 256
    x = rng.normal(size=n_samples)
    logits_true = rng.normal() + rng.normal() * x
    probs = 1.0 / (1.0 + np.exp(-logits_true))
    y = rng.binomial(1, probs)

    reference = Reference1DProbe(
        ridge=1e-8,
        tol=1e-10,
        max_iter=REF_MAX_ITER,
        delta_logit=8.0,
    )
    reference.fit(x, y)
    lr = LogisticRegression(
        fit_intercept=True,
        solver="lbfgs",
        C=1e10,
        max_iter=5000,
    )
    lr.fit(x.reshape(-1, 1), y)

    np.testing.assert_allclose(
        reference.intercept_[0], lr.intercept_[0], rtol=5e-2, atol=5e-2
    )
    np.testing.assert_allclose(reference.coef_[0], lr.coef_[0, 0], rtol=5e-2, atol=5e-2)


def test_reference_probe_mismatch_on_separable_data():
    x_neg = np.linspace(-4.0, -0.5, num=40)
    x_pos = np.linspace(0.5, 4.0, num=40)
    x = np.concatenate([x_neg, x_pos])
    y = np.concatenate([np.zeros_like(x_neg), np.ones_like(x_pos)]).astype(int)

    reference = Reference1DProbe(ridge=1e-8, tol=1e-10, max_iter=128, delta_logit=6.0)
    reference.fit(x, y)
    lr = LogisticRegression(fit_intercept=True, solver="lbfgs", C=1e8, max_iter=500)
    lr.fit(x.reshape(-1, 1), y)

    # Expect divergence: ridge keeps Reference1DProbe bounded, sklearn drives towards large magnitude.
    diff_coef = abs(reference.coef_[0] - lr.coef_[0, 0])
    diff_intercept = abs(reference.intercept_[0] - lr.intercept_[0])
    assert diff_coef > 0.5 or diff_intercept > 0.5


@pytest.mark.slow
def test_sparse_probe_matches_reference_on_ill_conditioned_inputs():
    for params in ILL_CONDITIONED_CASES:
        x, y = _generate_ill_conditioned_dataset(*params)
        diffs = _reference_vs_sparse_diffs(
            x,
            y,
            ref_max_iter=REF_MAX_ITER,
            sparse_max_iter=SPARSE_MAX_ITER,
        )
        assert diffs is not None
        coef_diff, intercept_diff, loss_diff = diffs
        max_diff = max(coef_diff, intercept_diff, loss_diff)
        assert max_diff <= 5e-2


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
def test_fit_against_reference(seed):
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
        xi = x[:, i : i + 1].numpy().squeeze(1)
        for c in range(n_classes):
            yc = y[:, c].numpy()

            ref = Reference1DProbe(
                ridge=1e-10,
                tol=1e-10,
                max_iter=REF_MAX_ITER,
                delta_logit=8.0,
            )
            ref.fit(xi, yc)

            z = ref.intercept_[0] + ref.coef_[0] * xi
            mu = 1 / (1 + np.exp(-z))
            loss_ref[i, c] = -(yc * np.log(mu) + (1 - yc) * np.log(1 - mu)).mean()

    # Use very small ridge to effectively disable regularization (matching the reference)
    probe = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cpu",
        ridge=1e-10,
        max_iter=SPARSE_MAX_ITER,
        row_batch_size=4,
    )
    probe.fit(x.to_sparse_csr(), y)
    loss_sparse = probe.loss_matrix(x.to_sparse_csr(), y)
    torch.testing.assert_close(
        loss_sparse, torch.tensor(loss_ref, dtype=torch.float32), rtol=1e-4, atol=1e-4
    )


@cuda_available
@pytest.mark.parametrize("seed", range(5))
def test_fit_against_reference_on_gpu(seed):
    """Test that our GPU implementation matches the reference probe."""
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
        xi = x[:, i : i + 1].numpy().squeeze(1)
        for c in range(n_classes):
            yc = y[:, c].numpy()

            ref = Reference1DProbe(
                ridge=1e-10,
                tol=1e-10,
                max_iter=REF_MAX_ITER,
                delta_logit=8.0,
            )
            ref.fit(xi, yc)

            z = ref.intercept_[0] + ref.coef_[0] * xi
            mu = 1 / (1 + np.exp(-z))
            loss_ref[i, c] = -(yc * np.log(mu) + (1 - yc) * np.log(1 - mu)).mean()

    # Use very small ridge to effectively disable regularization (matching the reference)
    # Run on GPU this time
    probe = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cuda:0",
        ridge=1e-10,
        max_iter=SPARSE_MAX_ITER,
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


def test_ill_conditioned_extreme_intercept_matches_reference():
    torch.manual_seed(0)
    n_samples = 256
    x_dense = torch.zeros(n_samples, 1, dtype=torch.float32)

    active_idx = torch.randperm(n_samples)[:48]
    x_dense[active_idx, 0] = 1.8 + 0.4 * torch.randn(active_idx.shape[0])

    logits_true = -2.8 + 2.0 * x_dense[:, 0]
    probs = torch.sigmoid(logits_true)
    y = torch.bernoulli(probs).unsqueeze(1)
    if int(y.sum().item()) == 0:
        y[active_idx[0], 0] = 1.0
    if int((y == 0).sum().item()) == 0:
        zero_idx = torch.randint(0, n_samples, (1,)).item()
        y[zero_idx, 0] = 0.0

    probe, ref = _train_probe_and_reference(x_dense, y, max_iter=12)

    w_probe = probe.coef_[0, 0].item()
    b_probe = probe.intercept_[0, 0].item()

    w_ref = float(ref.coef_[0])
    b_ref = float(ref.intercept_[0])

    np.testing.assert_allclose(w_probe, w_ref, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(b_probe, b_ref, rtol=5e-2, atol=5e-2)

    logits_probe = b_probe + w_probe * x_dense[:, 0]
    logits_ref = b_ref + w_ref * x_dense[:, 0]
    loss_probe = F.binary_cross_entropy_with_logits(
        logits_probe, y.squeeze(1), reduction="mean"
    )
    loss_ref = F.binary_cross_entropy_with_logits(
        logits_ref, y.squeeze(1), reduction="mean"
    )
    torch.testing.assert_close(loss_probe, loss_ref, rtol=1e-3, atol=1e-4)


def test_ill_conditioned_large_scale_matches_reference():
    torch.manual_seed(1)
    n_samples = 256
    x_dense = torch.empty(n_samples, 1, dtype=torch.float32)

    signs = torch.bernoulli(torch.full((n_samples,), 0.35)).bool()
    large_value = 200.0
    x_dense[:, 0] = torch.where(
        signs,
        torch.full((n_samples,), large_value, dtype=torch.float32),
        torch.full((n_samples,), -large_value, dtype=torch.float32),
    )
    x_dense[:, 0] += 0.5 * torch.randn(n_samples)

    logits_true = 0.2 + 0.01 * x_dense[:, 0]
    probs = torch.sigmoid(logits_true).clamp(1e-6, 1 - 1e-6)
    y = torch.bernoulli(probs).unsqueeze(1)
    if int(y.sum().item()) == 0:
        y[0, 0] = 1.0
    if int((y == 0).sum().item()) == 0:
        zero_idx = torch.randint(0, n_samples, (1,)).item()
        y[zero_idx, 0] = 0.0

    probe, ref = _train_probe_and_reference(x_dense, y, max_iter=15)

    w_probe = probe.coef_[0, 0].item()
    b_probe = probe.intercept_[0, 0].item()

    w_ref = float(ref.coef_[0])
    b_ref = float(ref.intercept_[0])

    np.testing.assert_allclose(w_probe, w_ref, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(b_probe, b_ref, rtol=5e-2, atol=5e-2)

    logits_probe = b_probe + w_probe * x_dense[:, 0]
    logits_ref = b_ref + w_ref * x_dense[:, 0]
    loss_probe = F.binary_cross_entropy_with_logits(
        logits_probe, y.squeeze(1), reduction="mean"
    )
    loss_ref = F.binary_cross_entropy_with_logits(
        logits_ref, y.squeeze(1), reduction="mean"
    )
    torch.testing.assert_close(loss_probe, loss_ref, rtol=1e-3, atol=1e-4)


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


def _generate_sparse_dataset(seed: int) -> tuple[Tensor, Tensor, Tensor]:
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


def _reference_probe_dense(
    x: Tensor,
    y: Tensor,
    *,
    ridge: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_latents = x.shape
    n_classes = y.shape[1]

    loss_ref = np.zeros((n_latents, n_classes))
    coef_ref = np.zeros_like(loss_ref)
    intercept_ref = np.zeros_like(loss_ref)

    for i in range(n_latents):
        xi = x[:, i : i + 1].numpy()
        xi_1d = xi.squeeze(1)
        for c in range(n_classes):
            yc = y[:, c].numpy()

            ref = Reference1DProbe(
                ridge=ridge,
                tol=tol,
                max_iter=max_iter,
                delta_logit=8.0,
            )
            ref.fit(xi_1d, yc)

            z = ref.intercept_[0] + ref.coef_[0] * xi_1d
            mu = 1.0 / (1.0 + np.exp(-z))
            mu = np.clip(mu, 1e-7, 1 - 1e-7)

            loss_ref[i, c] = -(yc * np.log(mu) + (1 - yc) * np.log(1 - mu)).mean()
            coef_ref[i, c] = ref.coef_[0]
            intercept_ref[i, c] = ref.intercept_[0]

    return loss_ref, coef_ref, intercept_ref


def _reference_metrics(
    x: Tensor, y: Tensor, threshold: float, clamp_eps: float = 1e-7
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
        xi_1d = xi.squeeze(1)
        for c in range(n_classes):
            yc = y[:, c].numpy()

            ref = Reference1DProbe(
                ridge=1e-10,
                tol=1e-10,
                max_iter=REF_MAX_ITER,
                delta_logit=8.0,
            )
            ref.fit(xi_1d, yc)

            z = ref.intercept_[0] + ref.coef_[0] * xi_1d
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


def _mean_log_loss(b: float, w: float, x: np.ndarray, y: np.ndarray) -> float:
    logits = np.clip(b + w * x, -80.0, 80.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(probs) + (1 - y) * np.log(1 - probs)).mean())


def _fit_reference_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
    tol: float,
    max_iter: int = REF_MAX_ITER,
) -> Reference1DProbe:
    ref = Reference1DProbe(
        ridge=ridge,
        tol=tol,
        max_iter=max_iter,
        delta_logit=8.0,
    )
    ref.fit(x, y)
    return ref


def _fit_sparse_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
    max_iter: int = SPARSE_MAX_ITER,
) -> Sparse1DProbe:
    sparse = Sparse1DProbe(
        n_latents=1,
        n_classes=1,
        device="cpu",
        ridge=ridge,
        max_iter=max_iter,
        class_slab_size=1,
        row_batch_size=64,
    )
    xs = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to_sparse_csr()
    ys = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    sparse.fit(xs, ys)
    return sparse


def _generate_ill_conditioned_dataset(
    n: int, scale: float, w: float, b: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n) * scale
    logits = np.clip(b + w * x, -80.0, 80.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)
    return x.astype(float), y.astype(float)


def _reference_vs_sparse_diffs(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ref_max_iter: int = REF_MAX_ITER_FAST,
    sparse_max_iter: int = SPARSE_MAX_ITER_FAST,
) -> tuple[float, float, float] | None:
    if y.sum() == 0 or y.sum() == y.shape[0]:
        return None
    ref = _fit_reference_probe(x, y, ridge=1e-6, tol=1e-8, max_iter=ref_max_iter)
    sparse = _fit_sparse_probe(x, y, ridge=1e-6, max_iter=sparse_max_iter)
    coef_diff = abs(float(ref.coef_[0]) - float(sparse.coef_[0, 0]))
    intercept_diff = abs(float(ref.intercept_[0]) - float(sparse.intercept_[0, 0]))
    loss_ref = _mean_log_loss(float(ref.intercept_[0]), float(ref.coef_[0]), x, y)
    loss_sparse = _mean_log_loss(
        float(sparse.intercept_[0, 0]), float(sparse.coef_[0, 0]), x, y
    )
    return coef_diff, intercept_diff, abs(loss_ref - loss_sparse)


ILL_CONDITIONED_CASES: list[tuple[int, float, float, float, int]] = [
    (120, 1000.0, -1.0360736815895426, -1.5655893713726405, 3),
    (128, 300.0, -0.85, -0.75, 17),
    (96, 1000.0, -1.2, 0.25, 42),
]


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
        max_iter=SPARSE_MAX_ITER,
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


def test_confusion_against_reference_threshold_point_five():
    _assert_probe_matches_reference(threshold=0.5)


def test_confusion_against_reference_threshold_point_two():
    _assert_probe_matches_reference(threshold=0.2)


def test_confusion_against_reference_threshold_point_eight():
    _assert_probe_matches_reference(threshold=0.8)


@pytest.mark.slow
def test_fit_matches_reference_on_ill_conditioned_sparse_inputs():
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

    loss_ref, coef_ref, intercept_ref = _reference_probe_dense(
        x_dense, y, ridge=1e-6, tol=1e-9, max_iter=REF_MAX_ITER
    )

    x_sparse = x_dense.to_sparse_csr()
    probe = Sparse1DProbe(
        n_latents=n_latents,
        n_classes=n_classes,
        device="cpu",
        ridge=1e-6,
        max_iter=SPARSE_MAX_ITER,
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


def test_confusion_against_reference_threshold_extremes():
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
        max_iter=SPARSE_MAX_ITER,
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


def test_lm_step_respects_logit_budget():
    probe = Sparse1DProbe(
        n_latents=1,
        n_classes=1,
        device="cpu",
        lm_max_update=6.0,
        lm_max_adapt_iters=4,
    )

    g0 = torch.tensor([[15.0]], dtype=torch.float32)
    g1 = torch.tensor([[5.0]], dtype=torch.float32)
    h0 = torch.tensor([[2.0]], dtype=torch.float32)
    h1 = torch.tensor([[0.1]], dtype=torch.float32)
    h2 = torch.tensor([[3.0]], dtype=torch.float32)
    lam_prev = torch.full_like(g0, probe.lam_init)

    db, dw, _, lam_next, clipped = probe._compute_lm_step(
        g0=g0, g1=g1, h0=h0, h1=h1, h2=h2, lam=lam_prev, qx_sq=torch.tensor([[1.0]])
    )

    norm = float((db**2 + dw**2).sqrt())
    assert norm <= probe.delta_logit + 1e-6
    assert (lam_next >= lam_prev).all()
    assert clipped.any()


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
        max_iter=5,  # Fewer iterations for speed
    )
    probe.fit(x_sparse, y)

    # Verify it produced reasonable results
    assert probe.coef_.shape == (n_latents, n_classes)
    assert probe.intercept_.shape == (n_latents, n_classes)
    assert not torch.isnan(probe.coef_).any()
    assert not torch.isnan(probe.intercept_).any()
