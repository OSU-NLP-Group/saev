"""
Sparse 1D logistic probes for trait discovery.

This module implements Newton-style optimization and evaluation for per-latent / per-class logistic probes on high-sparsity SAE activations.
The key invariants across implementations are:

* Sparse feature matrix `x` is streamed in CSR format without materializing tensors shaped `(nnz, n_classes)`.
* Classes are processed in configurable slabs (`class_slab_size`) while rows are processed in configurable micro-batches (`row_batch_size`).
* All compute paths (`fit`, `loss_matrix`, `loss_matrix_with_aux`) share the same sparse event iterator to guarantee identical traversal order.

The public surface area is intentionally small and designed to be used by tests and training sweeps. The heavy lifting occurs in `Sparse1DProbe`, which exposes the learned coefficients, loss computation helpers, and confusion-matrix diagnostics.
"""

import dataclasses
import logging
import math
import pathlib
import typing as tp
from collections.abc import Iterator
from typing import NamedTuple

import beartype
import einops
import numpy as np
import scipy.sparse
import sklearn.base
import torch
import tyro
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

import saev.data
import saev.disk
import saev.helpers


@jaxtyped(typechecker=beartype.beartype)
class SparseEventsBatch(NamedTuple):
    """Streaming view over CSR non-zeros for a row-aligned batch.

    Args:
        row_start: Inclusive row index where this batch starts.
        row_end: Exclusive row index where this batch ends.
        latent_idx: Column indices of the non-zero entries in this batch.
        values: Values of the non-zero entries in this batch.
        row_idx: Absolute row index for each non-zero entry.
    """

    row_start: int
    row_end: int
    latent_idx: Int[Tensor, " event"]
    values: Float[Tensor, " event"]
    row_idx: Int[Tensor, " event"]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SlabStats:
    """Aggregated Newton statistics for a `(latent, class)` slab.

    Attributes:
        g0: Gradient of the intercept component.
        g1: Gradient of the weight component.
        h0: Hessian entry for intercept x intercept.
        h1: Hessian entry for intercept x weight.
        h2: Hessian entry for weight x weight.
        loss_nz: Sum of negative log-likelihood over non-zero events.
        pos_nz: Count of positive labels observed in non-zero events.
        mu_nz: Sum of predicted probabilities on non-zero events.
    """

    g0: Float[Tensor, "n_latents c_b"]
    g1: Float[Tensor, "n_latents c_b"]
    h0: Float[Tensor, "n_latents c_b"]
    h1: Float[Tensor, "n_latents c_b"]
    h2: Float[Tensor, "n_latents c_b"]
    loss_nz: Float[Tensor, "n_latents c_b"]
    pos_nz: Float[Tensor, "n_latents c_b"]
    mu_nz: Float[Tensor, "n_latents c_b"]


@beartype.beartype
class StepResult(NamedTuple):
    """Holds state returned by `_solve_lm_step`."""

    db: Float[Tensor, "n_latents c_b"]
    dw: Float[Tensor, "n_latents c_b"]
    clipped_mask: Bool[Tensor, "n_latents c_b"]
    det_h: Float[Tensor, "n_latents c_b"]
    det_h_raw: Float[Tensor, "n_latents c_b"]
    h0_eff: Float[Tensor, "n_latents c_b"]
    h2_eff: Float[Tensor, "n_latents c_b"]
    lambda_next: Float[Tensor, "n_latents c_b"]
    predicted_reduction: Float[Tensor, "n_latents c_b"]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class StepContext:
    """Snapshot of state passed to `_log_lm_step` for reporting.

    Attributes:
        class_range: Inclusive/exclusive class indices `(c0, c1)` for this slab.
        iteration: Zero-based iteration counter within the slab loop.
        mean_loss_value: Mean NLL across the slab at this iteration.
        loss_slab: Full loss matrix for this slab.
        stats: Newton statistics accumulated by `_accumulate_slab_stats`.
        lambda_prev: Damping tensor before the LM update.
        lambda_next: Damping tensor after the LM update.
        db: Newton update for intercepts.
        dw: Newton update for coefficients.
        improved_mask: Boolean mask of classes whose best latent improved.
        curr_best_loss: Current best losses per class within the slab.
        curr_best_latent_idx: Indices of the best latent per class.
        best_update_summary: Human-readable summaries of improvements.
        coef_slab: Coefficient slice for the slab (post-update).
        intercept_slab: Intercept slice for the slab (post-update).
        nnz_per_latent: Non-zero counts per latent.
        mu_0: Predicted probability for zero activations.
        n_zeros_per_latent: Count of implicit zeros per latent.
        pi_slab_device: Positive counts per class on device.
        lambda_step: Result of `_solve_lm_step` for further diagnostics.
    """

    class_range: tuple[int, int]
    iteration: int
    mean_loss_value: float
    loss_slab: Float[Tensor, "n_latents c_b"]
    stats: SlabStats
    lambda_prev: Float[Tensor, "n_latents c_b"]
    lambda_next: Float[Tensor, "n_latents c_b"]
    db: Float[Tensor, "n_latents c_b"]
    dw: Float[Tensor, "n_latents c_b"]
    improved_mask: Tensor
    curr_best_loss: Tensor
    curr_best_latent_idx: Tensor
    best_update_summary: list[str]
    coef_slab: Float[Tensor, "n_latents c_b"]
    intercept_slab: Float[Tensor, "n_latents c_b"]
    nnz_per_latent: Tensor
    mu_0: Float[Tensor, "n_latents c_b"]
    n_zeros_per_latent: Float[Tensor, "n_latents 1"]
    pi_slab_device: Tensor
    lambda_step: StepResult


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class StepSummary:
    """Lightweight iteration metrics emitted via `iteration_hook`.

    Attributes map one-to-one with the values logged inside `_log_lm_step` so
    external consumers can record convergence traces without parsing logs.
    """

    class_range: tuple[int, int]
    iteration: int
    mean_loss: float
    max_grad: float
    max_update: float
    lambda_min: float
    lambda_max: float
    predicted_reduction_max: float
    predicted_reduction_mean: float


@beartype.beartype
def sigmoid(z: np.ndarray | float) -> np.ndarray:
    # stable logistic
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return np.clip(out, 1e-12, 1 - 1e-12)


@beartype.beartype
class SlowProbe(sklearn.base.BaseEstimator):
    def __init__(
        self,
        ridge: float = 1e-8,
        tol: float = 1e-6,
        max_iter: int = 200,
        lam_init: float = 1e-3,
        lam_shrink: float = 0.1,
        lam_grow: float = 10.0,
        delta_logit: float = 6.0,
        qx: float | None = None,
        use_elliptical: bool = False,
    ):
        if lam_shrink <= 0 or lam_shrink >= 1:
            msg = f"lam_shrink must lie in (0,1), got {lam_shrink}."
            raise ValueError(msg)
        if lam_grow <= 1:
            msg = f"lam_grow must be >1, got {lam_grow}."
            raise ValueError(msg)
        if delta_logit <= 0:
            msg = f"delta_logit must be >0, got {delta_logit}."
            raise ValueError(msg)
        if qx is not None and qx <= 0:
            msg = f"qx must be positive when provided, got {qx}."
            raise ValueError(msg)

        self.ridge = float(ridge)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.lam_init = float(lam_init)
        self.lam_shrink = float(lam_shrink)
        self.lam_grow = float(lam_grow)
        self.delta_logit = float(delta_logit)
        self.qx_override = float(qx) if qx is not None else None
        self.use_elliptical = bool(use_elliptical)
        self.lam_min = 1e-12
        self.lam_max = 1e12

        self.intercept_: float | None = None
        self.weight_: float | None = None
        self.converged_: bool = False
        self.n_iter_: int = 0

    def fit(self, X, y):
        """
        Dense, single (latent,class) solver.
        Accepts either a 1D array (n,) or a column vector (n,1).
        Ridge penalty: 0.5*(w^2 + (b - b0)^2)
        Trust region: either box (|Δb|<=δ, |Δw|<=δ/qx) or elliptical ||DΔ||2<=δ with D=diag(1,qx).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            x = X.reshape(-1)
        else:
            if X.ndim != 2 or X.shape[1] != 1:
                msg = (
                    "SlowProbe expects exactly one feature; "
                    f"received array with shape {X.shape}."
                )
                raise ValueError(msg)
            x = X[:, 0]

        y = np.asarray(y, dtype=float).reshape(-1)
        if x.shape[0] != y.shape[0]:
            msg = (
                "x and y must have matching lengths; "
                f"received {x.shape[0]} and {y.shape[0]}."
            )
            raise ValueError(msg)
        if x.shape[0] == 0:
            msg = "x and y must contain at least one sample."
            raise ValueError(msg)
        if np.any((y < 0) | (y > 1)):
            msg = "y must contain only probabilities in [0, 1]."
            raise ValueError(msg)

        pi = y.mean()
        pi = np.clip(pi, 1e-12, 1 - 1e-12)
        b0 = np.log(pi / (1 - pi))

        b = float(b0)
        w = 0.0
        lam = float(self.lam_init)
        qx_value = self.qx_override
        if qx_value is None:
            # robust scale; 95th percentile of |x| is fine too
            nnz = x[x != 0]
            if nnz.size == 0:
                qx_value = 1.0
            else:
                qx_value = np.quantile(np.abs(nnz), 0.95)
                if not np.isfinite(qx_value) or qx_value <= 1e-12:
                    qx_value = float(np.sqrt(np.mean(nnz**2)))
        qx_value = max(float(qx_value), 1e-12)

        self.qx_ = qx_value

        prev_pred = None
        prev_loss_before_step = None
        step_max = float("inf")
        grad_max = float("inf")

        def loss(b, w):
            mu = sigmoid(b + w * x)
            # NLL + ridge
            return -(
                y * np.log(mu) + (1 - y) * np.log(1 - mu)
            ).sum() + 0.5 * self.ridge * (w**2 + (b - b0) ** 2)

        loss_curr = loss(b, w)

        for it in range(self.max_iter):
            rho = None
            if prev_pred is not None:
                actual = prev_loss_before_step - loss_curr
                rho = actual / max(prev_pred, 1e-18)
                if not np.isfinite(rho):
                    lam = min(lam * self.lam_grow, self.lam_max)
                elif rho >= 0.75:
                    lam = max(lam * self.lam_shrink, self.lam_min)
                elif rho <= 0.25:
                    lam = min(lam * self.lam_grow, self.lam_max)

            z = b + w * x
            mu = sigmoid(z)
            s = mu * (1 - mu)
            r = mu - y

            g0 = r.sum() + self.ridge * (b - b0)
            g1 = (r * x).sum() + self.ridge * w
            h0 = s.sum() + self.ridge
            h1 = (s * x).sum()
            h2 = (s * x * x).sum() + self.ridge

            grad_max = max(abs(g0), abs(g1))

            # LM step with retries
            tried = 0
            db = dw = 0.0
            while tried < 6:
                if self.use_elliptical:
                    # scaled LM: H + lam * D^T D with D=diag(1,qx)
                    h0_eff = h0 + lam * 1.0
                    h2_eff = h2 + lam * (qx_value * qx_value)
                else:
                    h0_eff = h0 + lam
                    h2_eff = h2 + lam
                det = h0_eff * h2_eff - h1 * h1
                if abs(det) < 1e-18:
                    lam = min(lam * self.lam_grow, self.lam_max)
                    tried += 1
                    continue

                db = (h2_eff * g0 - h1 * g1) / det
                dw = (-h1 * g0 + h0_eff * g1) / det

                # box or elliptical trust region
                if self.use_elliptical:
                    norm = np.sqrt(db * db + (qx_value * dw) * (qx_value * dw))
                    if norm > self.delta_logit:
                        scale = self.delta_logit / (norm + 1e-18)
                        db *= scale
                        dw *= scale
                else:
                    changed = False
                    if abs(db) > self.delta_logit:
                        db = np.sign(db) * self.delta_logit
                        changed = True
                    dw_limit = self.delta_logit / qx_value
                    if abs(dw) > dw_limit:
                        dw = np.sign(dw) * dw_limit
                        changed = True
                    if changed:
                        # mark as a "clipped" step: encourage larger damping
                        lam = min(lam * self.lam_grow, self.lam_max)

                # predicted quadratic-model decrease (correct sign)
                pred = (
                    g0 * db
                    + g1 * dw
                    - 0.5 * (h0 * db * db + 2 * h1 * db * dw + h2 * dw * dw)
                )
                if not np.isfinite(pred) or pred <= 0:
                    lam = min(lam * self.lam_grow, self.lam_max)
                    tried += 1
                    continue
                break

            # apply step
            b_new = b - db
            w_new = w - dw
            loss_new = loss(b_new, w_new)

            prev_pred = pred
            prev_loss_before_step = loss_curr
            loss_curr = loss_new
            b, w = b_new, w_new

            step_max = max(abs(db), abs(dw))
            if grad_max < self.tol and step_max < self.tol:
                self.intercept_ = b
                self.weight_ = w
                self.converged_ = True
                self.n_iter_ = it + 1
                self.coef_ = np.array([w], dtype=float)
                self.intercept_ = np.array([b], dtype=float)
                return self

        self.intercept_ = b
        self.weight_ = w
        self.converged_ = False
        self.n_iter_ = self.max_iter
        self.coef_ = np.array([w], dtype=float)
        self.intercept_ = np.array([b], dtype=float)
        return self

    def decision_function(self, X):
        if not hasattr(self, "coef_") or not hasattr(self, "intercept_"):
            msg = "SlowProbe instance is not fitted yet."
            raise RuntimeError(msg)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            x = X.reshape(-1)
        else:
            if X.ndim != 2 or X.shape[1] != 1:
                msg = (
                    "SlowProbe expects exactly one feature; "
                    f"received array with shape {X.shape}."
                )
                raise ValueError(msg)
            x = X[:, 0]
        return self.intercept_[0] + self.coef_[0] * x

    def predict_proba(self, X):
        logits = self.decision_function(X)
        probs = sigmoid(logits)
        return np.stack([1 - probs, probs], axis=1)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


@jaxtyped(typechecker=beartype.beartype)
class Sparse1DProbe(sklearn.base.BaseEstimator):
    """Streaming Newton optimizer for per-latent logistic probes.

    For each latent ell and class c we fit a logistic model with parameters (b_{ell,c}, w_{ell,c}) on the sparse SAE activations `x_{j,ell}`. The loss is the negative log-likelihood L(b, w) = sum_j BCE(y_{j,c}, sigma(b + w x_{j,ell})). We initialize b_{ell,c} to the class prevalence logit logit(pi_c) so the model starts at the intercept-only optimum, and keep w_{ell,c}=0.

    One Newton step proceeds as follows.

    1. Clamp probabilities away from 0/1 with eps=1e-7 and reuse mu_0 = sigma(b_{ell,c}) and s_0 = mu_0(1-mu_0) for the implicit zeros. This stabilizes the BCE against saturated logits.
    2. Stream the non-zero events of the CSR matrix and accumulate the sufficient statistics described in the Probe1D design doc: gradients G_0, G_1 and Hessian entries H_0, H_1, H_2. Zeros contribute in closed form via mu_0 and s_0. We also record sum mu_nonzero to reuse in the gradient.
    3. Add ridge terms: G_1 <- G_1 + lambda w_{ell,c} and H_2 <- H_2 + lambda. For the intercept we regularize the deviation from the prevalence baseline, G_0 <- G_0 + lambda (b_{ell,c} - logit(pi_c)) and H_0 <- H_0 + lambda. This keeps the maximum-likelihood solution finite even when the pair is (nearly) separable.
    4. Solve the 2x2 Newton system using a Levenberg-Marquardt damping step (Levenberg 1944, Marquardt 1963). Concretely, we add the current damping lambda_k to the diagonal (H_0 + lambda_k, H_2 + lambda_k) before inverting, compute Delta b, Delta w, and predict the quadratic reduction 0.5 (Delta b G_0 + Delta w G_1).
    5. Guard the step with a simple trust-region update inspired by More's dogleg methods: if abs(Delta b) or abs(Delta w) exceeds `lm_max_update`, or if the predicted reduction is non-positive/NaN, enlarge lambda_k (by `lm_lambda_grow`) and retry, up to `lm_max_adapt_iters`. Remaining coordinates are clipped to the trust radius. Successful coordinates shrink lambda_k by `lm_lambda_shrink`.
    6. Apply (b, w) <- (b, w) - (Delta b, Delta w), recompute the zero-loss contributions, and continue until gradients and step sizes fall below `tol` or `n_iter` iterations have been attempted.

    The implementation keeps memory bounded by iterating rows in batches (`row_batch_size`) and classes in slabs (`class_slab_size`), never materializing tensors shaped (nnz, n_classes). All tensor traversals use the same event iterator so the loss and metric routines stay numerically aligned with training.
    """

    def __init__(
        self,
        *,
        n_latents: int,
        n_classes: int,
        tol: float = 1e-4,
        device: str = "cuda",
        n_iter: int = 100,
        ridge: float = 1e-8,
        class_slab_size: int = 8,
        row_batch_size: int = 1024,
        hessian_floor: float = 1e-4,
        lm_lambda_init: float = 1e-3,
        lm_lambda_shrink: float = 0.1,
        lm_lambda_grow: float = 10.0,
        lm_lambda_min: float = 1e-12,
        lm_lambda_max: float = 1e12,
        lm_max_update: float = 50.0,
        lm_max_adapt_iters: int = 6,
        iteration_hook: tp.Callable[[StepSummary], None] | None = None,
    ):
        if not 0.0 < lm_lambda_shrink < 1.0:
            msg = f"lm_lambda_shrink must be in (0,1), got {lm_lambda_shrink}."
            raise ValueError(msg)
        if not lm_lambda_grow > 1.0:
            msg = f"lm_lambda_grow must be >1, got {lm_lambda_grow}."
            raise ValueError(msg)
        if not lm_lambda_min > 0.0:
            msg = f"lm_lambda_min must be >0, got {lm_lambda_min}."
            raise ValueError(msg)
        if not lm_lambda_max >= lm_lambda_min:
            msg = (
                f"lm_lambda_max must be >= lm_lambda_min, "
                f"got {lm_lambda_max} < {lm_lambda_min}."
            )
            raise ValueError(msg)
        if not lm_max_update > 0:
            msg = f"lm_max_update must be >0, got {lm_max_update}."
            raise ValueError(msg)
        if lm_max_adapt_iters <= 0:
            msg = f"lm_max_adapt_iters must be positive, got {lm_max_adapt_iters}."
            raise ValueError(msg)

        self.n_latents = n_latents
        self.n_classes = n_classes
        self.tol = tol
        self.device = device
        self.n_iter = n_iter
        self.ridge = ridge  # L2 regularization strength
        self.class_slab_size = class_slab_size
        self.row_batch_size = row_batch_size
        self.logger = logging.getLogger("sparse1d")
        self.eps = 1e-7
        self.hessian_floor = hessian_floor
        self.lm_lambda_init = lm_lambda_init
        self.lm_lambda_shrink = lm_lambda_shrink
        self.lm_lambda_grow = lm_lambda_grow
        self.lm_lambda_min = lm_lambda_min
        self.lm_lambda_max = lm_lambda_max
        self.lm_max_update = lm_max_update
        self.lm_max_adapt_iters = lm_max_adapt_iters
        self.iteration_hook = iteration_hook
        self._quantile_targets = torch.tensor(
            (0.5, 0.95), dtype=torch.float32, device="cpu"
        )
        self._base_intercept = None

    def _init_best_latent_buffers(
        self, n_classes_slab: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Allocate trackers for the best latent per class within a slab."""

        best_loss = torch.full(
            (n_classes_slab,), float("inf"), device=device, dtype=torch.float32
        )
        best_latent = torch.full(
            (n_classes_slab,), -1, device=device, dtype=torch.int64
        )
        best_coef = torch.zeros((n_classes_slab,), device=device, dtype=torch.float32)
        best_intercept = torch.zeros(
            (n_classes_slab,), device=device, dtype=torch.float32
        )
        return best_loss, best_latent, best_coef, best_intercept

    def _init_lambda_slab(
        self, n_classes_slab: int, device: torch.device
    ) -> Float[Tensor, "n_latents c_b"]:
        """Create the LM damping tensor for a slab."""

        return torch.full(
            (self.n_latents, n_classes_slab),
            self.lm_lambda_init,
            device=device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def fit(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Float[Tensor, "n_samples n_classes"],
    ):
        assert x.layout == torch.sparse_csr

        n_samples, n_latents = x.shape
        assert n_latents == self.n_latents
        n_samples_f = float(n_samples)

        dd: dict[str, tp.Any] = dict(device=self.device, dtype=torch.float32)

        x = x.to(self.device)
        y = y.to(dtype=torch.float32, device="cpu")

        self.coef_ = torch.zeros((self.n_latents, self.n_classes), **dd)
        self.intercept_ = torch.zeros((self.n_latents, self.n_classes), **dd)

        pi = y.sum(dim=0)
        prevalence = torch.clamp(pi / float(n_samples), self.eps, 1 - self.eps)
        base_intercept = torch.logit(prevalence).to(self.device)
        self.intercept_.copy_(
            einops.repeat(base_intercept, "c -> l c", l=self.n_latents)
        )
        self._base_intercept = base_intercept

        nnz_per_latent = torch.bincount(x.col_indices(), minlength=self.n_latents)
        n_zeros_per_latent = (n_samples - nnz_per_latent).to(**dd).view(-1, 1)

        pi_device = pi.to(self.device)
        check_convergence = self.tol > 0

        for c0, c1 in saev.helpers.batched_idx(self.n_classes, self.class_slab_size):
            self.logger.debug(f"Processing classes {c0}:{c1} ({c1 - c0} classes)")

            y_slab = y[:, c0:c1]
            if self.device != "cpu":
                y_slab = y_slab.pin_memory()
            y_slab_device = y_slab.to(self.device, non_blocking=self.device != "cpu")
            pi_slab_device = pi_device[c0:c1]

            intercept_slab = self.intercept_[:, c0:c1]
            coef_slab = self.coef_[:, c0:c1]
            n_classes_slab = c1 - c0

            (
                best_loss_per_class,
                best_latent_per_class,
                best_coef_per_class,
                best_intercept_per_class,
            ) = self._init_best_latent_buffers(n_classes_slab, self.device)
            lambda_slab = self._init_lambda_slab(n_classes_slab, self.device)

            for it in range(self.n_iter):
                # Step 1: compute intercept-only statistics reused across the slab.
                mu_0 = torch.sigmoid(intercept_slab).clamp_(self.eps, 1 - self.eps)
                s_0 = mu_0 * (1 - mu_0)

                # Step 2: stream the sparse activations and accumulate Newton stats.
                stats = self._accumulate_slab_stats(
                    x=x,
                    y_slab=y_slab_device,
                    intercept_slab=intercept_slab,
                    coef_slab=coef_slab,
                    mu_0=mu_0,
                    s_0=s_0,
                    n_zeros_per_latent=n_zeros_per_latent,
                    pi_slab_device=pi_slab_device,
                    base_intercept_slab=self._base_intercept[c0:c1],
                )
                pos_zero = torch.clamp(
                    pi_slab_device.view(1, n_classes_slab) - stats.pos_nz, min=0.0
                )
                pos_zero = torch.minimum(pos_zero, n_zeros_per_latent)
                neg_zero = n_zeros_per_latent - pos_zero
                zero_loss = -(
                    pos_zero * torch.log(mu_0) + neg_zero * torch.log1p(-mu_0)
                )
                loss_slab = (stats.loss_nz + zero_loss) / n_samples_f
                mean_loss_value = loss_slab.mean().item()

                # Step 3: take a Levenberg–Marquardt step with trust-region safety.
                lambda_prev = lambda_slab.clone()
                lm_step = self._solve_lm_step(
                    stats.g0, stats.g1, stats.h0, stats.h1, stats.h2, lambda_prev
                )
                db = lm_step.db
                dw = lm_step.dw
                lambda_slab = lm_step.lambda_next

                curr_best_loss, curr_best_latent_idx = loss_slab.min(dim=0)
                improved_mask = torch.logical_or(
                    curr_best_latent_idx != best_latent_per_class,
                    curr_best_loss + 1e-6 < best_loss_per_class,
                )
                # Step 4: record the best latent per class for downstream reporting.
                best_update_summary = self._update_best_latents(
                    best_loss_per_class=best_loss_per_class,
                    best_latent_per_class=best_latent_per_class,
                    best_coef_per_class=best_coef_per_class,
                    best_intercept_per_class=best_intercept_per_class,
                    curr_best_loss=curr_best_loss,
                    curr_best_latent_idx=curr_best_latent_idx,
                    coef_slab=coef_slab,
                    intercept_slab=intercept_slab,
                    improved_mask=improved_mask,
                    global_class_start=c0,
                )

                # Step 5: apply the Newton step in-place.
                intercept_slab -= db
                coef_slab -= dw

                # Step 6: package everything we learned this iteration for logging.
                context = StepContext(
                    class_range=(c0, c1),
                    iteration=it,
                    mean_loss_value=mean_loss_value,
                    loss_slab=loss_slab,
                    stats=stats,
                    lambda_prev=lambda_prev,
                    lambda_next=lambda_slab,
                    db=db,
                    dw=dw,
                    improved_mask=improved_mask,
                    curr_best_loss=curr_best_loss,
                    curr_best_latent_idx=curr_best_latent_idx,
                    best_update_summary=best_update_summary,
                    coef_slab=coef_slab,
                    intercept_slab=intercept_slab,
                    nnz_per_latent=nnz_per_latent,
                    mu_0=mu_0,
                    n_zeros_per_latent=n_zeros_per_latent,
                    pi_slab_device=pi_slab_device,
                    lambda_step=lm_step,
                )

                max_grad_val, max_update_val = self._log_lm_step(context)

                if not check_convergence:
                    continue

                if math.isnan(max_grad_val) or math.isnan(max_update_val):
                    continue

                if max_grad_val < self.tol and max_update_val < self.tol:
                    self.logger.info(
                        "Classes %s:%s converged at iteration %s.", c0, c1, it
                    )
                    break
            else:
                self.logger.warning(
                    "Classes %s:%s did not converge after %s iterations",
                    c0,
                    c1,
                    self.n_iter,
                )

            if self.logger.isEnabledFor(logging.INFO):
                best_summary = []
                for offset, loss_value, latent_idx, coef_value, intercept_value in zip(
                    range(n_classes_slab),
                    best_loss_per_class.tolist(),
                    best_latent_per_class.tolist(),
                    best_coef_per_class.tolist(),
                    best_intercept_per_class.tolist(),
                ):
                    class_idx = c0 + offset
                    nnz_value = (
                        int(nnz_per_latent[latent_idx].item()) if latent_idx >= 0 else 0
                    )
                    best_summary.append(
                        (
                            f"class={class_idx} latent={latent_idx} "
                            f"loss={loss_value:.6f} coef={coef_value:.3f} "
                            f"intercept={intercept_value:.3f} nnz={nnz_value}"
                        )
                    )
                self.logger.info(
                    "Best latents after classes %s:%s -> %s",
                    c0,
                    c1,
                    " | ".join(best_summary),
                )

            del y_slab_device

    def _iter_sparse_events(
        self, x: Float[Tensor, "n_samples n_latents"]
    ) -> Iterator[SparseEventsBatch]:
        """Yield CSR non-zero spans for each row batch.

        Args:
            x: Sparse CSR matrix with shape `(n_samples, n_latents)` that we are streaming over.

        Yields:
            `SparseEventsBatch` objects describing the rows spanned by the batch,
            their latent indices, non-zero values, and global row indices.
        """

        crow_indices = x.crow_indices()
        col_indices = x.col_indices()
        values = x.values()
        n_samples = x.shape[0]

        for r0, r1 in saev.helpers.batched_idx(n_samples, self.row_batch_size):
            start = crow_indices[r0].item()
            end = crow_indices[r1].item()
            if start == end:
                continue

            lengths_per_row = crow_indices[r0 + 1 : r1 + 1] - crow_indices[r0:r1]
            row_idx_local = torch.repeat_interleave(
                torch.arange(
                    r1 - r0, device=x.device, dtype=torch.long, requires_grad=False
                ),
                lengths_per_row,
            )
            row_idx = row_idx_local + r0

            yield SparseEventsBatch(
                row_start=r0,
                row_end=r1,
                latent_idx=col_indices[start:end],
                values=values[start:end],
                row_idx=row_idx,
            )

    def _accumulate_slab_stats(
        self,
        *,
        x: Float[Tensor, "n_samples n_latents"],
        y_slab: Float[Tensor, "n_samples c_b"],
        intercept_slab: Float[Tensor, "n_latents c_b"],
        coef_slab: Float[Tensor, "n_latents c_b"],
        mu_0: Float[Tensor, "n_latents c_b"],
        s_0: Float[Tensor, "n_latents c_b"],
        n_zeros_per_latent: Float[Tensor, "n_latents 1"],
        pi_slab_device: Tensor,
        base_intercept_slab: Float[Tensor, " c_b"],
    ) -> SlabStats:
        """Compute per-latent Newton statistics for a class slab.

        Args:
            x: Sparse CSR activations for all samples.
            y_slab: Dense label matrix for the current class slab.
            intercept_slab: Intercept parameters for the slab.
            coef_slab: Coefficient parameters for the slab.
            mu_0: Sigmoid of intercepts reused for zero activations.
            s_0: Variance term `mu_0 * (1 - mu_0)` for zeros.
            n_zeros_per_latent: Count of implicit zeros per latent.
            pi_slab_device: Positive label counts per class on device.
            base_intercept_slab: Baseline intercept (logit prevalence) per class.

        Returns:
            SlabStats populated with gradients, Hessians, and auxiliary sums.
        """

        mu_nz = torch.zeros_like(intercept_slab)
        g1 = torch.zeros_like(intercept_slab)
        h0 = torch.zeros_like(intercept_slab)
        h1 = torch.zeros_like(intercept_slab)
        h2 = torch.zeros_like(intercept_slab)
        loss_nz = torch.zeros_like(intercept_slab)
        pos_nz = torch.zeros_like(intercept_slab)

        for batch in self._iter_sparse_events(x):
            values_E1 = batch.values[:, None]

            eta_EC = (
                intercept_slab[batch.latent_idx]
                + coef_slab[batch.latent_idx] * values_E1
            )
            mu_EC = torch.sigmoid(eta_EC).clamp_(self.eps, 1 - self.eps)
            s_EC = mu_EC * (1 - mu_EC)

            y_nz_EC = y_slab[batch.row_idx]
            residual_EC = mu_EC - y_nz_EC

            mu_nz.index_add_(0, batch.latent_idx, mu_EC)
            g1.index_add_(0, batch.latent_idx, residual_EC * values_E1)
            h0.index_add_(0, batch.latent_idx, s_EC)
            h1.index_add_(0, batch.latent_idx, s_EC * values_E1)
            h2.index_add_(0, batch.latent_idx, s_EC * (values_E1**2))
            nz_loss = -(
                y_nz_EC * torch.log(mu_EC) + (1 - y_nz_EC) * torch.log1p(-mu_EC)
            )
            loss_nz.index_add_(0, batch.latent_idx, nz_loss)
            pos_nz.index_add_(0, batch.latent_idx, y_nz_EC)

        g0 = mu_nz + n_zeros_per_latent * mu_0 - pi_slab_device.view(1, -1)
        g1 = g1 + self.ridge * coef_slab
        g0 = g0 + self.ridge * (intercept_slab - base_intercept_slab.view(1, -1))
        h0 = h0 + n_zeros_per_latent * s_0 + self.ridge
        h2 = h2 + self.ridge

        return SlabStats(
            g0=g0,
            g1=g1,
            h0=h0,
            h1=h1,
            h2=h2,
            loss_nz=loss_nz,
            pos_nz=pos_nz,
            mu_nz=mu_nz,
        )

    def _solve_lm_step(
        self,
        g0: Float[Tensor, "n_latents c_b"],
        g1: Float[Tensor, "n_latents c_b"],
        h0: Float[Tensor, "n_latents c_b"],
        h1: Float[Tensor, "n_latents c_b"],
        h2: Float[Tensor, "n_latents c_b"],
        lambda_prev: Float[Tensor, "n_latents c_b"],
    ) -> StepResult:
        """Solve the LM-damped Newton system and adapt the damping parameter.

        Args:
            g0: Gradients w.r.t. intercepts.
            g1: Gradients w.r.t. weights.
            h0: Intercept-intercept Hessian entries.
            h1: Intercept-weight Hessian entries.
            h2: Weight-weight Hessian entries.
            lambda_prev: Damping parameter from the previous iteration.

        Returns:
            LMStepResult containing the Newton step, Hessians, and next damping.
        """

        det_h_raw = h0 * h2 - h1 * h1
        h0_clamped = torch.clamp(h0, min=self.hessian_floor)
        h2_clamped = torch.clamp(h2, min=self.hessian_floor)

        lambda_curr = torch.clamp(
            lambda_prev, min=self.lm_lambda_min, max=self.lm_lambda_max
        )

        db_result = torch.zeros_like(g0)
        dw_result = torch.zeros_like(g0)
        h0_eff_result = torch.zeros_like(g0)
        h2_eff_result = torch.zeros_like(g0)
        det_result = torch.zeros_like(g0)
        lambda_applied = lambda_curr.clone()
        success_mask = torch.zeros_like(g0, dtype=torch.bool)

        for _ in range(self.lm_max_adapt_iters):
            h0_eff = h0_clamped + lambda_curr
            h2_eff = h2_clamped + lambda_curr
            det_h = h0_eff * h2_eff - h1 * h1
            det_h_sign = det_h.sign()
            det_h_sign = torch.where(
                det_h_sign == 0, torch.ones_like(det_h_sign), det_h_sign
            )
            det_h = torch.where(det_h.abs() < 1e-12, det_h_sign * 1e-12, det_h)

            db = (h2_eff * g0 - h1 * g1) / det_h
            dw = (-h1 * g0 + h0_eff * g1) / det_h

            step_mag = torch.maximum(db.abs(), dw.abs())
            predicted_reduction_iter = 0.5 * (db * g0 + dw * g1)
            finite_mask = torch.isfinite(db) & torch.isfinite(dw)
            success_iter = (
                finite_mask
                & (step_mag <= self.lm_max_update)
                & (predicted_reduction_iter > 0.0)
            )
            new_success = success_iter & (~success_mask)

            if bool(new_success.any()):
                db_result[new_success] = db[new_success]
                dw_result[new_success] = dw[new_success]
                h0_eff_result[new_success] = h0_eff[new_success]
                h2_eff_result[new_success] = h2_eff[new_success]
                det_result[new_success] = det_h[new_success]
                lambda_applied[new_success] = lambda_curr[new_success]

            success_mask = success_mask | success_iter
            if success_mask.all():
                break

            failure_mask = ~success_iter
            if not bool(failure_mask.any()):
                break

            lambda_curr = torch.where(
                failure_mask,
                torch.clamp(
                    lambda_curr * self.lm_lambda_grow,
                    min=self.lm_lambda_min,
                    max=self.lm_lambda_max,
                ),
                lambda_curr,
            )

        remaining = ~success_mask
        clipped_mask = remaining.clone()

        if bool(remaining.any()):
            h0_eff = h0_clamped + lambda_curr
            h2_eff = h2_clamped + lambda_curr
            det_h = h0_eff * h2_eff - h1 * h1
            det_h_sign = det_h.sign()
            det_h_sign = torch.where(
                det_h_sign == 0, torch.ones_like(det_h_sign), det_h_sign
            )
            det_h = torch.where(det_h.abs() < 1e-12, det_h_sign * 1e-12, det_h)

            db = (h2_eff * g0 - h1 * g1) / det_h
            dw = (-h1 * g0 + h0_eff * g1) / det_h

            step_mag = torch.maximum(db.abs(), dw.abs())
            finite_mask = torch.isfinite(db) & torch.isfinite(dw)
            scale = torch.ones_like(step_mag)
            scale = torch.where(
                step_mag > self.lm_max_update,
                self.lm_max_update / (step_mag + 1e-12),
                scale,
            )
            scale = torch.where(finite_mask, scale, torch.zeros_like(scale))
            db = torch.where(finite_mask, db * scale, torch.zeros_like(db))
            dw = torch.where(finite_mask, dw * scale, torch.zeros_like(dw))

            clipped_mask = remaining & (
                (step_mag > self.lm_max_update) | (~finite_mask)
            )

            db_result[remaining] = db[remaining]
            dw_result[remaining] = dw[remaining]
            h0_eff_result[remaining] = h0_eff[remaining]
            h2_eff_result[remaining] = h2_eff[remaining]
            det_result[remaining] = det_h[remaining]
            lambda_applied[remaining] = lambda_curr[remaining]

        predicted_reduction = 0.5 * (db_result * g0 + dw_result * g1)

        lambda_success = torch.clamp(
            lambda_applied * self.lm_lambda_shrink,
            min=self.lm_lambda_min,
            max=self.lm_lambda_max,
        )
        lambda_fail = torch.clamp(
            lambda_applied * self.lm_lambda_grow,
            min=self.lm_lambda_min,
            max=self.lm_lambda_max,
        )
        lambda_next = torch.where(success_mask, lambda_success, lambda_fail)

        return StepResult(
            db=db_result,
            dw=dw_result,
            clipped_mask=clipped_mask,
            det_h=det_result,
            det_h_raw=det_h_raw,
            h0_eff=h0_eff_result,
            h2_eff=h2_eff_result,
            lambda_next=lambda_next,
            predicted_reduction=predicted_reduction,
        )

    def _update_best_latents(
        self,
        *,
        best_loss_per_class: Tensor,
        best_latent_per_class: Tensor,
        best_coef_per_class: Tensor,
        best_intercept_per_class: Tensor,
        curr_best_loss: Tensor,
        curr_best_latent_idx: Tensor,
        coef_slab: Float[Tensor, "n_latents c_b"],
        intercept_slab: Float[Tensor, "n_latents c_b"],
        improved_mask: Tensor,
        global_class_start: int,
    ) -> list[str]:
        """Update running best latents per class and summarize improvements."""

        if not bool(improved_mask.any()):
            return []

        improved_offsets = torch.nonzero(improved_mask, as_tuple=False).flatten()
        best_loss_per_class.index_copy_(
            0, improved_offsets, curr_best_loss.index_select(0, improved_offsets)
        )
        best_latent_per_class.index_copy_(
            0,
            improved_offsets,
            curr_best_latent_idx.index_select(0, improved_offsets),
        )

        gathered_latents = curr_best_latent_idx.index_select(0, improved_offsets)
        best_coef_values = coef_slab[gathered_latents, improved_offsets]
        best_intercept_values = intercept_slab[gathered_latents, improved_offsets]

        best_coef_per_class.index_copy_(0, improved_offsets, best_coef_values)
        best_intercept_per_class.index_copy_(0, improved_offsets, best_intercept_values)

        summaries: list[str] = []
        losses_improved = curr_best_loss.index_select(0, improved_offsets)
        limit = min(4, int(improved_offsets.numel()))
        for offset_tensor, latent_tensor, loss_tensor in zip(
            improved_offsets[:limit].tolist(),
            gathered_latents[:limit].tolist(),
            losses_improved[:limit].tolist(),
        ):
            class_idx = global_class_start + offset_tensor
            summaries.append(f"{class_idx}->{latent_tensor} loss={loss_tensor:.6f}")

        remaining = int(improved_offsets.numel())
        if remaining > limit:
            summaries.append(f"+{remaining - limit} more")

        return summaries

    def _log_lm_step(self, ctx: StepContext) -> tuple[float, float]:
        """Emit diagnostics for a single LM iteration.

        Args:
            ctx: Snapshot containing tensors and metadata for logging.

        Returns:
            tuple[float, float]: Max gradient magnitude and max update magnitude.
        """
        logger = self.logger
        need_info = logger.isEnabledFor(logging.INFO)
        need_debug = logger.isEnabledFor(logging.DEBUG)

        stats = ctx.stats
        abs_g0 = stats.g0.abs()
        abs_g1 = stats.g1.abs()
        abs_db = ctx.db.abs()
        abs_dw = ctx.dw.abs()

        g0_flat = abs_g0.view(-1)
        g1_flat = abs_g1.view(-1)
        db_flat = abs_db.view(-1)
        dw_flat = abs_dw.view(-1)

        max_g0_val, max_g0_idx = g0_flat.max(dim=0)
        max_g1_val, max_g1_idx = g1_flat.max(dim=0)
        max_db_val, max_db_idx = db_flat.max(dim=0)
        max_dw_val, max_dw_idx = dw_flat.max(dim=0)

        max_grad_val = torch.stack((max_g0_val, max_g1_val)).max().item()
        max_update_val = torch.stack((max_db_val, max_dw_val)).max().item()

        lambda_min_value = ctx.lambda_next.min().item()
        lambda_max_value = ctx.lambda_next.max().item()

        clipped_count = int(ctx.lambda_step.clipped_mask.sum().item())

        if need_info:
            if clipped_count:
                logger.info(
                    (
                        "Classes %s:%s at iteration %s: loss=%.6f, grad=%s, update=%s "
                        "lambda=[%.3e, %.3e] clipped=%d"
                    ),
                    ctx.class_range[0],
                    ctx.class_range[1],
                    ctx.iteration,
                    ctx.mean_loss_value,
                    max_grad_val,
                    max_update_val,
                    lambda_min_value,
                    lambda_max_value,
                    clipped_count,
                )
            else:
                logger.info(
                    (
                        "Classes %s:%s at iteration %s: loss=%.6f, grad=%s, update=%s "
                        "lambda=[%.3e, %.3e]"
                    ),
                    ctx.class_range[0],
                    ctx.class_range[1],
                    ctx.iteration,
                    ctx.mean_loss_value,
                    max_grad_val,
                    max_update_val,
                    lambda_min_value,
                    lambda_max_value,
                )
            if ctx.best_update_summary:
                logger.info(
                    "improved_best_latents %s", ", ".join(ctx.best_update_summary)
                )

        summary = StepSummary(
            class_range=ctx.class_range,
            iteration=ctx.iteration,
            mean_loss=ctx.mean_loss_value,
            max_grad=max_grad_val,
            max_update=max_update_val,
            lambda_min=lambda_min_value,
            lambda_max=lambda_max_value,
            predicted_reduction_max=ctx.lambda_step.predicted_reduction.max().item(),
            predicted_reduction_mean=ctx.lambda_step.predicted_reduction.mean().item(),
        )
        if self.iteration_hook is not None:
            self.iteration_hook(summary)

        result = (max_grad_val, max_update_val)
        if not need_debug:
            return result

        grad_source_is_g0 = max_g0_val >= max_g1_val
        update_source_is_db = max_db_val >= max_dw_val

        grad_idx = torch.unravel_index(
            max_g0_idx if grad_source_is_g0 else max_g1_idx, abs_g0.shape
        )
        update_idx = torch.unravel_index(
            max_db_idx if update_source_is_db else max_dw_idx, abs_db.shape
        )

        grad_latent_idx = int(grad_idx[0])
        grad_class_offset = int(grad_idx[1])
        update_latent_idx = int(update_idx[0])
        update_class_offset = int(update_idx[1])

        grad_class_idx = ctx.class_range[0] + grad_class_offset
        update_class_idx = ctx.class_range[0] + update_class_offset

        grad_coef_value = ctx.coef_slab[grad_latent_idx, grad_class_offset].item()
        grad_intercept_value = ctx.intercept_slab[
            grad_latent_idx, grad_class_offset
        ].item()
        grad_loss_value = ctx.loss_slab[grad_latent_idx, grad_class_offset].item()
        grad_det_raw = ctx.lambda_step.det_h_raw[
            grad_latent_idx, grad_class_offset
        ].item()
        grad_det_clamped = ctx.lambda_step.det_h[
            grad_latent_idx, grad_class_offset
        ].item()
        grad_nnz = ctx.nnz_per_latent[grad_latent_idx].item()
        grad_g0_value = stats.g0[grad_latent_idx, grad_class_offset].item()
        grad_mu_nz_value = stats.mu_nz[grad_latent_idx, grad_class_offset].item()
        grad_mu0_value = ctx.mu_0[grad_latent_idx, grad_class_offset].item()
        grad_n_zeros_value = ctx.n_zeros_per_latent[grad_latent_idx, 0].item()
        grad_g0_zero_value = grad_n_zeros_value * grad_mu0_value
        grad_pi_value = ctx.pi_slab_device[grad_class_offset].item()
        grad_g1_value = stats.g1[grad_latent_idx, grad_class_offset].item()
        grad_g1_ridge_value = self.ridge * grad_coef_value
        grad_g1_nz_value = grad_g1_value - grad_g1_ridge_value
        grad_h0_value = ctx.lambda_step.h0_eff[
            grad_latent_idx, grad_class_offset
        ].item()
        grad_h1_value = stats.h1[grad_latent_idx, grad_class_offset].item()
        grad_h2_value = ctx.lambda_step.h2_eff[
            grad_latent_idx, grad_class_offset
        ].item()
        grad_lambda_value = ctx.lambda_prev[grad_latent_idx, grad_class_offset].item()

        update_coef_value = ctx.coef_slab[update_latent_idx, update_class_offset].item()
        update_intercept_value = ctx.intercept_slab[
            update_latent_idx, update_class_offset
        ].item()
        update_loss_value = ctx.loss_slab[update_latent_idx, update_class_offset].item()
        update_det_raw = ctx.lambda_step.det_h_raw[
            update_latent_idx, update_class_offset
        ].item()
        update_det_clamped = ctx.lambda_step.det_h[
            update_latent_idx, update_class_offset
        ].item()
        update_nnz = ctx.nnz_per_latent[update_latent_idx].item()
        update_g0_value = stats.g0[update_latent_idx, update_class_offset].item()
        update_mu_nz_value = stats.mu_nz[update_latent_idx, update_class_offset].item()
        update_mu0_value = ctx.mu_0[update_latent_idx, update_class_offset].item()
        update_n_zeros_value = ctx.n_zeros_per_latent[update_latent_idx, 0].item()
        update_g0_zero_value = update_n_zeros_value * update_mu0_value
        update_pi_value = ctx.pi_slab_device[update_class_offset].item()
        update_g1_value = stats.g1[update_latent_idx, update_class_offset].item()
        update_g1_ridge_value = self.ridge * update_coef_value
        update_g1_nz_value = update_g1_value - update_g1_ridge_value
        update_h0_value = ctx.lambda_step.h0_eff[
            update_latent_idx, update_class_offset
        ].item()
        update_h1_value = stats.h1[update_latent_idx, update_class_offset].item()
        update_h2_value = ctx.lambda_step.h2_eff[
            update_latent_idx, update_class_offset
        ].item()
        update_lambda_value = ctx.lambda_prev[
            update_latent_idx, update_class_offset
        ].item()
        update_db_value = ctx.db[update_latent_idx, update_class_offset].item()
        update_dw_value = ctx.dw[update_latent_idx, update_class_offset].item()

        quantile_targets = self._quantile_targets.to(
            device=stats.g0.device, dtype=stats.g0.dtype
        )
        g0_quantiles = torch.quantile(g0_flat, quantile_targets).tolist()
        g1_quantiles = torch.quantile(g1_flat, quantile_targets).tolist()
        loss_quantiles = torch.quantile(
            ctx.loss_slab.view(-1), quantile_targets
        ).tolist()

        logger.debug(
            "Classes %s:%s predicted_reduction max=%s mean=%s",
            ctx.class_range[0],
            ctx.class_range[1],
            float(ctx.lambda_step.predicted_reduction.max().item()),
            float(ctx.lambda_step.predicted_reduction.mean().item()),
        )
        logger.debug(
            "worst_grad source=%s latent=%s class=%s nnz=%s coef=%s intercept=%s "
            "loss=%s det_raw=%s det_clamped=%s g0_q50=%s g0_q95=%s g1_q50=%s "
            "g1_q95=%s loss_q50=%s loss_q95=%s",
            "g0" if grad_source_is_g0 else "g1",
            grad_latent_idx,
            grad_class_idx,
            grad_nnz,
            grad_coef_value,
            grad_intercept_value,
            grad_loss_value,
            grad_det_raw,
            grad_det_clamped,
            g0_quantiles[0],
            g0_quantiles[1],
            g1_quantiles[0],
            g1_quantiles[1],
            loss_quantiles[0],
            loss_quantiles[1],
        )
        logger.debug(
            "worst_update source=%s latent=%s class=%s nnz=%s coef=%s intercept=%s "
            "loss=%s det_raw=%s det_clamped=%s delta=%s",
            "db" if update_source_is_db else "dw",
            update_latent_idx,
            update_class_idx,
            update_nnz,
            update_coef_value,
            update_intercept_value,
            update_loss_value,
            update_det_raw,
            update_det_clamped,
            (max_db_val if update_source_is_db else max_dw_val).item(),
        )
        logger.debug(
            "worst_grad_diagnostics g0=%s g0_nz=%s g0_zero=%s g0_pi=%s g1=%s "
            "g1_nz=%s g1_ridge=%s h0=%s h1=%s h2=%s lambda=%s",
            grad_g0_value,
            grad_mu_nz_value,
            grad_g0_zero_value,
            grad_pi_value,
            grad_g1_value,
            grad_g1_nz_value,
            grad_g1_ridge_value,
            grad_h0_value,
            grad_h1_value,
            grad_h2_value,
            grad_lambda_value,
        )
        logger.debug(
            "worst_update_diagnostics g0=%s g0_nz=%s g0_zero=%s g0_pi=%s g1=%s "
            "g1_nz=%s g1_ridge=%s h0=%s h1=%s h2=%s lambda=%s db=%s dw=%s",
            update_g0_value,
            update_mu_nz_value,
            update_g0_zero_value,
            update_pi_value,
            update_g1_value,
            update_g1_nz_value,
            update_g1_ridge_value,
            update_h0_value,
            update_h1_value,
            update_h2_value,
            update_lambda_value,
            update_db_value,
            update_dw_value,
        )

        return result

    def _compute_loss_slab(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y_slab: Tensor,  # Float[Tensor, "n_samples c_b"]
        c0: int,
        c1: int,
        n_samples: int,
        pi_slab: Tensor,
        n_zeros_per_latent: Tensor,
    ) -> Tensor:
        """Compute loss for a class slab using event chunking."""

        loss = torch.zeros(
            (self.n_latents, c1 - c0), dtype=torch.float32, device=self.device
        )
        pos_nz = torch.zeros_like(loss)

        intercept_slab = self.intercept_[:, c0:c1]
        coef_slab = self.coef_[:, c0:c1]

        mu_0 = torch.sigmoid(intercept_slab).clamp_(self.eps, 1 - self.eps)

        for batch in self._iter_sparse_events(x):
            cols_chunk = batch.latent_idx
            vals_chunk = batch.values
            row_idx_chunk = batch.row_idx
            y_nz = y_slab[row_idx_chunk]

            vals_expanded = vals_chunk.view(-1, 1)
            b_cols = intercept_slab[cols_chunk]
            w_cols = coef_slab[cols_chunk]
            eta = b_cols + w_cols * vals_expanded
            mu = torch.sigmoid(eta).clamp_(self.eps, 1 - self.eps)

            nz_loss = -(y_nz * torch.log(mu) + (1 - y_nz) * torch.log1p(-mu))
            loss.index_add_(0, cols_chunk, nz_loss)
            pos_nz.index_add_(0, cols_chunk, y_nz)

        pos_zero = torch.clamp(pi_slab.view(1, c1 - c0) - pos_nz, min=0.0)
        pos_zero = torch.minimum(pos_zero, n_zeros_per_latent)
        neg_zero = n_zeros_per_latent - pos_zero

        zero_loss = -(pos_zero * torch.log(mu_0) + neg_zero * torch.log1p(-mu_0))
        loss = loss + zero_loss

        return loss / float(n_samples)

    def _compute_loss(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Float[Tensor, "n_samples n_classes"],
    ) -> Float[Tensor, "n_latents n_classes"]:
        """Compute negative log-likelihood loss for all (latent, class) pairs using chunking."""
        n_samples = x.shape[0]
        loss = torch.zeros(
            (self.n_latents, self.n_classes), dtype=torch.float32, device=self.device
        )

        # Get CSR components
        col_indices = x.col_indices()
        nnz_per_latent = torch.bincount(col_indices, minlength=self.n_latents)
        n_zeros_per_latent = (
            (n_samples - nnz_per_latent)
            .to(device=self.device, dtype=torch.float32)
            .view(-1, 1)
        )

        # Compute pi on CPU
        pi = y.sum(dim=0, dtype=torch.float64).to(torch.float32)

        # Process classes in chunks
        for c0 in range(0, self.n_classes, self.class_slab_size):
            c1 = min(c0 + self.class_slab_size, self.n_classes)

            # Move class slab to device
            y_slab = y[:, c0:c1].to(self.device).to(torch.float32)
            pi_slab = pi[c0:c1].to(self.device)

            # Compute loss for this slab
            loss[:, c0:c1] = self._compute_loss_slab(
                x,
                y_slab,
                c0,
                c1,
                n_samples,
                pi_slab,
                n_zeros_per_latent,
            )

            del y_slab, pi_slab

        return loss

    @torch.no_grad()
    def loss_matrix(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Bool[Tensor, "n_samples n_classes"],
    ) -> Float[Tensor, "n_latents n_classes"]:
        """Returns the NLL loss matrix. Cheap to compute because we just use intercept_ and coef_ to recalculate loss."""
        sklearn.utils.validation.check_is_fitted(self, "intercept_")
        sklearn.utils.validation.check_is_fitted(self, "coef_")

        # Move data to device and ensure correct types
        x = x.to(self.device)
        y = y.to(self.device).float()

        return self._compute_loss(x, y)

    def _compute_confusion_slab(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y_slab: Tensor,
        c0: int,
        c1: int,
        pi_slab: Tensor,
        n_zeros_per_latent: Tensor,
        threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute confusion-matrix counts for a class slab using event chunking."""

        tp = torch.zeros(
            (self.n_latents, c1 - c0), dtype=torch.float32, device=self.device
        )
        fp = torch.zeros_like(tp)
        tn = torch.zeros_like(tp)
        fn = torch.zeros_like(tp)
        pos_nz = torch.zeros_like(tp)

        intercept_slab = self.intercept_[:, c0:c1]
        coef_slab = self.coef_[:, c0:c1]

        mu_0 = torch.sigmoid(intercept_slab).clamp_(self.eps, 1 - self.eps)
        pred_zero = mu_0 > threshold

        for batch in self._iter_sparse_events(x):
            cols_chunk = batch.latent_idx
            vals_chunk = batch.values
            row_idx_chunk = batch.row_idx
            y_nz = y_slab[row_idx_chunk]
            y_nz_bool = y_nz > 0.5

            vals_expanded = vals_chunk.view(-1, 1)
            b_cols = intercept_slab[cols_chunk]
            w_cols = coef_slab[cols_chunk]
            mu = torch.sigmoid(b_cols + w_cols * vals_expanded)
            pred_nz_bool = mu > threshold

            tp_chunk = torch.logical_and(pred_nz_bool, y_nz_bool).to(torch.float32)
            fp_chunk = torch.logical_and(pred_nz_bool, ~y_nz_bool).to(torch.float32)
            fn_chunk = torch.logical_and(~pred_nz_bool, y_nz_bool).to(torch.float32)
            tn_chunk = torch.logical_and(~pred_nz_bool, ~y_nz_bool).to(torch.float32)

            tp.index_add_(0, cols_chunk, tp_chunk)
            fp.index_add_(0, cols_chunk, fp_chunk)
            fn.index_add_(0, cols_chunk, fn_chunk)
            tn.index_add_(0, cols_chunk, tn_chunk)
            pos_nz.index_add_(0, cols_chunk, y_nz)

        pos_zero = torch.clamp(pi_slab.view(1, c1 - c0) - pos_nz, min=0.0)
        pos_zero = torch.minimum(pos_zero, n_zeros_per_latent)
        neg_zero = n_zeros_per_latent - pos_zero

        zero_mask = pred_zero.to(torch.float32)
        tp_zero = zero_mask * pos_zero
        fp_zero = zero_mask * neg_zero
        fn_zero = (1.0 - zero_mask) * pos_zero
        tn_zero = (1.0 - zero_mask) * neg_zero

        tp = tp + tp_zero
        fp = fp + fp_zero
        fn = fn + fn_zero
        tn = tn + tn_zero

        return tp, fp, tn, fn

    @torch.no_grad()
    def loss_matrix_with_aux(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Bool[Tensor, "n_samples n_classes"],
        threshold: float = 0.5,
    ) -> tuple[
        Float[Tensor, "n_latents n_classes"],
        Float[Tensor, "n_latents n_classes"],
        Float[Tensor, "n_latents n_classes"],
        Float[Tensor, "n_latents n_classes"],
        Float[Tensor, "n_latents n_classes"],
    ]:
        """Returns the NLL loss matrix and confusion-matrix counts for each (latent, class) pair."""
        sklearn.utils.validation.check_is_fitted(self, "intercept_")
        sklearn.utils.validation.check_is_fitted(self, "coef_")

        if not (0.0 < threshold < 1.0):
            msg = f"threshold must be between 0 and 1, got {threshold}."
            raise ValueError(msg)

        # Move sparse matrix to device
        x = x.to(self.device)
        n_samples = x.shape[0]

        # Compute loss using chunked implementation
        loss = self._compute_loss(x, y.to(torch.float32))

        # Count nnz per latent
        col_indices = x.col_indices()
        nnz_per_latent = torch.bincount(col_indices, minlength=self.n_latents)
        n_zeros_per_latent = (
            (n_samples - nnz_per_latent)
            .to(device=self.device, dtype=torch.float32)
            .view(-1, 1)
        )

        # Compute confusion counts using chunked implementation
        pi = y.to(torch.float32).sum(dim=0, dtype=torch.float64).to(torch.float32)

        tp = torch.zeros(
            (self.n_latents, self.n_classes), dtype=torch.float32, device=self.device
        )
        fp = torch.zeros_like(tp)
        tn = torch.zeros_like(tp)
        fn = torch.zeros_like(tp)

        for c0 in range(0, self.n_classes, self.class_slab_size):
            c1 = min(c0 + self.class_slab_size, self.n_classes)

            y_slab = y[:, c0:c1].to(self.device).to(torch.float32)
            pi_slab = pi[c0:c1].to(self.device)

            tp_chunk, fp_chunk, tn_chunk, fn_chunk = self._compute_confusion_slab(
                x,
                y_slab,
                c0,
                c1,
                pi_slab,
                n_zeros_per_latent,
                threshold,
            )

            tp[:, c0:c1] = tp_chunk
            fp[:, c0:c1] = fp_chunk
            tn[:, c0:c1] = tn_chunk
            fn[:, c0:c1] = fn_chunk

            del y_slab, pi_slab

        return loss, tp, fp, tn, fn


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    run: pathlib.Path = pathlib.Path("./runs/abcdefg")
    """Run directory."""
    shards_dir: pathlib.Path = pathlib.Path("./shards/e967c008")
    """Shards directory."""
    # Optimization
    ridge: float = 1e-7
    """Ridge value."""
    debug: bool = False
    """Debug logging."""
    # Hardware
    device: str = "cuda"
    """Which accelerator to use."""
    mem_gb: int = 80
    """Node memory in GB."""
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 4.0
    """Slurm job length in hours."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to log Slurm job stdout/stderr."""


def sp_csr_to_pt(csr: scipy.sparse.csr_matrix, *, device: str) -> Tensor:
    return torch.sparse_csr_tensor(csr.indptr, csr.indices, csr.data, device=device)


@beartype.beartype
def worker_fn(cfg: Config) -> int:
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger("probe1d")

    logger.info("Started main().")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA device available, using CPU.")
        cfg = dataclasses.replace(cfg, device="cpu")

    if not (cfg.shards_dir / "labels.bin").exists():
        logger.error("--shards-dir %s doesn't have a labels.bin.", cfg.shards_dir)
        return 1

    run = saev.disk.Run(cfg.run)

    if not (run.inference / cfg.shards_dir.name).exists():
        logger.error(
            "Directory %s doesn't exist. Use inference.py to run inference.",
            run.inference / cfg.shards_dir.name,
        )
        return 1

    # Load metadata
    md = saev.data.Metadata.load(cfg.shards_dir)
    logger.info("Loaded metadata from %s.", cfg.shards_dir)

    # Load SAE activations (sparse matrix)
    token_acts = scipy.sparse.load_npz(
        run.inference / cfg.shards_dir.name / "token_acts.npz"
    )
    logger.info(
        "Loaded activations: shape=%s, nnz=%d.", token_acts.shape, token_acts.nnz
    )
    n_samples, n_latents = token_acts.shape
    token_acts = sp_csr_to_pt(token_acts, device=cfg.device)
    logger.info("Converted activations to Tensor on %s.", cfg.device)

    # Load patch labels from labels.bin
    labels = np.memmap(
        cfg.shards_dir / "labels.bin",
        mode="r",
        dtype=np.uint8,
        shape=(md.n_examples, md.content_tokens_per_example),
    )
    logger.info("Loaded labels: shape=%s.", labels.shape)

    # Flatten labels to (n_samples,) and convert to one-hot
    n_classes = int(labels.max()) + 1
    logger.info("Found %d classes in labels.", n_classes)

    # Convert to one-hot encoding
    y = np.zeros((n_samples, n_classes), dtype=float)
    y[np.arange(n_samples), labels.reshape(n_samples)] = 1.0
    y = torch.from_numpy(y)
    logger.info("Created one-hot labels: shape=%s.", y.shape)

    # Fit probe
    probe = Sparse1DProbe(
        n_latents=n_latents, n_classes=n_classes, device=cfg.device, ridge=cfg.ridge
    )
    logger.info("Fitting probe with %d latents and %d classes.", n_latents, n_classes)
    probe.fit(token_acts, y)
    logger.info("Fit probe.")

    # TODO: do this with a validation split
    loss, tp, fp, tn, fn = probe.loss_matrix_with_aux(token_acts, y.bool())

    out_fpath = run.inference / cfg.shards_dir.name / "probe1d_metrics.npz"
    out_fpath.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_fpath,
        loss=loss.cpu().numpy(),
        weights=probe.coef_.cpu().numpy(),
        biases=probe.intercept_.cpu().numpy(),
        tp=tp.cpu().numpy(),
        fp=fp.cpu().numpy(),
        tn=tn.cpu().numpy(),
        fn=fn.cpu().numpy(),
    )

    logger.info("Saved probe outputs to %s.", out_fpath)

    return 0


@beartype.beartype
def cli(cfg: tp.Annotated[Config, tyro.conf.arg(name="")]) -> int:
    """
    Fit a sparse 1D probe to each combination of SAE latent and segmentation class.

    Args:
        cfg: Config.
    """
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logger = logging.getLogger("probe1d")

    logger.info("Started cli().")

    if cfg.slurm_acct:
        import submitit

        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        n_cpus = 8
        if cfg.mem_gb // 10 > n_cpus:
            logger.info(
                "Using %d CPUs instead of %d to get more RAM.", cfg.mem_gb // 10, n_cpus
            )
            n_cpus = cfg.mem_gb // 10
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=n_cpus,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
        job = executor.submit(worker_fn, cfg)
        logger.info("Running job '%s'.", job.job_id)
        job.result()

    else:
        worker_fn(cfg)

    logger.info("Jobs done.")
    return 0
