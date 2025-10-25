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

import beartype
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


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SlabStats:
    mu_nz: Float[Tensor, "n_latents c_b"]
    g1: Float[Tensor, "n_latents c_b"]
    h0: Float[Tensor, "n_latents c_b"]
    h1: Float[Tensor, "n_latents c_b"]
    h2: Float[Tensor, "n_latents c_b"]
    loss_nz: Float[Tensor, "n_latents c_b"]
    pos_nz: Float[Tensor, "n_latents c_b"]


@beartype.beartype
class RowChunk(tp.NamedTuple):
    """Bounds and cached row indices for a contiguous CSR slice containing `end_ptr - start_ptr` events."""

    row_start: int
    row_end: int
    start_ptr: int
    end_ptr: int
    row_idx_cpu: Int[Tensor, " event"]


@jaxtyped(typechecker=beartype.beartype)
class SparseEventsBatch(tp.NamedTuple):
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
class Reference1DProbe(sklearn.base.BaseEstimator):
    """Dense baseline that mirrors the probe1d trust-region spec step-for-step.

    The implementation favors clarity over speed so it is easy to audit against
    docs/research/issues/probe1d-trust-region.md. It uses the two-parameter
    Levenberg-Marquardt update with mean-scaled statistics, clamps steps to the
    logit budget, enforces monotonic loss reduction, and falls back to a tiny
    projected gradient step whenever the LM loop fails. Additional termination
    guards (predicted reduction, curvature, relative decrease) match the spec so
    that future readers can validate correctness without cross-referencing the
    sparse implementation.
    """

    def __init__(
        self,
        ridge: float = 1e-8,
        tol: float = 1e-6,
        max_iter: int = 200,
        lam_init: float = 1e-3,
        lam_shrink: float = 0.1,
        lam_grow: float = 10.0,
        delta_logit: float = 6.0,
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

        self.ridge = float(ridge)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.lam_init = float(lam_init)
        self.lam_shrink = float(lam_shrink)
        self.lam_grow = float(lam_grow)
        self.delta_logit = float(delta_logit)
        self.lam_min = 1e-12
        self.lam_max = 1e12
        self.eps = 1e-8
        self.tol_pred = 1e-12
        self.tol_pred_rel = 1e-6
        self.tol_curv = 1e-12
        self.fallback_step_scale = 1e-3

        self.intercept_: np.ndarray = np.zeros(1, dtype=float)
        self.coef_: np.ndarray = np.zeros(1, dtype=float)
        self.weight_: float | None = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self._fitted: bool = False

        self.logger = logging.getLogger("reference")

    def fit(self, X, y):
        """
        Dense, single (latent,class) solver.
        Accepts either a 1D array (n,) or a column vector (n,1).
        Ridge penalty: 0.5*(w^2 + (b - b0)^2)
        Trust region: elliptical ||DΔ||2<=δ with D=diag(1,qx).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            x = X.reshape(-1)
        else:
            if X.ndim != 2 or X.shape[1] != 1:
                msg = f"Reference1DProbe expects exactly one feature; received array with shape {X.shape}."
                raise ValueError(msg)
            x = X[:, 0]

        y = np.asarray(y, dtype=float).reshape(-1)
        if x.shape[0] != y.shape[0]:
            msg = f"x and y must have matching lengths; received {x.shape[0]} and {y.shape[0]}."
            raise ValueError(msg)
        if x.shape[0] == 0:
            msg = "x and y must contain at least one sample."
            raise ValueError(msg)
        if np.any((y < 0) | (y > 1)):
            msg = "y must contain only probabilities in [0, 1]."
            raise ValueError(msg)

        pi = y.mean()
        pi = np.clip(pi, self.eps, 1 - self.eps)
        b0 = np.log(pi / (1 - pi))

        b = float(b0)
        w = 0.0
        lam = self.lam_init
        # robust scale; 95th percentile of |x| is fine too
        nnz = x[x != 0]
        if nnz.size == 0:
            qx_value = 1.0
        else:
            qx_value = float(np.sqrt(np.mean(nnz.astype(float) ** 2)))
            if not np.isfinite(qx_value) or qx_value <= 1e-12:
                qx_value = 1.0
        qx_value = max(float(qx_value), 1e-12)

        self.qx_ = qx_value

        prev_pred = None
        prev_loss_before_step = None
        step_max = float("inf")
        grad_max = float("inf")
        prev_step_clipped = False

        def loss(b, w):
            mu = sigmoid(b + w * x)
            # Mean NLL + ridge
            return -np.mean(
                y * np.log(mu) + (1 - y) * np.log(1 - mu)
            ) + 0.5 * self.ridge * (w**2 + (b - b0) ** 2)

        loss_curr = loss(b, w)

        termination_reason: str | None = None

        for it in range(self.max_iter):
            rho = None
            if prev_pred is not None:
                actual = prev_loss_before_step - loss_curr
                rho = actual / max(prev_pred, 1e-18)
                grow = False
                shrink = False
                if not np.isfinite(rho):
                    grow = True
                else:
                    if rho >= 0.75 and not prev_step_clipped:
                        shrink = True
                    if rho <= 0.25:
                        grow = True
                if prev_step_clipped:
                    grow = True
                if grow:
                    lam = min(lam * self.lam_grow, self.lam_max)
                elif shrink:
                    lam = max(lam * self.lam_shrink, self.lam_min)

            z = b + w * x
            mu = sigmoid(z)
            s = mu * (1 - mu)
            r = mu - y

            g0 = r.mean() + self.ridge * (b - b0)
            g1 = np.mean(r * x) + self.ridge * w
            h0 = s.mean() + self.ridge
            h1 = np.mean(s * x)
            h2 = np.mean(s * x * x) + self.ridge

            grad_max = max(abs(g0), abs(g1))
            curv_mean = float(s.mean())
            if curv_mean < self.tol_curv:
                termination_reason = "curvature"
                break

            # LM step with retries
            tried = 0
            db = dw = 0.0
            step_clipped = False
            pred = 0.0
            loss_candidate = loss_curr
            step_found = False
            while tried < 6:
                step_clipped_local = False
                # scaled LM: H + lam * D^T D with D=diag(1,qx)
                h0_eff = h0 + lam * 1.0
                h2_eff = h2 + lam * (qx_value * qx_value)
                det = h0_eff * h2_eff - h1 * h1
                if abs(det) < 1e-18:
                    lam = min(lam * self.lam_grow, self.lam_max)
                    tried += 1
                    continue

                db = (h2_eff * g0 - h1 * g1) / det
                dw = (-h1 * g0 + h0_eff * g1) / det

                # elliptical trust region
                norm = np.sqrt(db * db + (qx_value * dw) * (qx_value * dw))
                if norm > self.delta_logit:
                    scale = self.delta_logit / (norm + 1e-18)
                    db *= scale
                    dw *= scale
                    step_clipped_local = True

                if not np.isfinite(db) or not np.isfinite(dw):
                    lam = min(lam * self.lam_grow, self.lam_max)
                    tried += 1
                    continue

                # predicted quadratic-model decrease (correct sign)
                pred_candidate = (
                    g0 * db
                    + g1 * dw
                    - 0.5 * (h0 * db * db + 2 * h1 * db * dw + h2 * dw * dw)
                )
                if not np.isfinite(pred_candidate) or pred_candidate <= 0:
                    lam = min(lam * self.lam_grow, self.lam_max)
                    tried += 1
                    continue
                b_try = b - db
                w_try = w - dw
                loss_try = loss(b_try, w_try)
                if loss_try > loss_curr + self.eps:
                    lam = min(lam * self.lam_grow, self.lam_max)
                    tried += 1
                    continue
                pred = pred_candidate
                loss_candidate = loss_try
                step_clipped = step_clipped_local
                step_found = True
                break

            if not step_found:
                grad_sq = g0 * g0 + g1 * g1
                if grad_sq <= self.eps:
                    db = 0.0
                    dw = 0.0
                    pred = 0.0
                    loss_candidate = loss_curr
                else:
                    step_scale = self.fallback_step_scale
                    for _ in range(10):
                        db = -g0 * step_scale
                        dw = -g1 * step_scale
                        norm = math.sqrt(db * db + (qx_value * dw) * (qx_value * dw))
                        if norm > self.delta_logit:
                            scale = self.delta_logit / (norm + 1e-18)
                            db *= scale
                            dw *= scale
                            step_clipped = True
                        pred = (
                            g0 * db
                            + g1 * dw
                            - 0.5 * (h0 * db * db + 2 * h1 * db * dw + h2 * dw * dw)
                        )
                        if not np.isfinite(pred) or pred < 0:
                            pred = 0.0
                        b_try = b - db
                        w_try = w - dw
                        loss_try = loss(b_try, w_try)
                        if loss_try <= loss_curr + self.eps:
                            loss_candidate = loss_try
                            break
                        step_scale *= 0.5
                    else:
                        db = 0.0
                        dw = 0.0
                        pred = 0.0
                        loss_candidate = loss_curr

            step_max_candidate = max(abs(db), abs(dw))
            self.logger.info(
                "iter=%d grad_max=%.3e, step_max=%.3e lambda_mean=%.3e",
                it,
                grad_max,
                step_max_candidate,
                lam,
            )
            if pred <= self.tol_pred or pred <= self.tol_pred_rel * (
                abs(loss_curr) + 1e-12
            ):
                termination_reason = "predicted_reduction"
                break

            # apply step
            b_new = b - db
            w_new = w - dw
            loss_new = loss_candidate

            prev_pred = pred
            prev_loss_before_step = loss_curr
            loss_curr = loss_new
            b, w = b_new, w_new
            prev_step_clipped = step_clipped

            step_max = step_max_candidate
            if grad_max < self.tol and step_max < self.tol:
                self.weight_ = float(w)
                self.converged_ = True
                self.n_iter_ = it + 1
                self.coef_[0] = float(w)
                self.intercept_[0] = float(b)
                self._fitted = True
                return self

        if termination_reason is not None:
            self.weight_ = float(w)
            self.converged_ = True
            self.n_iter_ = it + 1
            self.coef_[0] = float(w)
            self.intercept_[0] = float(b)
            self._fitted = True
            self.logger.info("termination: %s", termination_reason)
            return self

        self.weight_ = float(w)
        self.converged_ = False
        self.n_iter_ = self.max_iter
        self.coef_[0] = float(w)
        self.intercept_[0] = float(b)
        self._fitted = True
        return self

    def decision_function(self, X):
        if not self._fitted:
            msg = "Reference1DProbe instance is not fitted yet."
            raise RuntimeError(msg)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            x = X.reshape(-1)
        else:
            if X.ndim != 2 or X.shape[1] != 1:
                msg = f"Reference1DProbe expects exactly one feature; received array with shape {X.shape}."
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
    """Sparse trust-region optimizer for per-latent logistic probes.

    Each `(latent, class)` pair is solved with a Levenberg--Marquardt step that
    mirrors :class:`Reference1DProbe`, while sparse CSR activations are streamed
    so loss and gradients are accumulated without densifying the feature matrix.
    """

    def __init__(
        self,
        *,
        n_latents: int,
        n_classes: int,
        ridge: float = 1e-8,
        tol: float = 1e-6,
        max_iter: int = 200,
        lam_init: float = 1e-3,
        lam_shrink: float = 0.1,
        lam_grow: float = 10.0,
        delta_logit: float = 6.0,
        device: str = "cuda",
        dtype=torch.float32,
        class_slab_size: int = 8,
        row_batch_size: int = 1024,
    ) -> None:
        if lam_shrink <= 0 or lam_shrink >= 1:
            msg = f"lam_shrink must lie in (0,1), got {lam_shrink}."
            raise ValueError(msg)
        if lam_grow <= 1:
            msg = f"lam_grow must be >1, got {lam_grow}."
            raise ValueError(msg)
        if delta_logit <= 0:
            msg = f"delta_logit must be >0, got {delta_logit}."
            raise ValueError(msg)

        self.n_latents = n_latents
        self.n_classes = n_classes
        self.tol = tol
        self.ridge = float(ridge)
        self.class_slab_size = class_slab_size
        self.row_batch_size = row_batch_size
        self.lam_init = lam_init
        self.lam_shrink = lam_shrink
        self.lam_grow = lam_grow
        self.delta_logit = delta_logit
        self.lam_min = 1e-12
        self.lam_max = 1e12
        self.eps = 1e-8
        self.fallback_step_scale = 1e-3

        self.logger = logging.getLogger("sparse1d")
        self.device = torch.device(device)
        self.dtype = dtype

        self.max_iter = max_iter

        self.intercept_: Float[Tensor, "n_latents n_classes"] = torch.zeros(
            (self.n_latents, self.n_classes), **self._dd
        )
        self.coef_: Float[Tensor, "n_latents n_classes"] = torch.zeros(
            (self.n_latents, self.n_classes), **self._dd
        )
        self._qx_per_latent: Float[Tensor, "n_latents"] = torch.zeros(
            (self.n_latents,), **self._dd
        )
        self._debug_last: dict[str, Tensor] = {}
        self.n_iter_: Tensor = torch.zeros(
            (self.n_classes,), dtype=torch.int32, device=self.device
        )

    @torch.no_grad()
    def fit(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Float[Tensor, "n_samples n_classes"],
    ) -> tp.Self:
        if not x.is_sparse_csr:
            raise TypeError("x must be a torch.sparse_csr_tensor.")

        n_samples, n_latents = x.shape
        if n_latents != self.n_latents:
            raise ValueError(f"x has {n_latents} latents, expected {self.n_latents}.")
        if y.shape != (n_samples, self.n_classes):
            raise ValueError(
                f"y has shape {tuple(y.shape)}, expected ({n_samples}, {self.n_classes})."
            )

        values = x.values()
        nnz = int(values.numel())
        crow_indices = x.crow_indices()
        col_indices = x.col_indices()
        values_dtype = values.dtype
        crow_dtype = crow_indices.dtype
        col_dtype = col_indices.dtype
        values_bytes = nnz * values.element_size()
        crow_bytes = crow_indices.numel() * crow_indices.element_size()
        col_bytes = col_indices.numel() * col_indices.element_size()
        total_bytes = values_bytes + crow_bytes + col_bytes
        target_value_bytes = nnz * torch.tensor([], dtype=torch.float32).element_size()
        total_bytes_target = target_value_bytes + crow_bytes + col_bytes
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                "Sparse design stats before device transfer: shape=%s, nnz=%d, values_dtype=%s, crow_dtype=%s, col_dtype=%s, approx_bytes=%.2f GiB, approx_bytes_target=%.2f GiB",
                tuple(x.shape),
                nnz,
                values_dtype,
                crow_dtype,
                col_dtype,
                total_bytes / (1024**3),
                total_bytes_target / (1024**3),
            )

        x = x.to(**self._dd)
        y = y.to(**self._dd)
        self.n_iter_.zero_()

        n_samples_f = float(n_samples)
        pi = torch.clamp(y.mean(dim=0), self.eps, 1 - self.eps)
        base_intercept = torch.log(pi / (1 - pi))

        intercept = (
            base_intercept.to(**self._dd)
            .view(1, -1)
            .expand(self.n_latents, -1)
            .contiguous()
            .clone()
        )
        coef = torch.zeros((self.n_latents, self.n_classes), **self._dd)

        col_indices = x.col_indices()
        values = x.values().to(self.dtype)

        nnz_counts = torch.zeros(self.n_latents, dtype=torch.int64, device=self.device)
        sum_sq = torch.zeros(self.n_latents, **self._dd)
        nnz = int(values.numel())
        if nnz > 0:
            chunk_size = max(1, min(nnz, 1_000_000))
            ones_buffer = torch.ones(chunk_size, dtype=torch.int64, device=self.device)
            for start in range(0, nnz, chunk_size):
                end = min(start + chunk_size, nnz)
                cols_chunk = col_indices[start:end]
                nnz_counts.index_add_(0, cols_chunk, ones_buffer[: end - start])
                sum_sq.index_add_(0, cols_chunk, torch.square(values[start:end]))

        empty_latent_mask = (nnz_counts == 0).view(-1, 1)

        nnz_per_latent = nnz_counts.to(self.dtype)
        n_zeros_per_latent = (n_samples_f - nnz_per_latent).clamp(min=0.0).view(-1, 1)

        rms = torch.sqrt(
            torch.where(
                nnz_per_latent > 0,
                sum_sq / torch.clamp(nnz_per_latent, min=1.0),
                torch.ones_like(sum_sq),
            )
        )
        qx = torch.where(nnz_per_latent > 0, rms, torch.ones_like(rms))
        qx = torch.clamp(qx, min=1e-6)
        qx = qx.view(-1, 1)
        qx_sq = qx * qx
        self.latent_qx_ = qx.view(-1).clone()

        chunk_target = max(1, self.row_batch_size * 32)
        crow_indices_cpu = crow_indices.to(torch.long).cpu()
        row_chunks = self._plan_row_chunks(crow_indices_cpu, chunk_target)

        base_intercept_matrix = base_intercept.view(1, -1)

        for c0, c1 in saev.helpers.batched_idx(self.n_classes, self.class_slab_size):
            y_slab = y[:, c0:c1]
            intercept_slab = intercept[:, c0:c1]
            coef_slab = coef[:, c0:c1]

            lam = torch.full((self.n_latents, c1 - c0), self.lam_init, **self._dd)
            prev_pred = torch.full_like(lam, float("nan"))
            prev_loss = torch.full_like(lam, float("nan"))
            prev_step_clipped = torch.zeros_like(lam, dtype=torch.bool)

            pi_mean = y_slab.mean(dim=0).to(self.dtype)
            base_slab = base_intercept_matrix[:, c0:c1]
            empty_mask_slab = empty_latent_mask[:, c0:c1]
            base_broadcast = base_slab.expand_as(intercept_slab)
            lam_reset = None
            nan_reset = None
            zero_loss_reset = None
            zero_step_reset = None
            if empty_mask_slab.any():
                lam_reset = torch.full_like(lam, self.lam_init)
                nan_reset = torch.full_like(prev_pred, float("nan"))
                zero_loss_reset = torch.full_like(prev_loss, float("nan"))
                zero_step_reset = torch.zeros_like(prev_step_clipped)
                intercept_slab = torch.where(
                    empty_mask_slab, base_broadcast, intercept_slab
                )
                coef_slab = torch.where(
                    empty_mask_slab, torch.zeros_like(coef_slab), coef_slab
                )
                lam = torch.where(empty_mask_slab, lam_reset, lam)
                prev_pred = torch.where(empty_mask_slab, nan_reset, prev_pred)
                prev_loss = torch.where(empty_mask_slab, zero_loss_reset, prev_loss)
                prev_step_clipped = torch.where(
                    empty_mask_slab, zero_step_reset, prev_step_clipped
                )

            for iter_idx in range(self.max_iter):
                iter_count = iter_idx + 1
                track_vram = self.device.type == "cuda" and self.logger.isEnabledFor(
                    logging.DEBUG
                )
                if track_vram:
                    torch.cuda.reset_peak_memory_stats(self.device)

                stats = self._compute_slab_stats(
                    x, y_slab, intercept_slab, coef_slab, row_chunks
                )

                mu_0 = torch.sigmoid(intercept_slab).clamp_(self.eps, 1 - self.eps)
                s_0 = mu_0 * (1 - mu_0)

                mu_nz_mean = stats.mu_nz / n_samples_f
                pos_nz_mean = stats.pos_nz / n_samples_f
                zeros_frac = n_zeros_per_latent / n_samples_f
                g0 = mu_nz_mean + zeros_frac * mu_0 - pi_mean
                g0 = g0 + self.ridge * (intercept_slab - base_slab)
                g1 = stats.g1 / n_samples_f + self.ridge * coef_slab

                h0 = stats.h0 / n_samples_f + zeros_frac * s_0 + self.ridge
                h1 = stats.h1 / n_samples_f
                h2 = stats.h2 / n_samples_f + self.ridge

                qx_sq_step = qx_sq.expand(-1, c1 - c0)

                pos_zero_mean = torch.clamp(pi_mean - pos_nz_mean, min=0.0)
                pos_zero_mean = torch.minimum(pos_zero_mean, zeros_frac)
                neg_zero_mean = zeros_frac - pos_zero_mean
                zero_loss = -(
                    pos_zero_mean * torch.log(mu_0)
                    + neg_zero_mean * torch.log1p(-mu_0.clamp(max=1 - self.eps))
                )
                ridge_penalty = (
                    0.5
                    * self.ridge
                    * (coef_slab**2 + (intercept_slab - base_slab) ** 2)
                )
                loss_curr = stats.loss_nz / n_samples_f + zero_loss + ridge_penalty

                if empty_mask_slab.any():
                    g0 = torch.where(empty_mask_slab, torch.zeros_like(g0), g0)
                    g1 = torch.where(empty_mask_slab, torch.zeros_like(g1), g1)
                    lam = torch.where(empty_mask_slab, lam_reset, lam)

                mask_prev = torch.isfinite(prev_pred) & torch.isfinite(prev_loss)
                rho = torch.zeros_like(loss_curr)
                if mask_prev.any():
                    actual = prev_loss[mask_prev] - loss_curr[mask_prev]
                    rho[mask_prev] = actual / torch.clamp(
                        prev_pred[mask_prev], min=1e-18
                    )
                    grow_mask = mask_prev & ((rho <= 0.25) | prev_step_clipped)
                    shrink_mask = mask_prev & (rho >= 0.75) & (~prev_step_clipped)
                    lam = torch.where(shrink_mask, lam * self.lam_shrink, lam)
                    lam = torch.where(grow_mask, lam * self.lam_grow, lam)
                    lam = lam.clamp(self.lam_min, self.lam_max)

                db, dw, pred, lam_next, step_clipped, success = self.compute_lm_step(
                    g0=g0, g1=g1, h0=h0, h1=h1, h2=h2, lam=lam, qx_sq=qx_sq_step
                )

                intercept_slab = intercept_slab - db
                coef_slab = coef_slab - dw

                lam = lam_next
                prev_pred = pred
                prev_loss = loss_curr
                prev_step_clipped = step_clipped

                if empty_mask_slab.any():
                    intercept_slab = torch.where(
                        empty_mask_slab, base_broadcast, intercept_slab
                    )
                    coef_slab = torch.where(
                        empty_mask_slab, torch.zeros_like(coef_slab), coef_slab
                    )
                    lam = torch.where(empty_mask_slab, lam_reset, lam)
                    db = torch.where(empty_mask_slab, torch.zeros_like(db), db)
                    dw = torch.where(empty_mask_slab, torch.zeros_like(dw), dw)
                    pred = torch.where(empty_mask_slab, torch.zeros_like(pred), pred)
                    prev_pred = torch.where(empty_mask_slab, nan_reset, prev_pred)
                    prev_step_clipped = torch.where(
                        empty_mask_slab, zero_step_reset, prev_step_clipped
                    )
                    step_clipped = torch.where(
                        empty_mask_slab, torch.zeros_like(step_clipped), step_clipped
                    )
                    success = torch.where(
                        empty_mask_slab, torch.zeros_like(success), success
                    )

                qx = torch.sqrt(qx_sq_step)
                qx_safe = torch.clamp(qx, min=1e-12)
                grad_abs = torch.maximum(g0.abs(), (g1 / qx_safe).abs())
                grad_norm = grad_abs.max().item()
                scaled_step = torch.maximum(db.abs(), (qx * dw).abs())
                step_norm = scaled_step.max().item()
                lambda_mean = float(lam.mean().item())

                debug_enabled = self.logger.isEnabledFor(logging.DEBUG)
                if debug_enabled:
                    self._debug_last = {
                        "g0": g0,
                        "g1": g1,
                        "h0": h0,
                        "h1": h1,
                        "h2": h2,
                        "mu_nz_mean": mu_nz_mean,
                        "pos_nz_mean": pos_nz_mean,
                        "zeros_frac": zeros_frac,
                        "loss_curr": loss_curr,
                        "pred": pred,
                        "db": db,
                        "dw": dw,
                        "lam": lam,
                    }
                    loss_mean = loss_curr.mean().item()
                    loss_max = loss_curr.max().item()
                    rho_vals = rho[mask_prev]
                    rho_mean = (
                        rho_vals.mean().item() if rho_vals.numel() > 0 else math.nan
                    )
                    rho_min = (
                        rho_vals.min().item() if rho_vals.numel() > 0 else math.nan
                    )
                    total_coords = success.numel()
                    success_count = success.sum().item()
                    fallback_count = total_coords - success_count
                    success_frac = (
                        success_count / total_coords if total_coords > 0 else math.nan
                    )
                    step_clipped_count = step_clipped.sum().item()
                    pred_success = pred[success]
                    pred_success_mean = (
                        pred_success.mean().item()
                        if pred_success.numel() > 0
                        else math.nan
                    )

                vram_alloc_mb: float | None = None
                vram_reserved_mb: float | None = None
                if track_vram:
                    torch.cuda.synchronize(self.device)
                    alloc = torch.cuda.max_memory_allocated(self.device)
                    reserved = torch.cuda.max_memory_reserved(self.device)
                    vram_alloc_mb = float(alloc) / (1024 * 1024)
                    vram_reserved_mb = float(reserved) / (1024 * 1024)
                    self.logger.debug(
                        "slab=%s iter=%d grad_max=%.3e step_max=%.3e lambda_mean=%.3e loss_mean=%.3e loss_max=%.3e rho_mean=%.3e rho_min=%.3e success_frac=%.3f fallback=%d step_clipped=%d pred_mean=%.3e peak_alloc_mb=%.2f peak_reserved_mb=%.2f",
                        (c0, c1),
                        iter_idx,
                        grad_norm,
                        step_norm,
                        lambda_mean,
                        loss_mean,
                        loss_max,
                        rho_mean,
                        rho_min,
                        success_frac,
                        fallback_count,
                        step_clipped_count,
                        pred_success_mean,
                        vram_alloc_mb,
                        vram_reserved_mb,
                    )
                elif debug_enabled:
                    self.logger.debug(
                        "slab=%s iter=%d grad_max=%.3e step_max=%.3e lambda_mean=%.3e loss_mean=%.3e loss_max=%.3e rho_mean=%.3e rho_min=%.3e success_frac=%.3f fallback=%d step_clipped=%d pred_mean=%.3e",
                        (c0, c1),
                        iter_idx,
                        grad_norm,
                        step_norm,
                        lambda_mean,
                        loss_mean,
                        loss_max,
                        rho_mean,
                        rho_min,
                        success_frac,
                        fallback_count,
                        step_clipped_count,
                        pred_success_mean,
                    )

                if torch.all(grad_abs <= self.tol) or (
                    grad_norm < self.tol and step_norm < self.tol
                ):
                    break
            else:
                iter_count = self.max_iter

            intercept[:, c0:c1] = intercept_slab
            coef[:, c0:c1] = coef_slab
            self.n_iter_.narrow(0, c0, c1 - c0).fill_(iter_count)

        self.intercept_ = intercept.to(self.dtype)
        self.coef_ = coef.to(self.dtype)
        return self

    def _compute_slab_stats(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y_slab: Float[Tensor, "n_samples c_b"],
        intercept_slab: Float[Tensor, "n_latents c_b"],
        coef_slab: Float[Tensor, "n_latents c_b"],
        row_chunks: list[RowChunk],
    ) -> SlabStats:
        mu_nz = torch.zeros_like(intercept_slab)
        g1 = torch.zeros_like(intercept_slab)
        h0 = torch.zeros_like(intercept_slab)
        h1 = torch.zeros_like(intercept_slab)
        h2 = torch.zeros_like(intercept_slab)
        loss_nz = torch.zeros_like(intercept_slab)
        pos_nz = torch.zeros_like(intercept_slab)

        values_all = x.values().to(self.dtype)
        cols_all = x.col_indices()

        for events in self._iter_event_chunks(values_all, cols_all, row_chunks):
            vals = events.values.view(-1, 1)
            idx = events.latent_idx
            rows = events.row_idx

            b_cols = intercept_slab[idx]
            w_cols = coef_slab[idx]
            logits = b_cols + w_cols * vals
            mu = torch.sigmoid(logits)
            s = mu * (1 - mu)

            y_chunk = y_slab[rows].to(self.dtype)
            residual = mu - y_chunk

            mu_nz.index_add_(0, idx, mu)
            g1.index_add_(0, idx, residual * vals)
            h0.index_add_(0, idx, s)
            h1.index_add_(0, idx, s * vals)
            h2.index_add_(0, idx, s * (vals**2))
            loss_chunk = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_chunk, reduction="none"
            )
            loss_nz.index_add_(0, idx, loss_chunk)
            pos_nz.index_add_(0, idx, y_chunk)

        return SlabStats(mu_nz, g1, h0, h1, h2, loss_nz, pos_nz)

    def compute_lm_step(
        self,
        *,
        g0: Tensor,
        g1: Tensor,
        h0: Tensor,
        h1: Tensor,
        h2: Tensor,
        lam: Tensor,
        qx_sq: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        lam_curr = lam.clone()
        grad_norm_init = torch.maximum(g0.abs(), g1.abs())
        inactive = grad_norm_init <= self.tol
        success = inactive.clone()
        db = torch.zeros_like(g0)
        dw = torch.zeros_like(g0)
        pred = torch.zeros_like(g0)
        step_clipped = torch.zeros_like(g0, dtype=torch.bool)

        for _ in range(5):
            active = ~success
            if not active.any():
                break

            h0_eff = h0 + lam_curr
            h2_eff = h2 + lam_curr * qx_sq
            det = h0_eff * h2_eff - h1 * h1

            valid = active & (det.abs() > 1e-18)
            det_safe = torch.where(valid, det, torch.ones_like(det))

            db_temp = (h2_eff * g0 - h1 * g1) / det_safe
            dw_temp = (h0_eff * g1 - h1 * g0) / det_safe
            db_temp = torch.where(valid, db_temp, torch.zeros_like(db_temp))
            dw_temp = torch.where(valid, dw_temp, torch.zeros_like(dw_temp))

            qx = torch.sqrt(qx_sq)
            norm = torch.sqrt(db_temp**2 + (qx * dw_temp) ** 2)
            clipped = active & (norm > self.delta_logit)
            scale = torch.ones_like(norm)
            scale = torch.where(clipped, self.delta_logit / (norm + 1e-18), scale)
            db_temp = db_temp * scale
            dw_temp = dw_temp * scale

            pred_temp = (
                g0 * db_temp
                + g1 * dw_temp
                - 0.5 * (h0 * db_temp**2 + 2 * h1 * db_temp * dw_temp + h2 * dw_temp**2)
            )
            valid_pred = active & torch.isfinite(pred_temp) & (pred_temp > 0)
            success_now = valid_pred
            if success_now.any():
                db = torch.where(success_now, db_temp, db)
                dw = torch.where(success_now, dw_temp, dw)
                pred = torch.where(success_now, pred_temp, pred)
                step_clipped = torch.where(success_now, clipped, step_clipped)
                success = success | success_now

            failure = active & (~valid_pred)
            if failure.any():
                lam_curr[failure] = torch.clamp(
                    lam_curr[failure] * self.lam_grow,
                    min=self.lam_min,
                    max=self.lam_max,
                )

        if (~success).any():
            failed = ~success
            n_fail = int(failed.sum().item())
            lam_failed = lam_curr[failed]
            g0_failed = g0[failed].abs().max().item() if n_fail > 0 else 0.0
            g1_failed = g1[failed].abs().max().item() if n_fail > 0 else 0.0
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(
                    "LM step fallback: n_fail=%d lam_min=%.3e lam_max=%.3e "
                    "grad_max=(%.3e, %.3e)",
                    n_fail,
                    float(lam_failed.min().item()) if n_fail > 0 else 0.0,
                    float(lam_failed.max().item()) if n_fail > 0 else 0.0,
                    g0_failed,
                    g1_failed,
                )
            qx = torch.sqrt(qx_sq)
            qx_safe = torch.clamp(qx, min=1e-12)
            grad_scaled = torch.sqrt(g0**2 + (qx_safe * g1) ** 2)
            target_norm = self.fallback_step_scale * self.delta_logit
            alpha = torch.zeros_like(grad_scaled)
            positive_grad = grad_scaled > 0
            alpha = torch.where(
                positive_grad,
                target_norm / (grad_scaled + 1e-18),
                alpha,
            )
            db_fallback = -alpha * g0
            dw_fallback = -alpha * g1
            db = torch.where(failed, db_fallback, db)
            dw = torch.where(failed, dw_fallback, dw)
            pred_nan = torch.full_like(pred, float("nan"))
            pred = torch.where(failed, pred_nan, pred)
            step_clipped = step_clipped | failed
            success = success | failed

        lam_curr = lam_curr.clamp(self.lam_min, self.lam_max)
        return db, dw, pred, lam_curr, step_clipped, success

    def _plan_row_chunks(
        self, crow_indices: Int[Tensor, "..."], chunk_target: int
    ) -> list[RowChunk]:
        """Plan contiguous CSR row chunks for streaming sparse batches.

        We partition the CSR matrix into micro-batches that each contain roughly `chunk_target` events so downstream iterators can consume non-zeros deterministically without materializing row indices on the device or loading the full CSR structure at once. We also precompute CPU row indices (pinning them when using CUDA) so later transfers into device memory overlap with compute.

        Args:
            crow_indices: CSR row pointer tensor on CPU with length n_rows + 1; monotonic and zero-based.
            chunk_target: Preferred number of non-zero events per chunk before fallback adjustments.

        Returns:
            List of `RowChunk` items describing contiguous row bounds, CSR pointer offsets, and cached CPU row indices. Returns an empty list when there are no rows, and a single empty chunk when there are zero non-zero events.
        """
        n_rows = crow_indices.numel() - 1
        if n_rows <= 0:
            return []

        nnz_per_row_cpu = crow_indices[1:] - crow_indices[:-1]
        total_nnz = crow_indices[-1].item()
        if total_nnz == 0:
            empty_idx = torch.empty(0, dtype=torch.int32)
            if self.device.type == "cuda":
                empty_idx = empty_idx.pin_memory()
            return [(0, n_rows, 0, 0, empty_idx)]

        chunk_target = max(1, chunk_target)
        if total_nnz <= chunk_target:
            boundaries = torch.empty(0, dtype=crow_indices.dtype)
        else:
            boundaries_targets = torch.arange(
                chunk_target,
                total_nnz,
                chunk_target,
                dtype=crow_indices.dtype,
                device=crow_indices.device,
            )
            if boundaries_targets.numel() == 0:
                boundaries = torch.empty(0, dtype=crow_indices.dtype)
            else:
                cumulative = torch.cumsum(nnz_per_row_cpu, 0)
                boundaries = torch.searchsorted(cumulative, boundaries_targets)

        row_starts = torch.cat([
            torch.zeros(1, dtype=crow_indices.dtype),
            boundaries,
        ])
        row_ends = torch.cat([
            boundaries,
            torch.tensor([n_rows], dtype=crow_indices.dtype),
        ])

        chunks: list[RowChunk] = []
        for start, end in zip(row_starts.tolist(), row_ends.tolist()):
            if start >= end:
                continue
            start_ptr = int(crow_indices[start].item())
            end_ptr = int(crow_indices[end].item())
            if start_ptr == end_ptr:
                continue
            lengths_cpu = nnz_per_row_cpu[start:end]
            if lengths_cpu.numel() == 0:
                continue
            n_events = int(lengths_cpu.sum().item())
            if n_events == 0:
                continue
            rows_cpu = torch.arange(start, end, dtype=torch.int32)
            row_idx_cpu = torch.repeat_interleave(rows_cpu, lengths_cpu.to(torch.long))
            if self.device.type == "cuda":
                row_idx_cpu = row_idx_cpu.pin_memory()
            chunks.append(RowChunk(start, end, start_ptr, end_ptr, row_idx_cpu))

        if not chunks:
            rows_cpu = torch.arange(0, n_rows, dtype=torch.int32)
            lengths_cpu = nnz_per_row_cpu
            row_idx_cpu = torch.repeat_interleave(rows_cpu, lengths_cpu.to(torch.long))
            if self.device.type == "cuda":
                row_idx_cpu = row_idx_cpu.pin_memory()
            chunks.append(RowChunk(0, n_rows, 0, total_nnz, row_idx_cpu))

        return chunks

    def _iter_event_chunks(
        self,
        values: Tensor,
        col_indices: Tensor,
        row_chunks: list["RowChunk"],
    ) -> Iterator[SparseEventsBatch]:
        for row_chunk in row_chunks:
            start_ptr = row_chunk.start_ptr
            end_ptr = row_chunk.end_ptr
            if start_ptr == end_ptr:
                continue
            row_idx_cpu = row_chunk.row_idx_cpu
            if row_idx_cpu.numel() == 0:
                continue
            row_idx = row_idx_cpu
            if self.device.type == "cuda":
                row_idx = row_idx_cpu.to(
                    device=self.device, dtype=torch.long, non_blocking=True
                )
            else:
                row_idx = row_idx_cpu.to(device=self.device, dtype=torch.long)

            yield SparseEventsBatch(
                row_start=row_chunk.row_start,
                row_end=row_chunk.row_end,
                latent_idx=col_indices[start_ptr:end_ptr],
                values=values[start_ptr:end_ptr],
                row_idx=row_idx,
            )

    def _compute_loss_slab(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y_slab: Tensor,
        c0: int,
        c1: int,
        n_samples: int,
        pi_slab: Tensor,
        n_zeros_per_latent: Tensor,
        row_chunks: list[RowChunk],
    ) -> Tensor:
        loss = torch.zeros((self.n_latents, c1 - c0), **self._dd)
        pos_nz = torch.zeros_like(loss)

        intercept_slab = self.intercept_[:, c0:c1].to(self.dtype)
        coef_slab = self.coef_[:, c0:c1].to(self.dtype)

        values_all = x.values().to(self.dtype)
        cols_all = x.col_indices()
        for events in self._iter_event_chunks(values_all, cols_all, row_chunks):
            cols_chunk = events.latent_idx
            row_idx_chunk = events.row_idx
            y_nz = y_slab[row_idx_chunk].to(self.dtype)

            vals_expanded = events.values.view(-1, 1)
            b_cols = intercept_slab[cols_chunk]
            w_cols = coef_slab[cols_chunk]
            eta = b_cols + w_cols * vals_expanded

            nz_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                eta, y_nz, reduction="none"
            )
            loss.index_add_(0, cols_chunk, nz_loss)
            pos_nz.index_add_(0, cols_chunk, y_nz)

        pos_zero = torch.clamp(pi_slab.view(1, c1 - c0) - pos_nz, min=0.0)
        pos_zero = torch.minimum(pos_zero, n_zeros_per_latent)
        neg_zero = n_zeros_per_latent - pos_zero

        zero_loss_pos = pos_zero * torch.nn.functional.softplus(-intercept_slab)
        zero_loss_neg = neg_zero * torch.nn.functional.softplus(intercept_slab)
        loss = loss + zero_loss_pos + zero_loss_neg

        return (loss / float(n_samples)).to(self.dtype)

    def _compute_loss(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Float[Tensor, "n_samples n_classes"],
    ) -> Float[Tensor, "n_latents n_classes"]:
        n_samples = x.shape[0]
        loss = torch.zeros((self.n_latents, self.n_classes), **self._dd)

        col_indices = x.col_indices()
        nnz_per_latent = torch.bincount(col_indices, minlength=self.n_latents)
        n_zeros_per_latent = (n_samples - nnz_per_latent).to(**self._dd).view(-1, 1)

        pi = y.sum(dim=0).to(self.dtype)
        crow_indices_cpu = x.crow_indices().to(torch.long).cpu()
        chunk_target = max(1, self.row_batch_size * 32)
        row_chunks = self._plan_row_chunks(crow_indices_cpu, chunk_target)

        for c0 in range(0, self.n_classes, self.class_slab_size):
            c1 = min(c0 + self.class_slab_size, self.n_classes)
            y_slab = y[:, c0:c1].to(**self._dd)
            pi_slab = pi[c0:c1].to(self.device)
            loss[:, c0:c1] = self._compute_loss_slab(
                x,
                y_slab,
                c0,
                c1,
                n_samples,
                pi_slab,
                n_zeros_per_latent,
                row_chunks,
            )
            del y_slab, pi_slab

        return loss.to(self.dtype)

    def loss_matrix(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Bool[Tensor, "n_samples n_classes"],
    ) -> Float[Tensor, "n_latents n_classes"]:
        sklearn.utils.validation.check_is_fitted(self, "intercept_")
        sklearn.utils.validation.check_is_fitted(self, "coef_")
        x = x.to(**self._dd)
        y = y.to(**self._dd)
        return self._compute_loss(x, y).to(torch.float32)

    def _compute_confusion_slab(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y_slab: Tensor,
        c0: int,
        c1: int,
        pi_slab: Tensor,
        n_zeros_per_latent: Tensor,
        threshold: float,
        row_chunks: list[RowChunk],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tp = torch.zeros((self.n_latents, c1 - c0), **self._dd)
        fp = torch.zeros_like(tp)
        tn = torch.zeros_like(tp)
        fn = torch.zeros_like(tp)
        pos_nz = torch.zeros_like(tp)

        intercept_slab = self.intercept_[:, c0:c1].to(self.dtype)
        coef_slab = self.coef_[:, c0:c1].to(self.dtype)

        mu_0 = torch.sigmoid(intercept_slab)
        pred_zero = mu_0 > threshold

        values_all = x.values().to(self.dtype)
        cols_all = x.col_indices()
        for events in self._iter_event_chunks(values_all, cols_all, row_chunks):
            cols_chunk = events.latent_idx
            row_idx_chunk = events.row_idx
            y_nz = y_slab[row_idx_chunk].to(self.dtype)
            y_nz_bool = y_nz > 0.5

            vals_expanded = events.values.view(-1, 1)
            b_cols = intercept_slab[cols_chunk]
            w_cols = coef_slab[cols_chunk]
            mu = torch.sigmoid(b_cols + w_cols * vals_expanded)
            pred_nz_bool = mu > threshold

            tp_chunk = torch.logical_and(pred_nz_bool, y_nz_bool).to(self.dtype)
            fp_chunk = torch.logical_and(pred_nz_bool, ~y_nz_bool).to(self.dtype)
            fn_chunk = torch.logical_and(~pred_nz_bool, y_nz_bool).to(self.dtype)
            tn_chunk = torch.logical_and(~pred_nz_bool, ~y_nz_bool).to(self.dtype)

            tp.index_add_(0, cols_chunk, tp_chunk)
            fp.index_add_(0, cols_chunk, fp_chunk)
            fn.index_add_(0, cols_chunk, fn_chunk)
            tn.index_add_(0, cols_chunk, tn_chunk)
            pos_nz.index_add_(0, cols_chunk, y_nz)

        pos_zero = torch.clamp(pi_slab.view(1, c1 - c0) - pos_nz, min=0.0)
        pos_zero = torch.minimum(pos_zero, n_zeros_per_latent)
        neg_zero = n_zeros_per_latent - pos_zero

        zero_mask = pred_zero.to(self.dtype)
        tp_zero = zero_mask * pos_zero
        fp_zero = zero_mask * neg_zero
        fn_zero = (1.0 - zero_mask) * pos_zero
        tn_zero = (1.0 - zero_mask) * neg_zero

        tp = tp + tp_zero
        fp = fp + fp_zero
        fn = fn + fn_zero
        tn = tn + tn_zero

        return (
            tp.to(self.dtype),
            fp.to(self.dtype),
            tn.to(self.dtype),
            fn.to(self.dtype),
        )

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
        sklearn.utils.validation.check_is_fitted(self, "intercept_")
        sklearn.utils.validation.check_is_fitted(self, "coef_")

        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be between 0 and 1.")

        x = x.to(**self._dd)
        n_samples = x.shape[0]

        loss = self._compute_loss(x, y.to(self.dtype))

        col_indices = x.col_indices()
        nnz_per_latent = torch.bincount(col_indices, minlength=self.n_latents)
        n_zeros_per_latent = (n_samples - nnz_per_latent).to(**self._dd).view(-1, 1)

        pi = y.sum(dim=0).to(self.dtype)
        crow_indices_cpu = x.crow_indices().to(torch.long).cpu()
        chunk_target = max(1, self.row_batch_size * 32)
        row_chunks = self._plan_row_chunks(crow_indices_cpu, chunk_target)

        tp = torch.zeros((self.n_latents, self.n_classes), **self._dd)
        fp = torch.zeros_like(tp)
        tn = torch.zeros_like(tp)
        fn = torch.zeros_like(tp)

        for c0 in range(0, self.n_classes, self.class_slab_size):
            c1 = min(c0 + self.class_slab_size, self.n_classes)

            y_slab = y[:, c0:c1].to(**self._dd)
            pi_slab = pi[c0:c1].to(self.device)

            tp_chunk, fp_chunk, tn_chunk, fn_chunk = self._compute_confusion_slab(
                x,
                y_slab,
                c0,
                c1,
                pi_slab,
                n_zeros_per_latent,
                threshold,
                row_chunks,
            )

            tp[:, c0:c1] = tp_chunk
            fp[:, c0:c1] = fp_chunk
            tn[:, c0:c1] = tn_chunk
            fn[:, c0:c1] = fn_chunk

            del y_slab, pi_slab

        return (
            loss.to(torch.float32),
            tp.to(torch.float32),
            fp.to(torch.float32),
            tn.to(torch.float32),
            fn.to(torch.float32),
        )

    @property
    def _dd(self) -> dict[str, tp.Any]:
        return dict(device=self.device, dtype=self.dtype)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    run: pathlib.Path = pathlib.Path("./runs/abcdefg")
    """Run directory."""
    shards_dir: pathlib.Path = pathlib.Path("./shards/e967c008")
    """Shards directory."""
    # Optimization
    ridge: float = 1e-8
    """Ridge value."""
    class_slab_size: int = 8
    """Number of classes to optimize in parallel."""
    row_batch_size: int = 1024
    """Number of rows to query per batch."""
    max_iter: int = 100
    """Number of iterations in the solver."""
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
        n_latents=n_latents,
        n_classes=n_classes,
        device=cfg.device,
        ridge=cfg.ridge,
        max_iter=cfg.max_iter,
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
