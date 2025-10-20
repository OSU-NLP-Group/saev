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
        use_elliptical: bool = True,
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
                    "Reference1DProbe expects exactly one feature; "
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
        prev_step_clipped = False

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

            g0 = r.sum() + self.ridge * (b - b0)
            g1 = (r * x).sum() + self.ridge * w
            h0 = s.sum() + self.ridge
            h1 = (s * x).sum()
            h2 = (s * x * x).sum() + self.ridge

            grad_max = max(abs(g0), abs(g1))

            # LM step with retries
            tried = 0
            db = dw = 0.0
            step_clipped = False
            while tried < 6:
                step_clipped_local = False
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
                        step_clipped_local = True
                else:
                    delta_b = self.delta_logit
                    delta_w = self.delta_logit / qx_value
                    ratio_b = abs(db) / delta_b
                    ratio_w = abs(dw) / delta_w
                    ratio = max(ratio_b, ratio_w)
                    if ratio > 1.0:
                        scale = 1.0 / ratio
                        db *= scale
                        dw *= scale
                        step_clipped_local = True

                if not np.isfinite(db) or not np.isfinite(dw):
                    lam = min(lam * self.lam_grow, self.lam_max)
                    tried += 1
                    continue

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
                step_clipped = step_clipped_local
                break

            # apply step
            b_new = b - db
            w_new = w - dw
            loss_new = loss(b_new, w_new)

            prev_pred = pred
            prev_loss_before_step = loss_curr
            loss_curr = loss_new
            b, w = b_new, w_new
            prev_step_clipped = step_clipped

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
            msg = "Reference1DProbe instance is not fitted yet."
            raise RuntimeError(msg)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            x = X.reshape(-1)
        else:
            if X.ndim != 2 or X.shape[1] != 1:
                msg = (
                    "Reference1DProbe expects exactly one feature; "
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
        tol: float = 1e-4,
        device: str = "cuda",
        max_iter: int = 100,
        ridge: float = 1e-8,
        class_slab_size: int = 8,
        row_batch_size: int = 1024,
        hessian_floor: float = 1e-4,
        lm_lambda_init: float = 1e-3,
        lm_lambda_shrink: float = 0.1,
        lm_lambda_grow: float = 10.0,
        lm_lambda_min: float = 1e-12,
        lm_lambda_max: float = 1e12,
        lm_max_update: float = 8.0,
        lm_max_adapt_iters: int = 6,
        lm_qx_min_scale: float = 5e-2,
        iteration_hook: tp.Callable[[dict[str, tp.Any]], None] | None = None,
    ) -> None:
        if n_latents <= 0 or n_classes <= 0:
            raise ValueError("n_latents and n_classes must be positive.")
        if not 0.0 < lm_lambda_shrink < 1.0:
            raise ValueError("lm_lambda_shrink must lie in (0,1).")
        if lm_lambda_grow <= 1.0:
            raise ValueError("lm_lambda_grow must be larger than 1.")
        if lm_max_adapt_iters <= 0:
            raise ValueError("lm_max_adapt_iters must be positive.")
        if not 0.0 < lm_qx_min_scale <= 1.0:
            raise ValueError("lm_qx_min_scale must lie in (0, 1].")

        self.n_latents = int(n_latents)
        self.n_classes = int(n_classes)
        self.tol = float(tol)
        self.device = torch.device(device)
        self.ridge = float(ridge)
        self.class_slab_size = int(class_slab_size)
        self.row_batch_size = int(row_batch_size)
        self.lam_init = float(lm_lambda_init)
        self.lam_lambda_shrink = float(lm_lambda_shrink)
        self.lam_lambda_grow = float(lm_lambda_grow)
        self.lam_min = float(lm_lambda_min)
        self.lam_max = float(lm_lambda_max)
        self.lm_max_adapt_iters = int(lm_max_adapt_iters)
        self.delta_logit = float(lm_max_update)
        self.logger = logging.getLogger("sparse1d")
        self.iteration_hook = iteration_hook
        self.lm_qx_min_scale = float(lm_qx_min_scale)
        self.eps = 1e-7
        self.hessian_floor = float(hessian_floor)
        self.param_dtype = torch.float32
        self.compute_dtype = torch.float32
        self.max_iter = int(max_iter)

        self.intercept_: Float[Tensor, "n_latents n_classes"] | None = None
        self.coef_: Float[Tensor, "n_latents n_classes"] | None = None
        self._qx_per_latent: Float[Tensor, "n_latents"] | None = None

    @torch.no_grad()
    def fit(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Float[Tensor, "n_samples n_classes"],
    ) -> "Sparse1DProbe":
        if not x.is_sparse_csr:
            raise TypeError("x must be a torch.sparse_csr_tensor.")

        n_samples, n_latents = x.shape
        if n_latents != self.n_latents:
            raise ValueError(f"x has {n_latents} latents, expected {self.n_latents}.")
        if y.shape != (n_samples, self.n_classes):
            raise ValueError(
                f"y has shape {tuple(y.shape)}, expected ({n_samples}, {self.n_classes})."
            )

        x = x.to(self.device, dtype=torch.float64)
        y = y.to(self.device, dtype=torch.float64)
        self.compute_dtype = x.dtype

        n_samples_f = float(n_samples)
        pi = torch.clamp(y.mean(dim=0), self.eps, 1 - self.eps)
        base_intercept = torch.log(pi / (1 - pi))

        intercept = (
            base_intercept.to(self.compute_dtype)
            .view(1, -1)
            .expand(self.n_latents, -1)
            .contiguous()
            .clone()
        )
        coef = torch.zeros(
            (self.n_latents, self.n_classes),
            dtype=self.compute_dtype,
            device=self.device,
        )

        col_indices = x.col_indices()
        values = x.values()
        nnz_per_latent = torch.zeros(
            self.n_latents, dtype=self.compute_dtype, device=self.device
        )
        nnz_per_latent.index_add_(
            0, col_indices, torch.ones_like(values, dtype=self.compute_dtype)
        )
        n_zeros_per_latent = (n_samples_f - nnz_per_latent).clamp(min=0.0).view(-1, 1)

        sum_sq = torch.zeros_like(nnz_per_latent)
        values_cast = values.to(self.compute_dtype)
        sum_sq.index_add_(0, col_indices, values_cast**2)

        qx = torch.ones_like(nnz_per_latent)
        nnz_long = nnz_per_latent.to(torch.long)
        if int(nnz_long.sum().item()) > 0:
            order = torch.argsort(col_indices)
            values_abs_sorted = values_cast.abs()[order]
            offsets = torch.zeros(
                self.n_latents + 1, dtype=torch.long, device=self.device
            )
            offsets[1:] = torch.cumsum(nnz_long, dim=0)
            for latent_idx in range(self.n_latents):
                start = int(offsets[latent_idx].item())
                end = int(offsets[latent_idx + 1].item())
                if end == start:
                    continue
                segment = values_abs_sorted[start:end]
                count = segment.numel()
                if count == 1:
                    q_val = segment[0]
                else:
                    k = int(math.ceil(0.95 * (count - 1)))
                    k = max(min(k, count - 1), 0)
                    q_val = segment.kthvalue(k + 1).values
                if not torch.isfinite(q_val) or q_val <= 0:
                    q_val = torch.sqrt((segment**2).mean())
                qx[latent_idx] = torch.clamp(q_val, min=1e-6)

        qx = torch.where(nnz_per_latent > 0, qx, torch.ones_like(qx))
        qx = torch.maximum(
            qx, torch.tensor(1e-6, dtype=self.compute_dtype, device=self.device)
        )
        qx = qx.view(-1, 1)
        qx_sq = qx * qx
        self._qx_per_latent = qx.view(-1).clone()

        lam = torch.full(
            (self.n_latents, self.n_classes),
            self.lam_init,
            dtype=self.compute_dtype,
            device=self.device,
        )
        row_indices = self._build_row_indices(x)
        prev_pred = torch.full_like(lam, float("nan"))
        prev_loss = torch.zeros_like(lam)
        prev_step_clipped = torch.zeros_like(lam, dtype=torch.bool)

        pi_total = y.sum(dim=0).to(self.compute_dtype)
        base_intercept_matrix = base_intercept.view(1, -1)

        row_indices = self._build_row_indices(x)

        for c0 in range(0, self.n_classes, self.class_slab_size):
            c1 = min(c0 + self.class_slab_size, self.n_classes)
            y_slab = y[:, c0:c1]
            intercept_slab = intercept[:, c0:c1]
            coef_slab = coef[:, c0:c1]
            lam_slab = lam[:, c0:c1]
            prev_pred_slab = prev_pred[:, c0:c1]
            prev_loss_slab = prev_loss[:, c0:c1]
            prev_step_clipped_slab = prev_step_clipped[:, c0:c1]

            pi_slab = pi_total[c0:c1].view(1, -1)
            base_slab = base_intercept_matrix[:, c0:c1]

            for iter_idx in range(self.max_iter):
                track_vram = self.device.type == "cuda" and self.logger.isEnabledFor(
                    logging.DEBUG
                )
                if track_vram:
                    torch.cuda.reset_peak_memory_stats(self.device)

                stats = self._compute_slab_stats(
                    x, y_slab, intercept_slab, coef_slab, row_indices
                )

                mu_0 = torch.sigmoid(intercept_slab).clamp_(self.eps, 1 - self.eps)
                s_0 = mu_0 * (1 - mu_0)

                g0 = stats["mu_nz"] + n_zeros_per_latent * mu_0 - pi_slab
                g1 = stats["g1"] + self.ridge * coef_slab
                g0 = g0 + self.ridge * (intercept_slab - base_slab)

                h0 = stats["h0"] + n_zeros_per_latent * s_0 + self.ridge
                h1 = stats["h1"]
                h2 = stats["h2"] + self.ridge

                qx_sq_slab = qx_sq.expand(-1, c1 - c0)
                curv_sq = h2 / torch.clamp(h0, min=1e-12)
                qx_sq_floor = torch.clamp(qx_sq_slab * self.lm_qx_min_scale, min=1e-6)
                qx_sq_step = torch.clamp(curv_sq, min=qx_sq_floor, max=qx_sq_slab)

                pos_zero = torch.clamp(pi_slab - stats["pos_nz"], min=0.0)
                pos_zero = torch.minimum(pos_zero, n_zeros_per_latent)
                neg_zero = n_zeros_per_latent - pos_zero
                zero_loss = -(
                    pos_zero * torch.log(mu_0)
                    + neg_zero * torch.log1p(-mu_0.clamp(max=1 - self.eps))
                )
                ridge_penalty = (
                    0.5
                    * self.ridge
                    * (coef_slab**2 + (intercept_slab - base_slab) ** 2)
                )
                loss_curr = stats["loss_nz"] + zero_loss + ridge_penalty

                mask_prev = torch.isfinite(prev_pred_slab)
                if mask_prev.any():
                    rho = torch.zeros_like(loss_curr)
                    rho[mask_prev] = (
                        prev_loss_slab[mask_prev] - loss_curr[mask_prev]
                    ) / torch.clamp(prev_pred_slab[mask_prev], min=1e-18)
                    grow_mask = mask_prev & ((rho <= 0.25) | prev_step_clipped_slab)
                    shrink_mask = mask_prev & (rho >= 0.75) & (~prev_step_clipped_slab)
                    lam_slab = torch.where(
                        shrink_mask, lam_slab * self.lam_lambda_shrink, lam_slab
                    )
                    lam_slab = torch.where(
                        grow_mask, lam_slab * self.lam_lambda_grow, lam_slab
                    )
                    lam_slab = lam_slab.clamp(self.lam_min, self.lam_max)

                db, dw, pred, lam_next, step_clipped = self._compute_lm_step(
                    g0=g0,
                    g1=g1,
                    h0=h0,
                    h1=h1,
                    h2=h2,
                    lam=lam_slab,
                    qx_sq=qx_sq_step,
                )

                intercept_slab = intercept_slab - db
                coef_slab = coef_slab - dw

                lam_slab = lam_next
                prev_pred_slab = pred
                prev_loss_slab = loss_curr
                prev_step_clipped_slab = step_clipped

                grad_norm = torch.maximum(g0.abs(), g1.abs()).max().item()
                step_norm = torch.maximum(db.abs(), dw.abs()).max().item()

                vram_alloc_mb: float | None = None
                vram_reserved_mb: float | None = None
                if track_vram:
                    torch.cuda.synchronize(self.device)
                    alloc = torch.cuda.max_memory_allocated(self.device)
                    reserved = torch.cuda.max_memory_reserved(self.device)
                    vram_alloc_mb = float(alloc) / (1024 * 1024)
                    vram_reserved_mb = float(reserved) / (1024 * 1024)
                    self.logger.debug(
                        "slab=%s iter=%d grad_max=%.3e step_max=%.3e "
                        "lambda_mean=%.3e peak_alloc_mb=%.2f peak_reserved_mb=%.2f",
                        (c0, c1),
                        iter_idx,
                        grad_norm,
                        step_norm,
                        float(lam_slab.mean().item()),
                        vram_alloc_mb,
                        vram_reserved_mb,
                    )
                elif self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "slab=%s iter=%d grad_max=%.3e step_max=%.3e lambda_mean=%.3e",
                        (c0, c1),
                        iter_idx,
                        grad_norm,
                        step_norm,
                        float(lam_slab.mean().item()),
                    )

                if self.iteration_hook is not None:
                    payload = {
                        "class_range": (c0, c1),
                        "iteration": iter_idx,
                        "max_grad": grad_norm,
                        "max_step": step_norm,
                        "lambda_mean": float(lam_slab.mean().item()),
                    }
                    if vram_alloc_mb is not None and vram_reserved_mb is not None:
                        payload["peak_alloc_mb"] = vram_alloc_mb
                        payload["peak_reserved_mb"] = vram_reserved_mb
                    self.iteration_hook(payload)
                if grad_norm < self.tol and step_norm < self.tol:
                    break

            intercept[:, c0:c1] = intercept_slab
            coef[:, c0:c1] = coef_slab
            lam[:, c0:c1] = lam_slab
            prev_pred[:, c0:c1] = prev_pred_slab
            prev_loss[:, c0:c1] = prev_loss_slab
            prev_step_clipped[:, c0:c1] = prev_step_clipped_slab

        self.intercept_ = intercept.to(torch.float32)
        self.coef_ = coef.to(torch.float32)
        return self

    def _compute_slab_stats(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y_slab: Float[Tensor, "n_samples c_b"],
        intercept_slab: Float[Tensor, "n_latents c_b"],
        coef_slab: Float[Tensor, "n_latents c_b"],
        row_indices: Tensor,
    ) -> dict[str, Tensor]:
        mu_nz = torch.zeros_like(intercept_slab)
        g1 = torch.zeros_like(intercept_slab)
        h0 = torch.zeros_like(intercept_slab)
        h1 = torch.zeros_like(intercept_slab)
        h2 = torch.zeros_like(intercept_slab)
        loss_nz = torch.zeros_like(intercept_slab)
        pos_nz = torch.zeros_like(intercept_slab)

        values_all = x.values().to(self.compute_dtype)
        cols_all = x.col_indices()

        for vals, idx, rows in self._iter_event_chunks(
            values_all, cols_all, row_indices
        ):
            vals = vals.view(-1, 1)

            b_cols = intercept_slab[idx]
            w_cols = coef_slab[idx]
            logits = b_cols + w_cols * vals
            mu = torch.sigmoid(logits)
            s = mu * (1 - mu)

            y_chunk = y_slab[rows].to(self.compute_dtype)
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

        return {
            "mu_nz": mu_nz,
            "g1": g1,
            "h0": h0,
            "h1": h1,
            "h2": h2,
            "loss_nz": loss_nz,
            "pos_nz": pos_nz,
        }

    def _compute_lm_step(
        self,
        *,
        g0: Tensor,
        g1: Tensor,
        h0: Tensor,
        h1: Tensor,
        h2: Tensor,
        lam: Tensor,
        qx_sq: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        lam_curr = lam.clone()
        success = torch.zeros_like(g0, dtype=torch.bool)
        db = torch.zeros_like(g0)
        dw = torch.zeros_like(g0)
        pred = torch.zeros_like(g0)
        step_clipped = torch.zeros_like(g0, dtype=torch.bool)

        for _ in range(self.lm_max_adapt_iters):
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
                    lam_curr[failure] * self.lam_lambda_grow,
                    min=self.lam_min,
                    max=self.lam_max,
                )

        if (~success).any():
            raise RuntimeError("LM step failed to converge for some coordinates.")

        lam_curr = lam_curr.clamp(self.lam_min, self.lam_max)
        return db, dw, pred, lam_curr, step_clipped

    def _build_row_indices(self, x: Tensor) -> Tensor:
        crow_indices = x.crow_indices()
        lengths = crow_indices[1:] - crow_indices[:-1]
        row_idx = torch.repeat_interleave(
            torch.arange(x.shape[0], device=x.device, dtype=torch.long),
            lengths,
        )
        return row_idx

    def _iter_event_chunks(
        self, values: Tensor, col_indices: Tensor, row_indices: Tensor
    ) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        chunk_size = max(1, self.row_batch_size * 32)
        nnz = values.numel()
        for start in range(0, nnz, chunk_size):
            end = min(start + chunk_size, nnz)
            yield values[start:end], col_indices[start:end], row_indices[start:end]

    def _compute_loss_slab(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y_slab: Tensor,
        c0: int,
        c1: int,
        n_samples: int,
        pi_slab: Tensor,
        n_zeros_per_latent: Tensor,
        row_indices: Tensor,
    ) -> Tensor:
        loss = torch.zeros(
            (self.n_latents, c1 - c0), dtype=self.compute_dtype, device=self.device
        )
        pos_nz = torch.zeros_like(loss)

        intercept_slab = self.intercept_[:, c0:c1].to(self.compute_dtype)
        coef_slab = self.coef_[:, c0:c1].to(self.compute_dtype)

        values_all = x.values().to(self.compute_dtype)
        cols_all = x.col_indices()
        for vals, cols_chunk, row_idx_chunk in self._iter_event_chunks(
            values_all, cols_all, row_indices
        ):
            y_nz = y_slab[row_idx_chunk].to(self.compute_dtype)

            vals_expanded = vals.view(-1, 1)
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

        return (loss / float(n_samples)).to(self.param_dtype)

    def _compute_loss(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Float[Tensor, "n_samples n_classes"],
    ) -> Float[Tensor, "n_latents n_classes"]:
        n_samples = x.shape[0]
        loss = torch.zeros(
            (self.n_latents, self.n_classes),
            dtype=self.compute_dtype,
            device=self.device,
        )

        col_indices = x.col_indices()
        nnz_per_latent = torch.bincount(col_indices, minlength=self.n_latents)
        n_zeros_per_latent = (
            (n_samples - nnz_per_latent)
            .to(device=self.device, dtype=self.compute_dtype)
            .view(-1, 1)
        )

        pi = y.sum(dim=0).to(self.compute_dtype)
        row_indices = self._build_row_indices(x)

        for c0 in range(0, self.n_classes, self.class_slab_size):
            c1 = min(c0 + self.class_slab_size, self.n_classes)
            y_slab = y[:, c0:c1].to(self.device, dtype=self.compute_dtype)
            pi_slab = pi[c0:c1].to(self.device)
            loss[:, c0:c1] = self._compute_loss_slab(
                x,
                y_slab,
                c0,
                c1,
                n_samples,
                pi_slab,
                n_zeros_per_latent,
                row_indices,
            )
            del y_slab, pi_slab

        return loss.to(self.param_dtype)

    def loss_matrix(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Bool[Tensor, "n_samples n_classes"],
    ) -> Float[Tensor, "n_latents n_classes"]:
        sklearn.utils.validation.check_is_fitted(self, "intercept_")
        sklearn.utils.validation.check_is_fitted(self, "coef_")
        x = x.to(self.device, dtype=self.compute_dtype)
        y = y.to(self.device).to(self.compute_dtype)
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
        row_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tp = torch.zeros(
            (self.n_latents, c1 - c0), dtype=self.compute_dtype, device=self.device
        )
        fp = torch.zeros_like(tp)
        tn = torch.zeros_like(tp)
        fn = torch.zeros_like(tp)
        pos_nz = torch.zeros_like(tp)

        intercept_slab = self.intercept_[:, c0:c1].to(self.compute_dtype)
        coef_slab = self.coef_[:, c0:c1].to(self.compute_dtype)

        mu_0 = torch.sigmoid(intercept_slab)
        pred_zero = mu_0 > threshold

        values_all = x.values().to(self.compute_dtype)
        cols_all = x.col_indices()
        for vals, cols_chunk, row_idx_chunk in self._iter_event_chunks(
            values_all, cols_all, row_indices
        ):
            y_nz = y_slab[row_idx_chunk].to(self.compute_dtype)
            y_nz_bool = y_nz > 0.5

            vals_expanded = vals.view(-1, 1)
            b_cols = intercept_slab[cols_chunk]
            w_cols = coef_slab[cols_chunk]
            mu = torch.sigmoid(b_cols + w_cols * vals_expanded)
            pred_nz_bool = mu > threshold

            tp_chunk = torch.logical_and(pred_nz_bool, y_nz_bool).to(self.compute_dtype)
            fp_chunk = torch.logical_and(pred_nz_bool, ~y_nz_bool).to(
                self.compute_dtype
            )
            fn_chunk = torch.logical_and(~pred_nz_bool, y_nz_bool).to(
                self.compute_dtype
            )
            tn_chunk = torch.logical_and(~pred_nz_bool, ~y_nz_bool).to(
                self.compute_dtype
            )

            tp.index_add_(0, cols_chunk, tp_chunk)
            fp.index_add_(0, cols_chunk, fp_chunk)
            fn.index_add_(0, cols_chunk, fn_chunk)
            tn.index_add_(0, cols_chunk, tn_chunk)
            pos_nz.index_add_(0, cols_chunk, y_nz)

        pos_zero = torch.clamp(pi_slab.view(1, c1 - c0) - pos_nz, min=0.0)
        pos_zero = torch.minimum(pos_zero, n_zeros_per_latent)
        neg_zero = n_zeros_per_latent - pos_zero

        zero_mask = pred_zero.to(self.compute_dtype)
        tp_zero = zero_mask * pos_zero
        fp_zero = zero_mask * neg_zero
        fn_zero = (1.0 - zero_mask) * pos_zero
        tn_zero = (1.0 - zero_mask) * neg_zero

        tp = tp + tp_zero
        fp = fp + fp_zero
        fn = fn + fn_zero
        tn = tn + tn_zero

        return (
            tp.to(self.param_dtype),
            fp.to(self.param_dtype),
            tn.to(self.param_dtype),
            fn.to(self.param_dtype),
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

        x = x.to(self.device, dtype=self.compute_dtype)
        n_samples = x.shape[0]

        loss = self._compute_loss(x, y.to(self.compute_dtype))

        col_indices = x.col_indices()
        nnz_per_latent = torch.bincount(col_indices, minlength=self.n_latents)
        n_zeros_per_latent = (
            (n_samples - nnz_per_latent)
            .to(device=self.device, dtype=self.compute_dtype)
            .view(-1, 1)
        )

        pi = y.to(self.compute_dtype).sum(dim=0).to(self.compute_dtype)

        tp = torch.zeros(
            (self.n_latents, self.n_classes),
            dtype=self.compute_dtype,
            device=self.device,
        )
        fp = torch.zeros_like(tp)
        tn = torch.zeros_like(tp)
        fn = torch.zeros_like(tp)

        row_indices = self._build_row_indices(x)

        for c0 in range(0, self.n_classes, self.class_slab_size):
            c1 = min(c0 + self.class_slab_size, self.n_classes)

            y_slab = y[:, c0:c1].to(self.device).to(self.compute_dtype)
            pi_slab = pi[c0:c1].to(self.device)

            tp_chunk, fp_chunk, tn_chunk, fn_chunk = self._compute_confusion_slab(
                x,
                y_slab,
                c0,
                c1,
                pi_slab,
                n_zeros_per_latent,
                threshold,
                row_indices,
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
