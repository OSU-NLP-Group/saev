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


@jaxtyped(typechecker=beartype.beartype)
class Sparse1DProbe(sklearn.base.BaseEstimator):
    """Newton-Raphson optimizer for 1D logistic regression.

    `fit(x, y)` streams sparse x and optimizes (b, w) for every (latent, class) pair.
    Results are exposed as attributes and helper methods.

    To make fit() memory-efficient: tile across classes and stream over rows. Never create anything shaped (nnz, n_classes).

    Args:

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

        device = torch.device(self.device)

        x = x.to(device)
        y = y.to(dtype=torch.float32, device="cpu")

        self.coef_ = torch.zeros(
            (self.n_latents, self.n_classes), device=device, dtype=torch.float32
        )
        self.intercept_ = torch.zeros(
            (self.n_latents, self.n_classes), device=device, dtype=torch.float32
        )

        pi = y.sum(dim=0)
        prevalence = torch.clamp(pi / float(n_samples), self.eps, 1 - self.eps)
        base_intercept = torch.logit(prevalence).to(device)
        self.intercept_.copy_(
            einops.repeat(base_intercept, "c -> l c", l=self.n_latents)
        )

        nnz_per_latent = torch.bincount(x.col_indices(), minlength=self.n_latents)
        n_zeros_per_latent = (
            (n_samples - nnz_per_latent)
            .to(device=device, dtype=torch.float32)
            .view(-1, 1)
        )

        pi_device = pi.to(device)
        check_convergence = self.tol > 0

        for c0, c1 in saev.helpers.batched_idx(self.n_classes, self.class_slab_size):
            self.logger.debug(f"Processing classes {c0}:{c1} ({c1 - c0} classes)")

            y_slab = y[:, c0:c1]
            if device.type != "cpu":
                y_slab = y_slab.pin_memory()
            y_slab_device = y_slab.to(device, non_blocking=device.type != "cpu")
            pi_slab_device = pi_device[c0:c1]

            intercept_slab = self.intercept_[:, c0:c1]
            coef_slab = self.coef_[:, c0:c1]
            n_classes_slab = c1 - c0

            best_loss_per_class = torch.full(
                (n_classes_slab,),
                float("inf"),
                device=device,
                dtype=torch.float32,
            )
            best_latent_per_class = torch.full(
                (n_classes_slab,), -1, device=device, dtype=torch.int64
            )
            best_coef_per_class = torch.zeros(
                (n_classes_slab,), device=device, dtype=torch.float32
            )
            best_intercept_per_class = torch.zeros(
                (n_classes_slab,), device=device, dtype=torch.float32
            )
            lambda_slab = torch.full(
                (self.n_latents, n_classes_slab),
                self.lm_lambda_init,
                device=device,
                dtype=torch.float32,
            )

            for it in range(self.n_iter):
                # For x_j = 0, we can pre-calculate mu and s.
                mu_0 = torch.sigmoid(intercept_slab).clamp_(self.eps, 1 - self.eps)
                s_0 = mu_0 * (1 - mu_0)

                # All are R^{D x c_b}
                mu_nz = torch.zeros_like(intercept_slab)
                g1 = torch.zeros_like(intercept_slab)
                h0 = torch.zeros_like(intercept_slab)
                h1 = torch.zeros_like(intercept_slab)
                h2 = torch.zeros_like(intercept_slab)
                loss_nz = torch.zeros_like(mu_nz)
                pos_nz = torch.zeros_like(mu_nz)

                for batch in self._iter_sparse_events(x):
                    values_E1 = batch.values[:, None]

                    eta_EC = (
                        self.intercept_[batch.latent_idx, c0:c1]
                        + self.coef_[batch.latent_idx, c0:c1] * values_E1
                    )
                    mu_EC = torch.sigmoid(eta_EC).clamp_(self.eps, 1 - self.eps)
                    s_EC = mu_EC * (1 - mu_EC)

                    y_nz_EC = y_slab_device[batch.row_idx, :]

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

                g0 = mu_nz + n_zeros_per_latent * mu_0 - pi_slab_device
                g1 = g1 + self.ridge * coef_slab

                h0 = h0 + n_zeros_per_latent * s_0 + self.ridge
                h2 = h2 + self.ridge
                mean_loss_value = float("nan")
                det_h_raw = h0 * h2 - h1 * h1
                h0_clamped = torch.clamp(h0, min=self.hessian_floor)
                h2_clamped = torch.clamp(h2, min=self.hessian_floor)
                h0 = h0_clamped + lambda_slab
                h2 = h2_clamped + lambda_slab
                pos_zero = torch.clamp(
                    pi_slab_device.view(1, n_classes_slab) - pos_nz, min=0.0
                )
                pos_zero = torch.minimum(pos_zero, n_zeros_per_latent)
                neg_zero = n_zeros_per_latent - pos_zero
                zero_loss = -(
                    pos_zero * torch.log(mu_0) + neg_zero * torch.log1p(-mu_0)
                )
                loss_slab = (loss_nz + zero_loss) / n_samples_f
                mean_loss_value = loss_slab.mean().item()

                curr_best_loss, curr_best_latent_idx = loss_slab.min(dim=0)
                improved_mask = torch.logical_or(
                    curr_best_latent_idx != best_latent_per_class,
                    curr_best_loss + 1e-6 < best_loss_per_class,
                )
                if improved_mask.any():
                    class_idx_vec = torch.arange(
                        n_classes_slab, device=device, dtype=torch.int64
                    )
                    selected_coef = coef_slab[curr_best_latent_idx, class_idx_vec]
                    selected_intercept = intercept_slab[
                        curr_best_latent_idx, class_idx_vec
                    ]
                    best_loss_per_class = torch.where(
                        improved_mask, curr_best_loss, best_loss_per_class
                    )
                    best_latent_per_class = torch.where(
                        improved_mask, curr_best_latent_idx, best_latent_per_class
                    )
                    best_coef_per_class = torch.where(
                        improved_mask, selected_coef, best_coef_per_class
                    )
                    best_intercept_per_class = torch.where(
                        improved_mask, selected_intercept, best_intercept_per_class
                    )

                det_h = h0 * h2 - h1 * h1
                det_h_sign = det_h.sign()
                det_h_sign = torch.where(
                    det_h_sign == 0, torch.ones_like(det_h_sign), det_h_sign
                )
                det_h = torch.where(det_h.abs() < 1e-12, det_h_sign * 1e-12, det_h)

                db = (h2 * g0 - h1 * g1) / det_h
                dw = (-h1 * g0 + h0 * g1) / det_h
                predicted_reduction = 0.5 * (db * g0 + dw * g1)

                intercept_slab -= db
                coef_slab -= dw
                lambda_prev = lambda_slab
                lambda_success = torch.clamp(
                    lambda_prev * self.lm_lambda_shrink,
                    min=self.lm_lambda_min,
                    max=self.lm_lambda_max,
                )
                lambda_fail = torch.clamp(
                    lambda_prev * self.lm_lambda_grow,
                    min=self.lm_lambda_min,
                    max=self.lm_lambda_max,
                )
                lambda_slab = torch.where(
                    predicted_reduction > 0.0, lambda_success, lambda_fail
                )

                if not check_convergence:
                    continue

                abs_g0 = g0.abs()
                abs_g1 = g1.abs()
                abs_db = db.abs()
                abs_dw = dw.abs()

                g0_flat = abs_g0.view(-1)
                g1_flat = abs_g1.view(-1)
                db_flat = abs_db.view(-1)
                dw_flat = abs_dw.view(-1)

                max_g0_val, max_g0_idx = g0_flat.max(dim=0)
                max_g1_val, max_g1_idx = g1_flat.max(dim=0)
                max_db_val, max_db_idx = db_flat.max(dim=0)
                max_dw_val, max_dw_idx = dw_flat.max(dim=0)

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

                grad_class_idx = c0 + grad_class_offset
                update_class_idx = c0 + update_class_offset

                grad_loss_value = loss_slab[grad_latent_idx, grad_class_offset].item()
                update_loss_value = loss_slab[
                    update_latent_idx, update_class_offset
                ].item()

                grad_coef_value = coef_slab[grad_latent_idx, grad_class_offset].item()
                grad_intercept_value = intercept_slab[
                    grad_latent_idx, grad_class_offset
                ].item()
                update_coef_value = coef_slab[
                    update_latent_idx, update_class_offset
                ].item()
                update_intercept_value = intercept_slab[
                    update_latent_idx, update_class_offset
                ].item()

                grad_det_raw = det_h_raw[grad_latent_idx, grad_class_offset].item()
                grad_det_clamped = det_h[grad_latent_idx, grad_class_offset].item()
                update_det_raw = det_h_raw[
                    update_latent_idx, update_class_offset
                ].item()
                update_det_clamped = det_h[
                    update_latent_idx, update_class_offset
                ].item()

                grad_nnz = nnz_per_latent[grad_latent_idx].item()
                update_nnz = nnz_per_latent[update_latent_idx].item()

                quantile_targets = torch.tensor((0.5, 0.95), device=device)
                g0_quantiles = torch.quantile(g0_flat, quantile_targets).tolist()
                g1_quantiles = torch.quantile(g1_flat, quantile_targets).tolist()
                loss_quantiles = torch.quantile(
                    loss_slab.view(-1), quantile_targets
                ).tolist()

                max_grad = torch.stack((max_g0_val, max_g1_val)).max()
                max_update = torch.stack((max_db_val, max_dw_val)).max()
                lambda_min_value = lambda_slab.min().item()
                lambda_max_value = lambda_slab.max().item()

                self.logger.info(
                    (
                        "Classes %s:%s at iteration %s: loss=%.6f, grad=%s, update=%s "
                        "lambda=[%.3e, %.3e]"
                    ),
                    c0,
                    c1,
                    it,
                    mean_loss_value,
                    max_grad.item(),
                    max_update.item(),
                    lambda_min_value,
                    lambda_max_value,
                )

                if improved_mask.any():
                    improved_classes = torch.nonzero(
                        improved_mask, as_tuple=False
                    ).flatten()
                    max_report = min(4, int(improved_classes.numel()))
                    summary_parts = []
                    for idx in improved_classes[:max_report]:
                        class_idx = c0 + int(idx.item())
                        latent_idx = int(curr_best_latent_idx[idx].item())
                        summary_parts.append(
                            f"{class_idx}->{latent_idx} loss={curr_best_loss[idx].item():.6f}"
                        )
                    if improved_classes.numel() > max_report:
                        remaining = int(improved_classes.numel() - max_report)
                        summary_parts.append(f"+{remaining} more")
                    self.logger.info(
                        "improved_best_latents %s", ", ".join(summary_parts)
                    )

                if self.logger.isEnabledFor(logging.DEBUG):
                    grad_g0_value = g0[grad_latent_idx, grad_class_offset].item()
                    grad_mu_nz_value = mu_nz[grad_latent_idx, grad_class_offset].item()
                    grad_mu0_value = mu_0[grad_latent_idx, grad_class_offset].item()
                    grad_n_zeros_value = n_zeros_per_latent[grad_latent_idx, 0].item()
                    grad_g0_zero_value = grad_n_zeros_value * grad_mu0_value
                    grad_pi_value = pi_slab_device[grad_class_offset].item()
                    grad_g1_value = g1[grad_latent_idx, grad_class_offset].item()
                    grad_g1_ridge_value = self.ridge * grad_coef_value
                    grad_g1_nz_value = grad_g1_value - grad_g1_ridge_value
                    grad_h0_value = h0[grad_latent_idx, grad_class_offset].item()
                    grad_h1_value = h1[grad_latent_idx, grad_class_offset].item()
                    grad_h2_value = h2[grad_latent_idx, grad_class_offset].item()

                    update_g0_value = g0[update_latent_idx, update_class_offset].item()
                    update_mu_nz_value = mu_nz[
                        update_latent_idx, update_class_offset
                    ].item()
                    update_mu0_value = mu_0[
                        update_latent_idx, update_class_offset
                    ].item()
                    update_n_zeros_value = n_zeros_per_latent[
                        update_latent_idx, 0
                    ].item()
                    update_g0_zero_value = update_n_zeros_value * update_mu0_value
                    update_pi_value = pi_slab_device[update_class_offset].item()
                    update_g1_value = g1[update_latent_idx, update_class_offset].item()
                    update_g1_ridge_value = self.ridge * update_coef_value
                    update_g1_nz_value = update_g1_value - update_g1_ridge_value
                    update_h0_value = h0[update_latent_idx, update_class_offset].item()
                    update_h1_value = h1[update_latent_idx, update_class_offset].item()
                    update_h2_value = h2[update_latent_idx, update_class_offset].item()
                    update_db_value = db[update_latent_idx, update_class_offset].item()
                    update_dw_value = dw[update_latent_idx, update_class_offset].item()
                    grad_lambda_value = lambda_prev[
                        grad_latent_idx, grad_class_offset
                    ].item()
                    update_lambda_value = lambda_prev[
                        update_latent_idx, update_class_offset
                    ].item()

                    self.logger.debug(
                        "worst_grad source=%s latent=%s class=%s nnz=%s coef=%s intercept=%s loss=%s det_raw=%s det_clamped=%s g0_q50=%s g0_q95=%s g1_q50=%s g1_q95=%s loss_q50=%s loss_q95=%s",
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
                    self.logger.debug(
                        "worst_update source=%s latent=%s class=%s nnz=%s coef=%s intercept=%s loss=%s det_raw=%s det_clamped=%s delta=%s",
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
                    self.logger.debug(
                        "worst_grad_diagnostics g0=%s g0_nz=%s g0_zero=%s g0_pi=%s "
                        "g1=%s g1_nz=%s g1_ridge=%s h0=%s h1=%s h2=%s lambda=%s",
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
                    self.logger.debug(
                        "worst_update_diagnostics g0=%s g0_nz=%s g0_zero=%s g0_pi=%s "
                        "g1=%s g1_nz=%s g1_ridge=%s h0=%s h1=%s h2=%s lambda=%s db=%s dw=%s",
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
                if torch.isnan(max_grad) or torch.isnan(max_update):
                    continue

                if max_grad < self.tol and max_update < self.tol:
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
        self,
        x: Float[Tensor, "n_samples n_latents"],
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
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=8,
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
