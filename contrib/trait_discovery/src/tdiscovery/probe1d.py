"""Sparse 1D logistic probes for trait discovery.

This module implements Newton-style optimization and evaluation for
per-latent / per-class logistic probes on high-sparsity SAE activations.
The key invariants across implementations are:

* Sparse feature matrix `x` is streamed in CSR format without materializing
  tensors shaped `(nnz, n_classes)`.
* Classes are processed in configurable slabs (`class_slab_size`) while rows are
  processed in configurable micro-batches (`row_batch_size`).
* All compute paths (`fit`, `loss_matrix`, `loss_matrix_with_aux`) share the
  same sparse event iterator to guarantee identical traversal order.

The public surface area is intentionally small and designed to be used by tests
and training sweeps. The heavy lifting occurs in `Sparse1DProbe`, which exposes
the learned coefficients, loss computation helpers, and accuracy diagnostics.
"""

import logging
from collections.abc import Iterator
from typing import NamedTuple

import beartype
import einops
import sklearn.base
import torch
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

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
    ):
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

    @torch.no_grad()
    def fit(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Float[Tensor, "n_samples n_classes"],
    ):
        assert x.layout == torch.sparse_csr

        n_samples, n_latents = x.shape
        assert n_latents == self.n_latents

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

            intercept_slab = self.intercept_[:, c0:c1]
            coef_slab = self.coef_[:, c0:c1]

            for it in range(self.n_iter):
                # For x_j = 0, we can pre-calculate mu and s.
                mu_0 = torch.sigmoid(intercept_slab).clamp_(self.eps, 1 - self.eps)
                s_0 = mu_0 * (1 - mu_0)

                # All are R^{D x Cb}
                mu_nz = torch.zeros(
                    (self.n_latents, c1 - c0), device=device, dtype=torch.float32
                )
                g1 = torch.zeros(
                    (self.n_latents, c1 - c0), device=device, dtype=torch.float32
                )
                h0 = torch.zeros(
                    (self.n_latents, c1 - c0), device=device, dtype=torch.float32
                )
                h1 = torch.zeros(
                    (self.n_latents, c1 - c0), device=device, dtype=torch.float32
                )
                h2 = torch.zeros(
                    (self.n_latents, c1 - c0), device=device, dtype=torch.float32
                )

                for batch in self._iter_sparse_events(x):
                    self.logger.debug(
                        "Rows %d:%d: %d non-zero elements.",
                        batch.row_start,
                        batch.row_end,
                        batch.values.numel(),
                    )

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

                g0 = mu_nz + n_zeros_per_latent * mu_0 - pi_device[c0:c1]
                g1 = g1 + self.ridge * coef_slab

                h0 = h0 + n_zeros_per_latent * s_0 + self.ridge
                h2 = h2 + self.ridge

                det_h = h0 * h2 - h1 * h1
                det_h_sign = det_h.sign()
                det_h_sign = torch.where(
                    det_h_sign == 0, torch.ones_like(det_h_sign), det_h_sign
                )
                det_h = torch.where(det_h.abs() < 1e-12, det_h_sign * 1e-12, det_h)

                db = (h2 * g0 - h1 * g1) / det_h
                dw = (-h1 * g0 + h0 * g1) / det_h

                intercept_slab -= db
                coef_slab -= dw

                if not check_convergence:
                    continue

                max_grad = torch.stack((g0.abs().max(), g1.abs().max())).max()
                max_update = torch.stack((db.abs().max(), dw.abs().max())).max()

                self.logger.info(
                    "Classes %s:%s at iteration %s: grad=%s, update=%s",
                    c0,
                    c1,
                    it,
                    max_grad.item(),
                    max_update.item(),
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
        y_slab: Tensor,  # Float[Tensor, "n_samples Cb"]
        c0: int,
        c1: int,
        n_samples: int,
        pi_slab: Tensor,
        n_zeros_per_latent: Tensor,
    ) -> Tensor:
        """Compute loss for a class slab using event chunking."""

        Cb = c1 - c0
        loss = torch.zeros(
            (self.n_latents, Cb), dtype=torch.float32, device=self.device
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

        pos_zero = torch.clamp(pi_slab.view(1, Cb) - pos_nz, min=0.0)
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

    def _compute_accuracy_slab(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y_slab: Tensor,
        c0: int,
        c1: int,
        n_samples: int,
        pi_slab: Tensor,
        n_zeros_per_latent: Tensor,
    ) -> Tensor:
        """Compute accuracy for a class slab using event chunking."""

        Cb = c1 - c0
        correct = torch.zeros(
            (self.n_latents, Cb), dtype=torch.float32, device=self.device
        )
        pos_nz = torch.zeros_like(correct)

        intercept_slab = self.intercept_[:, c0:c1]
        coef_slab = self.coef_[:, c0:c1]

        mu_0 = torch.sigmoid(intercept_slab).clamp_(self.eps, 1 - self.eps)
        pred_zero = mu_0 > 0.5

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
            pred_nz_bool = mu > 0.5

            nz_correct = (pred_nz_bool == y_nz_bool).to(torch.float32)

            correct.index_add_(0, cols_chunk, nz_correct)
            pos_nz.index_add_(0, cols_chunk, y_nz)

        pos_zero = torch.clamp(pi_slab.view(1, Cb) - pos_nz, min=0.0)
        pos_zero = torch.minimum(pos_zero, n_zeros_per_latent)
        neg_zero = n_zeros_per_latent - pos_zero
        correct_zero = torch.where(pred_zero, pos_zero, neg_zero)
        correct = correct + correct_zero

        return correct / float(n_samples)

    @torch.no_grad()
    def loss_matrix_with_aux(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Bool[Tensor, "n_samples n_classes"],
    ) -> tuple[Float[Tensor, "n_latents n_classes"], dict]:
        """Returns the NLL loss matrix and additional metadata needed to construct the parquet file."""
        sklearn.utils.validation.check_is_fitted(self, "intercept_")
        sklearn.utils.validation.check_is_fitted(self, "coef_")

        # Move sparse matrix to device
        x = x.to(self.device)
        n_samples = x.shape[0]

        # Compute loss using chunked implementation
        loss = self._compute_loss(x, y)

        # Count nnz per latent
        col_indices = x.col_indices()
        nnz_per_latent = torch.bincount(col_indices, minlength=self.n_latents)
        n_zeros_per_latent = (
            (n_samples - nnz_per_latent)
            .to(device=self.device, dtype=torch.float32)
            .view(-1, 1)
        )

        # Compute accuracy using chunked implementation
        pi = y.sum(dim=0, dtype=torch.float64).to(torch.float32)

        acc = torch.zeros(
            (self.n_latents, self.n_classes), dtype=torch.float32, device=self.device
        )

        for c0 in range(0, self.n_classes, self.class_slab_size):
            c1 = min(c0 + self.class_slab_size, self.n_classes)

            y_slab = y[:, c0:c1].to(self.device).to(torch.float32)
            pi_slab = pi[c0:c1].to(self.device)

            acc[:, c0:c1] = self._compute_accuracy_slab(
                x,
                y_slab,
                c0,
                c1,
                n_samples,
                pi_slab,
                n_zeros_per_latent,
            )

            del y_slab, pi_slab

        aux = {
            "accuracy": acc,
            "nnz_per_latent": nnz_per_latent,
            "n_samples": n_samples,
            "coef": self.coef_,
            "intercept": self.intercept_,
        }

        return loss, aux
