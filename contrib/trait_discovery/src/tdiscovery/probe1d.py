"""Per-feature x per-class 1D logistic probes on SAE activations."""

import logging

import beartype
import einops
import sklearn.base
import torch
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype.beartype)
class Sparse1DProbe(sklearn.base.BaseEstimator):
    """Newton-Raphson optimizer for 1D logistic regression.

    `fit(x, y)` streams sparse x and optimizes (b, w) for every (latent, class) pair.
    Results are exposed as attributes and helper methods.
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
    ):
        self.n_latents = n_latents
        self.n_classes = n_classes
        self.tol = tol
        self.device = device
        self.n_iter = n_iter
        self.ridge = ridge  # L2 regularization strength
        self.logger = logging.getLogger("sparse1d")
        self.eps = 1e-15

    @torch.no_grad()
    def fit(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Float[Tensor, "n_samples n_classes"],
    ):
        assert x.layout == torch.sparse_csr

        n_samples, n_latents = x.shape
        assert n_latents == self.n_latents
        shape = (self.n_latents, self.n_classes)
        dd = dict(dtype=torch.float32, device=self.device)

        # Move data to device
        x = x.to(self.device)
        y = y.to(self.device)

        # Initialize parameters
        self.coef_ = torch.zeros(shape, **dd)
        self.intercept_ = torch.zeros(shape, **dd)

        # Calculate initial intercept as logit of class prevalence
        pi = y.sum(dim=0)  # Total positives per class [n_classes]
        prevalence = pi / n_samples
        # Clip to avoid log(0) or log(1)
        prevalence = torch.clamp(prevalence, self.eps, 1 - self.eps)
        # Broadcast to all latents
        self.intercept_[:] = einops.repeat(
            torch.logit(prevalence), "c -> l c", l=self.n_latents
        )

        # Get CSR components
        crow_indices = x.crow_indices()
        col_indices = x.col_indices()
        values = x.values()

        prev_loss = torch.full(shape, torch.inf, **dd)

        for it in range(self.n_iter):
            # Compute mu_0 and s_0 for zero entries (when x_j = 0)
            mu_0 = torch.sigmoid(self.intercept_)  # [n_latents, n_classes]
            s_0 = mu_0 * (1 - mu_0)

            # Work with all non-zero entries at once
            # Get row indices for each non-zero value
            row_indices = torch.repeat_interleave(
                torch.arange(n_samples, device=self.device),
                crow_indices[1:] - crow_indices[:-1],
            )

            # Get the y values for each non-zero entry
            y_nz = y[row_indices]  # [nnz, n_classes]

            # Compute logits, mu, and s for all non-zero entries at once
            # x_values has shape [nnz], col_indices has shape [nnz]
            x_values_expanded = einops.rearrange(values, "nnz -> nnz 1")

            # self.intercept_[col_indices] has shape [nnz, n_classes]
            # self.coef_[col_indices] has shape [nnz, n_classes]
            eta = (
                self.intercept_[col_indices]
                + self.coef_[col_indices] * x_values_expanded
            )
            mu = torch.sigmoid(eta)
            mu = torch.clamp(mu, self.eps, 1 - self.eps)
            s = mu * (1 - mu)

            # Compute residuals
            residual = mu - y_nz

            # Accumulate statistics using scatter_add
            m_nz = torch.zeros(shape, **dd)
            g1 = torch.zeros(shape, **dd)
            s0 = torch.zeros(shape, **dd)
            s1 = torch.zeros(shape, **dd)
            s2 = torch.zeros(shape, **dd)

            # Use index_add to accumulate values for each latent
            m_nz.index_add_(0, col_indices, mu)
            g1.index_add_(0, col_indices, residual * x_values_expanded)
            s0.index_add_(0, col_indices, s)
            s1.index_add_(0, col_indices, s * x_values_expanded)
            s2.index_add_(0, col_indices, s * x_values_expanded**2)

            # Count nnz per latent
            nnz_per_latent = torch.bincount(col_indices, minlength=self.n_latents)

            # Add contributions from zero entries
            n_zeros_per_latent = n_samples - nnz_per_latent  # [n_latents]
            n_zeros_expanded = einops.rearrange(n_zeros_per_latent.float(), "l -> l 1")

            # G0 = M_nz + (N-n)*mu_0 - pi
            pi_expanded = einops.rearrange(pi, "c -> 1 c")
            g0 = m_nz + n_zeros_expanded * mu_0 - pi_expanded

            # Add L2 regularization gradient for weights
            g1 = g1 + self.ridge * self.coef_

            # S0 includes zero contributions
            s0 = s0 + n_zeros_expanded * s_0

            # Add L2 regularization to diagonal of Hessian
            # Add to both s0 and s2 for numerical stability
            s0 = s0 + self.ridge
            s2 = s2 + self.ridge

            # Calculate Newton updates
            det_h = s0 * s2 - s1 * s1

            # Check for numerical issues
            det_h = torch.where(
                torch.abs(det_h) < 1e-10, torch.ones_like(det_h) * 1e-10, det_h
            )

            # Newton updates
            db = (s2 * g0 - s1 * g1) / det_h
            dw = (-s1 * g0 + s0 * g1) / det_h

            # Apply updates
            self.intercept_ -= db
            self.coef_ -= dw

            # Calculate loss for convergence check
            self.loss_ = self._compute_loss(x, y)

            # Check convergence
            loss_change = torch.abs(prev_loss - self.loss_)
            if torch.all(loss_change < self.tol):
                self.logger.debug(f"Converged at iteration {it}")
                break

            prev_loss = self.loss_.clone()
        else:
            self.logger.debug(f"Did not converge after {self.n_iter} iterations")

    def _compute_loss(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Float[Tensor, "n_samples n_classes"],
    ) -> Float[Tensor, "n_latents n_classes"]:
        """Compute negative log-likelihood loss for all (latent, class) pairs."""
        n_samples = x.shape[0]
        loss = torch.zeros(
            (self.n_latents, self.n_classes), dtype=torch.float32, device=self.device
        )

        # Get CSR components
        crow_indices = x.crow_indices()
        col_indices = x.col_indices()
        values = x.values()

        # Compute mu_0 for zero entries
        mu_0 = torch.sigmoid(self.intercept_)
        mu_0 = torch.clamp(mu_0, self.eps, 1 - self.eps)

        # Initialize with contribution from all-zeros (will subtract non-zeros)
        # For zero entries: -y*log(mu_0) - (1-y)*log(1-mu_0)
        # Vectorized computation for all latents and classes at once
        n_pos = y.sum(dim=0)  # [n_classes] - number of positive samples per class
        n_neg = n_samples - n_pos  # [n_classes] - number of negative samples per class

        # Broadcast n_pos and n_neg to [n_latents, n_classes]
        n_pos_expanded = einops.repeat(n_pos, "c -> l c", l=self.n_latents)
        n_neg_expanded = einops.repeat(n_neg, "c -> l c", l=self.n_latents)

        # Compute loss for all (latent, class) pairs at once
        loss = -n_pos_expanded * torch.log(mu_0) - n_neg_expanded * torch.log(1 - mu_0)

        # Vectorized correction for non-zero entries
        if col_indices.numel() > 0:
            # Get row indices for each non-zero value
            row_indices = torch.repeat_interleave(
                torch.arange(n_samples, device=self.device),
                crow_indices[1:] - crow_indices[:-1],
            )

            # Get the y values for each non-zero entry
            y_nz = y[row_indices]  # [nnz, n_classes]

            # Compute mu for all non-zero entries at once
            x_values_expanded = einops.rearrange(values, "nnz -> nnz 1")
            eta = (
                self.intercept_[col_indices]
                + self.coef_[col_indices] * x_values_expanded
            )
            mu = torch.sigmoid(eta)
            mu = torch.clamp(mu, self.eps, 1 - self.eps)

            # Compute corrections for each non-zero entry
            # Remove zero contribution and add actual contribution
            zero_contrib = y_nz * torch.log(mu_0[col_indices]) + (1 - y_nz) * torch.log(
                1 - mu_0[col_indices]
            )
            actual_contrib = y_nz * torch.log(mu) + (1 - y_nz) * torch.log(1 - mu)

            # Accumulate corrections
            loss.index_add_(0, col_indices, zero_contrib - actual_contrib)

        return loss / n_samples  # Return mean NLL

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

    @torch.no_grad()
    def loss_matrix_with_aux(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Bool[Tensor, "n_samples n_classes"],
    ) -> tuple[Float[Tensor, "n_latents n_classes"], dict]:
        """Returns the NLL loss matrix and additional metadata needed to construct the parquet file."""
        sklearn.utils.validation.check_is_fitted(self, "intercept_")
        sklearn.utils.validation.check_is_fitted(self, "coef_")

        # Move data to device and ensure correct types
        x = x.to(self.device)
        y = y.to(self.device).float()

        loss = self._compute_loss(x, y)

        # Compute auxiliary metrics
        n_samples = x.shape[0]

        # Count nnz per latent
        crow_indices = x.crow_indices()
        col_indices = x.col_indices()
        nnz_per_latent = torch.bincount(col_indices, minlength=self.n_latents)

        # Compute accuracy (at threshold 0.5)
        # For zero entries
        mu_0 = torch.sigmoid(self.intercept_)
        pred_0 = (mu_0 > 0.5).float()

        # Count correct predictions assuming all entries are zero
        # pred_0 shape: [n_latents, n_classes]
        # y shape: [n_samples, n_classes]
        # Vectorized across all classes at once
        pred_0_expanded = einops.rearrange(
            pred_0, "l c -> l 1 c"
        )  # [n_latents, 1, n_classes]
        y_expanded = einops.rearrange(y, "s c -> 1 s c")  # [1, n_samples, n_classes]
        # Compare and sum over samples
        acc = (
            (pred_0_expanded == y_expanded).float().sum(dim=1)
        )  # [n_latents, n_classes]

        # Vectorized correction for non-zero entries
        if col_indices.numel() > 0:
            # Get row indices for each non-zero value
            row_indices = torch.repeat_interleave(
                torch.arange(n_samples, device=self.device),
                crow_indices[1:] - crow_indices[:-1],
            )

            # Get the y values for each non-zero entry
            y_nz = y[row_indices]  # [nnz, n_classes]

            # Compute predictions for non-zero entries
            x_values_expanded = einops.rearrange(x.values(), "nnz -> nnz 1")
            eta = (
                self.intercept_[col_indices]
                + self.coef_[col_indices] * x_values_expanded
            )
            mu = torch.sigmoid(eta)
            pred_nz = (mu > 0.5).float()

            # For each non-zero entry, adjust the accuracy
            # Remove the zero prediction and add the actual prediction
            pred_0_nz = pred_0[col_indices]  # [nnz, n_classes]

            # Correction: -1 if zero was correct but nonzero is wrong, +1 if zero was wrong but nonzero is correct
            zero_correct = (pred_0_nz == y_nz).float()
            nz_correct = (pred_nz == y_nz).float()
            correction = nz_correct - zero_correct

            # Accumulate corrections
            acc.index_add_(0, col_indices, correction)

        acc = acc / n_samples  # Convert to accuracy

        aux = {
            "accuracy": acc,
            "nnz_per_latent": nnz_per_latent,
            "n_samples": n_samples,
            "coef": self.coef_,
            "intercept": self.intercept_,
        }

        return loss, aux
