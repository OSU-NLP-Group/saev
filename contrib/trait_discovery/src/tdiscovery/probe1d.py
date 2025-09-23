"""Per-feature x per-class 1D logistic probes on SAE activations."""

import logging

import beartype
import sklearn.base
import torch
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype.beartype)
class Sparse1DProbe(sklearn.base.BaseEstimator):
    """Newton-Raphson optimizer for 1D logistic regression with L2 regularization. `fit(x, y)` streams sparse x and optimizes (b, w) for every (latent, class) pair. Results are exposed as attributes and helper methods."""

    def __init__(
        self,
        *,
        n_latents: int,
        n_classes: int,
        tol: float = 1e-6,
        device: str = "cuda",
        n_iter: int = 30,
    ):
        self.n_latents = n_latents
        self.n_classes = n_classes
        self.tol = tol
        self.device = device
        self.n_iter = n_iter
        self.logger = logging.getLogger("sparse1d")

    def fit(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Bool[Tensor, "n_samples n_classes"],
    ):
        assert x.layout == torch.sparse_csr

        shape = (self.n_latents, self.n_classes)
        dd = dict(dtype=torch.float32, device=self.device)

        self.coef_ = torch.zeros(shape, **dd)
        self.intercept_ = torch.zeros(shape, **dd)
        self.loss_ = torch.full(shape, torch.inf, **dd)

        for it in range(self.n_iter):
            # Reset gradient and Hessian accumulators for new epoch.
            self.g0_ = torch.zeros(shape, **dd)
            self.g1_ = torch.zeros_like(self.g0_)

            self.s0_ = torch.zeros_like(self.g0_)
            self.s1_ = torch.zeros_like(self.g0_)
            self.s2_ = torch.zeros_like(self.g0_)

            # Accumulate
            # TODO
            break
            raise NotImplementedError()

            # Calculate updates to b and w.
            db = None
            dw = None

            # Apply updates
            self.b_ -= db
            self.w_ -= dw

    def loss_matrix(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Bool[Tensor, "n_samples n_classes"],
    ) -> Float[Tensor, "n_latents n_classes"]:
        """Returns the NLL loss matrix. Cheap to compute because we just use intercept_ and coef_ to recalculate loss."""
        sklearn.utils.validation.check_is_fitted(self, "intercept_")
        sklearn.utils.validation.check_is_fitted(self, "coef_")
        raise NotImplementedError()

    def loss_matrix_with_aux(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y: Bool[Tensor, "n_samples n_classes"],
    ) -> tuple[Float[Tensor, "n_latents n_classes"], object]:
        """Returns the NLL loss matrix and additional metadata needed to construct the parquet file."""
        sklearn.utils.validation.check_is_fitted(self, "intercept_")
        sklearn.utils.validation.check_is_fitted(self, "coef_")
        raise NotImplementedError()
