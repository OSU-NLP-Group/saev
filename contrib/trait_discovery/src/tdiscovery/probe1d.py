r"""Per-feature x per-class 1D logistic probes on SAE activations.

Some feedback:

Short answer: as written, this will OOM long before it gets a chance to be "slow." The killer is every place you expand from nnz → (nnz, n\_classes). With 22k×256 = 5.632M samples and L0=400, you have \~2.2528B nonzeros. Anything of shape (nnz, 151) is astronomically large.

Here’s a surgical pass through `fit()` with concrete sizes and fixes.

# What blows up (and why)

* `y = y.to(self.device)`
  If `y` is float32, that’s 5.632M×151×4B ≈ 3.4 GB on GPU. This is already chunky. Prefer processing **one class (or a tiny slab of classes)** at a time so you only keep 5.6M labels in memory (≈22 MB as uint8/bool).

* `row_indices = torch.repeat_interleave(...)`
  Length = nnz = 2,252,800,000.
  int32: \~9.0 GB; int64: \~18 GB. This alone can eat your remaining memory. You must avoid materializing full `row_indices`—compute it **per row-block** (microbatch over rows), so the repeated\_interleave vector is at most tens of millions long at any moment.

* `y_nz = y[row_indices]  # [nnz, n_classes]`
  This is the show-stopper. Shape \~ (2.25B, 151).
  Even fp16: 2.25e9×151×2B ≈ 680 GB. fp32 is \~1.36 TB. Impossible. Fix: loop over **classes** (or tiny class slabs, e.g., 4–8) and **event microbatches** (row-blocks), so at any instant you only touch (E\_chunk, C\_chunk).

* `eta = self.intercept_[col_indices] + self.coef_[col_indices] * x_values_expanded`
  `self.intercept_[col_indices]` is also (nnz, n\_classes). Same explosion as above. Gather (L, C\_chunk) once, then index with (E\_chunk,) to get (E\_chunk, C\_chunk) only.

* `mu`, `s`, `residual` allocated as (nnz, n\_classes)
  Same problem—compute in **chunks** and **don’t store** these big intermediates at all; compute-and-accumulate.

Everything else (the (L, C) accumulators, params, counts) is tiny by comparison (a few hundred MB total).

# How to restructure so it runs

1. **Loop over classes (or a small slab):**

```python
for c0 in range(0, n_classes, Cb):  # e.g., Cb = 4 or 8
    b = self.intercept_[:, c0:c0+Cb]    # (L, Cb)
    w = self.coef_[:, c0:c0+Cb]         # (L, Cb)
    mu0 = torch.sigmoid(b); s0 = mu0*(1-mu0)
    # init accumulators M_nz, G1, S0, S1, S2 as (L, Cb) zeros
```

2. **Stream rows in blocks** to avoid a mega `row_indices`:

```python
for r0 in range(0, n_samples, Rb):     # e.g., Rb = 1–8 million rows
    r1 = min(r0+Rb, n_samples)
    # From CSR: [start:end) spans all nnz in these rows
    start = crow_indices[r0].item()
    end   = crow_indices[r1].item()
    cols  = col_indices[start:end]      # (E_chunk,)
    vals  = values[start:end]           # (E_chunk,)
    # Build a TEMPORARY row_indices only for [r0:r1]
    # lengths_per_row = crow[r0+1:r1+1] - crow[r0:r1]
    # row_idx_chunk = repeat_interleave(lengths_per_row)  # length = E_chunk (fits)
    y_chunk = y[r0:r1, c0:c0+Cb]        # (Rb, Cb)
    y_nz    = y_chunk[row_idx_chunk]    # (E_chunk, Cb)  <-- manageable
    # Gather params by latent for this event chunk
    b_cols = b[cols]                    # (E_chunk, Cb)
    w_cols = w[cols]                    # (E_chunk, Cb)
    # Compute and ACCUMULATE on the fly (don’t keep mu/s around):
    eta = b_cols + w_cols * vals.view(-1,1)
    mu  = torch.sigmoid(eta)
    s   = mu * (1 - mu)
    # segmented reduction by latent index:
    M_nz.index_add_(0, cols, mu)
    G1.index_add_(0, cols, (mu - y_nz) * vals.view(-1,1))
    S0.index_add_(0, cols, s)
    S1.index_add_(0, cols, s * vals.view(-1,1))
    S2.index_add_(0, cols, s * (vals**2).view(-1,1))
```

3. **Zero-part + Newton update for the class slab:**

```python
nnz_per_latent = torch.bincount(col_indices, minlength=L)  # (L,) – once globally
Z = (n_samples - nnz_per_latent).view(L,1).float()

G0 = M_nz + Z * mu0 - pi[c0:c0+Cb].view(1, Cb)
S0 = S0 + Z * s0
# ridge: G1 += lambda*w ; S2 += lambda (true L2). Add small damping to both diag entries if needed.
```

Then compute `db, dw` from the 2×2 per-(latent, class) systems and update `b, w`. Write back to `self.intercept_[:, c0:c0+Cb]`, `self.coef_[:, c0:c0+Cb]`. Repeat until convergence.

4. **Loss**: compute in the same class/event microbatch loops using the “zero baseline + nonzero correction” you wrote, but **accumulate directly** into the (L, Cb) loss matrix via `index_add_`—don’t allocate (nnz, Cb).

# Will it be too slow?

Once you avoid the (nnz, n\_classes) tensors, the bottleneck becomes **scatter/index\_add over \~2.25B events per class per Newton iteration**.

Very rough order-of-magnitude:

* Events: E = 2.2528e9.
* Classes: C = 151.
* Event–class operations per iteration: E×C ≈ 3.40e11.
* Per event–class you do \~10–20 flops and a handful of memory ops plus a scatter-add. This is **memory/atomics bound**, not compute bound.

Empirically on A100s:

* Coalesced `index_add_` (after sorting by `cols`) can push **hundreds of millions** of updates/sec; unsorted atomics can be far worse.
* If you hit \~100M event–class updates/sec, that’s \~3.4e3 s ≈ **57 min per Newton iteration**.
* With 6–10 Newton steps, you’re in the **6–10 hour** ballpark per SAE, which meets your “<12 hours” target—*if* you implement class+row microbatching, keep dtypes lean (bf16/half for activations), and reduce atomic contention (see below).

This is not a guarantee—throughput depends a lot on:

* **Atomics contention**: many events hitting the same latent line. Sorting `cols` within each event chunk before `index_add_` (or using `torch_scatter`/segmented reductions) often gives a **2–5×** boost.
* **Chunk sizes**: tune `Rb` (rows per block) and `Cb` (classes per slab) so your temporary buffers stay <1–2 GB and your kernels stay saturated.
* **Dtypes**: keep `vals` (activations) in fp16/bf16 and `y` in uint8/bool; keep accumulators in fp32.

# Minimal changes you should make now

* Do **not** allocate `row_indices` for the whole matrix; build it per row-block.
* Do **not** allocate anything with shape `(nnz, n_classes)`. Always tile classes and events.
* Move `y` to GPU **per class slab** (or stream from pinned host memory).
* Consider **sorting** `cols` inside each chunk (`torch.sort`) and apply the same permutation to `vals` and `row_idx_chunk`; then use `index_add_` or `segment_reduce` to cut atomics.
* Keep `values` and `mu` computations in **bf16/fp16**; keep accumulators in fp32.
* In `_compute_loss`, reuse the same class/event microbatching; don’t rebuild huge arrays.

If you want, I can rewrite your `fit()` skeleton into a tiled version (classes × row-blocks) with placeholders for chunk sizes and a micro-benchmark harness so you can time E×C throughput on one A100 and extrapolate.
"""

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
        class_chunk_size: int = 8,
        event_chunk_size: int = 1_000_000,
    ):
        self.n_latents = n_latents
        self.n_classes = n_classes
        self.tol = tol
        self.device = device
        self.n_iter = n_iter
        self.ridge = ridge  # L2 regularization strength
        self.class_chunk_size = class_chunk_size
        self.event_chunk_size = event_chunk_size
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
        dd = dict(dtype=torch.float32, device=self.device)

        # Move sparse matrix to device
        x = x.to(self.device)

        # Initialize parameters (full size)
        self.coef_ = torch.zeros((self.n_latents, self.n_classes), **dd)
        self.intercept_ = torch.zeros((self.n_latents, self.n_classes), **dd)

        # Calculate pi (total positives per class) on CPU to save memory
        # We'll move class slabs to device as needed
        pi = y.sum(dim=0).to(torch.float32)  # [n_classes]
        prevalence = pi / n_samples
        prevalence = torch.clamp(prevalence, self.eps, 1 - self.eps)

        # Get CSR components once
        crow_indices = x.crow_indices()
        col_indices = x.col_indices()
        values = x.values()

        # Count nnz per latent once (used for zero contributions)
        nnz_per_latent = torch.bincount(col_indices, minlength=self.n_latents)
        n_zeros_per_latent = n_samples - nnz_per_latent  # [n_latents]

        # Process classes in chunks
        for c0 in range(0, self.n_classes, self.class_chunk_size):
            c1 = min(c0 + self.class_chunk_size, self.n_classes)
            Cb = c1 - c0

            self.logger.debug(f"Processing classes {c0}:{c1} ({Cb} classes)")

            # Initialize params for this class slab
            self.intercept_[:, c0:c1] = einops.repeat(
                torch.logit(prevalence[c0:c1]).to(self.device),
                "c -> l c",
                l=self.n_latents,
            )
            # coef already initialized to zero

            # Move class slab of labels to device
            y_slab = y[:, c0:c1].to(self.device)  # [n_samples, Cb]
            pi_slab = pi[c0:c1].to(self.device)  # [Cb]

            prev_loss = torch.full((self.n_latents, Cb), torch.inf, **dd)

            # Newton iterations for this class slab
            for it in range(self.n_iter):
                # Compute mu_0 and s_0 for zero entries
                mu_0 = torch.sigmoid(self.intercept_[:, c0:c1])  # [n_latents, Cb]
                s_0 = mu_0 * (1 - mu_0)

                # Initialize accumulators for this iteration
                m_nz = torch.zeros((self.n_latents, Cb), **dd)
                g1 = torch.zeros((self.n_latents, Cb), **dd)
                s0_acc = torch.zeros((self.n_latents, Cb), **dd)
                s1 = torch.zeros((self.n_latents, Cb), **dd)
                s2 = torch.zeros((self.n_latents, Cb), **dd)

                # Stream over event/row blocks
                for r0 in range(0, n_samples, self.event_chunk_size):
                    r1 = min(r0 + self.event_chunk_size, n_samples)

                    # Get nnz range for this row block
                    start = crow_indices[r0].item()
                    end = crow_indices[r1].item()

                    if start == end:
                        # No nonzeros in this block
                        continue

                    # Get CSR data for this block
                    cols_chunk = col_indices[start:end]  # [E_chunk]
                    vals_chunk = values[start:end]  # [E_chunk]

                    # Build temporary row_indices for [r0:r1]
                    lengths_per_row = (
                        crow_indices[r0 + 1 : r1 + 1] - crow_indices[r0:r1]
                    )
                    row_idx_chunk = torch.repeat_interleave(
                        torch.arange(r1 - r0, device=self.device), lengths_per_row
                    )  # [E_chunk], values in [0, r1-r0)

                    # Get labels for this chunk
                    y_chunk = y_slab[r0:r1]  # [Rb, Cb]
                    y_nz = y_chunk[row_idx_chunk]  # [E_chunk, Cb]

                    # Gather params for this chunk's latents
                    b_cols = self.intercept_[cols_chunk][:, c0:c1]  # [E_chunk, Cb]
                    w_cols = self.coef_[cols_chunk][:, c0:c1]  # [E_chunk, Cb]

                    # Compute eta, mu, s for nonzeros
                    vals_expanded = vals_chunk.view(-1, 1)  # [E_chunk, 1]
                    eta = b_cols + w_cols * vals_expanded
                    mu = torch.sigmoid(eta)
                    mu = torch.clamp(mu, self.eps, 1 - self.eps)
                    s = mu * (1 - mu)

                    # Accumulate statistics
                    m_nz.index_add_(0, cols_chunk, mu)
                    g1.index_add_(0, cols_chunk, (mu - y_nz) * vals_expanded)
                    s0_acc.index_add_(0, cols_chunk, s)
                    s1.index_add_(0, cols_chunk, s * vals_expanded)
                    s2.index_add_(0, cols_chunk, s * (vals_expanded**2))

                # Add zero contributions
                n_zeros_expanded = n_zeros_per_latent.float().view(-1, 1)  # [L, 1]
                g0 = m_nz + n_zeros_expanded * mu_0 - pi_slab.view(1, -1)

                # Add L2 regularization
                g1 = g1 + self.ridge * self.coef_[:, c0:c1]

                # S0 includes zero contributions and ridge
                s0_total = s0_acc + n_zeros_expanded * s_0 + self.ridge
                s2 = s2 + self.ridge

                # Calculate Newton updates (2x2 system per (latent, class))
                det_h = s0_total * s2 - s1 * s1
                det_h = torch.where(
                    torch.abs(det_h) < 1e-10, torch.ones_like(det_h) * 1e-10, det_h
                )

                db = (s2 * g0 - s1 * g1) / det_h
                dw = (-s1 * g0 + s0_total * g1) / det_h

                # Apply updates to this class slab
                self.intercept_[:, c0:c1] -= db
                self.coef_[:, c0:c1] -= dw

                # Check convergence for this class slab
                loss_slab = self._compute_loss_slab(
                    x,
                    y_slab,
                    c0,
                    c1,
                    crow_indices,
                    col_indices,
                    values,
                    n_samples,
                    pi_slab,
                )

                loss_change = torch.abs(prev_loss - loss_slab)
                if torch.all(loss_change < self.tol):
                    self.logger.debug(f"Classes {c0}:{c1} converged at iteration {it}")
                    break

                prev_loss = loss_slab.clone()
            else:
                self.logger.debug(
                    f"Classes {c0}:{c1} did not converge after {self.n_iter} iterations"
                )

            # Free memory
            del y_slab, pi_slab

    def _compute_loss_slab(
        self,
        x: Float[Tensor, "n_samples n_latents"],
        y_slab: Tensor,  # Float[Tensor, "n_samples Cb"]
        c0: int,
        c1: int,
        crow_indices: Tensor,
        col_indices: Tensor,
        values: Tensor,
        n_samples: int,
        pi_slab: Tensor,  # Float[Tensor, "Cb"]
    ) -> Tensor:  # Float[Tensor, "n_latents Cb"]
        """Compute loss for a class slab using event chunking."""
        Cb = c1 - c0
        dd = dict(dtype=torch.float32, device=self.device)
        loss = torch.zeros((self.n_latents, Cb), **dd)

        # Compute mu_0 for zero entries
        mu_0 = torch.sigmoid(self.intercept_[:, c0:c1])
        mu_0 = torch.clamp(mu_0, self.eps, 1 - self.eps)

        # Count positives and negatives per class
        n_pos = pi_slab  # [Cb]
        n_neg = n_samples - n_pos  # [Cb]

        # Initialize with all-zeros contribution
        loss = -n_pos.view(1, -1) * torch.log(mu_0) - n_neg.view(1, -1) * torch.log(
            1 - mu_0
        )

        # Stream over events to compute corrections
        for r0 in range(0, n_samples, self.event_chunk_size):
            r1 = min(r0 + self.event_chunk_size, n_samples)

            start = crow_indices[r0].item()
            end = crow_indices[r1].item()

            if start == end:
                continue

            cols_chunk = col_indices[start:end]
            vals_chunk = values[start:end]

            lengths_per_row = crow_indices[r0 + 1 : r1 + 1] - crow_indices[r0:r1]
            row_idx_chunk = torch.repeat_interleave(
                torch.arange(r1 - r0, device=self.device), lengths_per_row
            )

            y_chunk = y_slab[r0:r1]
            y_nz = y_chunk[row_idx_chunk]  # [E_chunk, Cb]

            # Compute mu for nonzeros
            vals_expanded = vals_chunk.view(-1, 1)
            b_cols = self.intercept_[cols_chunk][:, c0:c1]
            w_cols = self.coef_[cols_chunk][:, c0:c1]
            eta = b_cols + w_cols * vals_expanded
            mu = torch.sigmoid(eta)
            mu = torch.clamp(mu, self.eps, 1 - self.eps)

            # Compute corrections
            mu_0_nz = mu_0[cols_chunk]  # [E_chunk, Cb]
            zero_contrib = y_nz * torch.log(mu_0_nz) + (1 - y_nz) * torch.log(
                1 - mu_0_nz
            )
            actual_contrib = y_nz * torch.log(mu) + (1 - y_nz) * torch.log(1 - mu)

            loss.index_add_(0, cols_chunk, zero_contrib - actual_contrib)

        return loss / n_samples

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
        crow_indices = x.crow_indices()
        col_indices = x.col_indices()
        values = x.values()

        # Compute pi on CPU
        pi = y.sum(dim=0).to(torch.float32)  # [n_classes]

        # Process classes in chunks
        for c0 in range(0, self.n_classes, self.class_chunk_size):
            c1 = min(c0 + self.class_chunk_size, self.n_classes)

            # Move class slab to device
            y_slab = y[:, c0:c1].to(self.device)
            pi_slab = pi[c0:c1].to(self.device)

            # Compute loss for this slab
            loss[:, c0:c1] = self._compute_loss_slab(
                x, y_slab, c0, c1, crow_indices, col_indices, values, n_samples, pi_slab
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
        y_slab: Tensor,  # Float[Tensor, "n_samples Cb"]
        c0: int,
        c1: int,
        crow_indices: Tensor,
        col_indices: Tensor,
        values: Tensor,
        n_samples: int,
    ) -> Tensor:  # Float[Tensor, "n_latents Cb"]
        """Compute accuracy for a class slab using event chunking."""

        # Predictions for zero entries
        mu_0 = torch.sigmoid(self.intercept_[:, c0:c1])
        pred_0 = (mu_0 > 0.5).float()  # [n_latents, Cb]

        # Initialize accuracy by counting correct zero predictions
        # For each (latent, class) pair, count how many samples match pred_0
        pred_0_expanded = pred_0.unsqueeze(1)  # [n_latents, 1, Cb]
        y_expanded = y_slab.unsqueeze(0)  # [1, n_samples, Cb]
        acc = (pred_0_expanded == y_expanded).float().sum(dim=1)  # [n_latents, Cb]

        # Stream over events to compute corrections for nonzeros
        for r0 in range(0, n_samples, self.event_chunk_size):
            r1 = min(r0 + self.event_chunk_size, n_samples)

            start = crow_indices[r0].item()
            end = crow_indices[r1].item()

            if start == end:
                continue

            cols_chunk = col_indices[start:end]
            vals_chunk = values[start:end]

            lengths_per_row = crow_indices[r0 + 1 : r1 + 1] - crow_indices[r0:r1]
            row_idx_chunk = torch.repeat_interleave(
                torch.arange(r1 - r0, device=self.device), lengths_per_row
            )

            y_chunk = y_slab[r0:r1]
            y_nz = y_chunk[row_idx_chunk]  # [E_chunk, Cb]

            # Compute predictions for nonzeros
            vals_expanded = vals_chunk.view(-1, 1)
            b_cols = self.intercept_[cols_chunk][:, c0:c1]
            w_cols = self.coef_[cols_chunk][:, c0:c1]
            eta = b_cols + w_cols * vals_expanded
            mu = torch.sigmoid(eta)
            pred_nz = (mu > 0.5).float()

            # Compute correction: swap zero prediction for actual prediction
            pred_0_nz = pred_0[cols_chunk]  # [E_chunk, Cb]
            zero_correct = (pred_0_nz == y_nz).float()
            nz_correct = (pred_nz == y_nz).float()
            correction = nz_correct - zero_correct

            acc.index_add_(0, cols_chunk, correction)

        return acc / n_samples

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

        # Compute accuracy using chunked implementation
        crow_indices = x.crow_indices()
        values = x.values()

        acc = torch.zeros(
            (self.n_latents, self.n_classes), dtype=torch.float32, device=self.device
        )

        for c0 in range(0, self.n_classes, self.class_chunk_size):
            c1 = min(c0 + self.class_chunk_size, self.n_classes)

            y_slab = y[:, c0:c1].to(self.device).float()

            acc[:, c0:c1] = self._compute_accuracy_slab(
                x, y_slab, c0, c1, crow_indices, col_indices, values, n_samples
            )

            del y_slab

        aux = {
            "accuracy": acc,
            "nnz_per_latent": nnz_per_latent,
            "n_samples": n_samples,
            "coef": self.coef_,
            "intercept": self.intercept_,
        }

        return loss, aux
