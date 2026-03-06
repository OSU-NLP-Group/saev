# Semi-NMF Baseline Specification

## Overview

Add Semi-NMF (Semi-Non-negative Matrix Factorization) as a third baseline method alongside K-Means and PCA in `baselines.py`. Semi-NMF factorizes activations A into codes Z and dictionary D where A ~= Z @ D, with the constraint that Z >= 0 (non-negative codes) while D can be positive or negative.

Like PCA, Semi-NMF is a **low-rank baseline** where the "sparsity" is controlled by the dictionary size k (number of concepts), not by sparsifying the codes. Each patch is represented by k dense coefficients.

## Background

### Why Mini-batch?

Standard Semi-NMF requires Z (n_samples x k) to exist during fit. For 100M patches x 1024 concepts at fp32, that's ~400GB - infeasible for GPU memory. We use mini-batch Semi-NMF instead, which processes data in chunks and updates D incrementally after each batch.

This approach is adapted from [scikit-learn's MiniBatchNMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html), modified for Semi-NMF (where only Z is constrained to be non-negative, not D). We perform one Z solve per batch and update D every N batches (default 10), allowing D to evolve as we see more data.

### Semi-NMF Algorithm

Solves: min ||A - Z @ D||_F^2, subject to Z >= 0.

**Notation:**
- A: (n_samples, n_features) - input activations (ViT patch embeddings). Can be positive or negative.
- Z: (n_samples, n_concepts) - codes/coefficients. **Non-negative** (Z >= 0).
- D: (n_concepts, n_features) - learned dictionary. No sign constraint.

**Multiplicative Update Rule (Ding et al., 2008):**

For updating Z (with D fixed):
```
Z = Z * sqrt((pos(A @ D.T) + Z @ neg(D @ D.T)) / (neg(A @ D.T) + Z @ pos(D @ D.T) + eps))
```

Where pos(X) = (|X| + X) / 2 and neg(X) = (|X| - X) / 2.

For updating D (with Z fixed), solve the regularized least squares problem:
```
(Z^T Z + λI) @ D = Z^T A
D = solve(ZtZ + λI, ZtA)
```

**Note:** The epsilon (default 1e-8) is added to the denominator only, not inside the sqrt. This prevents division by zero without biasing the update. Ridge regularization (λ=1e-6) stabilizes the D update for potentially ill-conditioned Z^T Z.

**Online D Update:** For mini-batch learning, we use exponential moving average (EMA) accumulators for Z^T Z and Z^T A, following sklearn's MiniBatchNMF. The forget_factor (default 0.7) downweights stale statistics from batches where Z was computed with an older D:
```
ZtZ_acc = forget * ZtZ_acc + (1 - forget) * (Z^T @ Z)
ZtA_acc = forget * ZtA_acc + (1 - forget) * (Z^T @ A)
```
D is updated every N batches (default 10) using the averaged accumulators. Ridge regularization is applied to the averaged statistics, maintaining stable regularization strength regardless of sample count.

## Requirements

### 1. MiniBatchSemiNMF Class

Add a `MiniBatchSemiNMF` class to `baselines.py` that processes data in batches.

**Class structure:**
```python
class MiniBatchSemiNMF(torch.nn.Module):
    method = "semi-nmf"

    def __init__(self, n_concepts: int, device: str = "cuda", z_iters: int = 10, encode_iters: int = 300, batch_size: int = 16384, ridge: float = 1e-6, eps: float = 1e-8, forget_factor: float = 0.7, d_update_every: int = 10):
        ...
        self.D_: Tensor | None = None  # (n_concepts, n_features)
        self.n_features_in_: int | None = None
        self.n_samples_seen_: int = 0
        self.ZtZ_acc_: Tensor | None = None  # EMA accumulator
        self.ZtA_acc_: Tensor | None = None  # EMA accumulator

    def partial_fit(self, batch: Tensor) -> Self:
        # For each batch:
        # 1. Warm-start Z using least-squares approximation
        # 2. Refine Z with z_iters multiplicative updates
        # 3. Update EMA accumulators: ZtZ_acc = f * ZtZ_acc + (1-f) * Z^T @ Z
        # 4. Every d_update_every batches: update D using averaged accumulators
        ...

    def transform(self, batch: Tensor) -> Tensor:
        # Iterative encode with fixed D
        ...
```

**Key implementation details:**
- Use **fp32 precision** throughout (not bf16) to avoid multiplicative update instability where zeros "stick"
- **Warm-start Z** using ridge-regularized least-squares: `Z = (A @ D.T @ inv(D @ D.T + eps*I)).clamp_min(eps)`. Clamp to eps (not zero) to avoid zeros that multiplicative updates cannot revive.
- **EMA accumulators** with forget_factor to downweight stale Z statistics
- **Cache** `D @ D.T`, `pos(D @ D.T)`, `neg(D @ D.T)` and recompute only after each D update (not every batch)
- **Update D every N batches** (default 10) to reduce solve overhead. Final D update at end if not aligned.
- Use `torch.linalg.solve` with ridge regularization for D update
- Set global random seed once at training/inference start for reproducibility
- Log reconstruction error periodically to stderr

### 2. Training Configuration

Extend `BaselineMethod` literal:
```python
BaselineMethod = tp.Literal["kmeans", "pca", "semi-nmf"]
```

**Training parameters:**
- `z_iters`: Multiplicative update iterations per batch during training (default: 10). With warm-start, 10 is sufficient.
- `encode_iters`: Multiplicative update iterations during inference/transform (default: 300).
- `batch_size`: Patches per batch (default: 16384)
- `ridge`: Regularization for D update (default: 1e-6). Applied to averaged accumulators.
- `eps`: Small constant for numerical stability in multiplicative update (default: 1e-8)
- `forget_factor`: EMA decay for accumulators (default: 0.7). Downweights stale Z statistics.
- `d_update_every`: Update D every N batches (default: 10). 625 D updates total for 6,250 batches.

**Training data:**
- Use shuffled dataloader (uniform random sampling)
- Sample size: **100M patches** total (single pass)
- Batch size: **16K patches** (6,250 batches total)
- Dictionary size sweep: **k = [1, 4, 16, 64, 256, 1024]** (matching PCA)
- Total: **6 configurations** per layer

**Validation:**
- Use IN1K val split for reconstruction loss at end of training

**Resource allocation:**
- Use nextgen nodes (40GB VRAM)
- fp32 precision

### 3. Training Metrics

Log to W&B and stderr:
- `train/recon_mse`: ||A - Z @ D||^2 / n_samples (computed periodically during training)
- `train/nmse`: ||A - Z @ D||^2 / ||A - mean(A)||^2 (normalized)
- `train/n_samples`: Total samples processed
- `val/recon_mse`: Reconstruction error on IN1K val split (at end of training)

### 4. Inference (Encode)

**Streaming encode:**
- Use fixed batch size (4096 patches per batch)
- Iterative encode: run multiplicative update for Z with fixed D (`encode_iters` iterations, default 300)
- Precompute `D @ D.T` and `pos(D @ D.T)`, `neg(D @ D.T)` once before encode loop

**Storage:**
- Store codes as dense arrays (k is small: 1-1024)
- No TopK sparsification - the low-rank k controls dimensionality
- For API consistency with probe1d pipeline, wrap in same interface as PCA

### 5. Serialization

Save only the dictionary D matrix (like PCA saves components):
```python
{
    "D": model.D_.cpu(),  # (n_concepts, n_features)
    "n_features_in": model.n_features_in_,
    "n_samples_seen": model.n_samples_seen_,
}
```

### 6. Sweep Configuration

Add Semi-NMF to `baselines_train.py`:
```python
# Match PCA sweep: low-rank baseline
methods = [
    ("pca", [1, 4, 16, 64, 256, 1024]),
    ("semi-nmf", [1, 4, 16, 64, 256, 1024]),
]

for layer in [13, 15, 17, 19, 21, 23]:
    for method, ks in methods:
        for k in ks:
            cfgs.append({
                "method": method,
                "k": k,
                "train_data": {"shards": ..., "layer": layer},
                ...
            })
```

### 7. Tests

Add `test_semi_nmf.py` with:
- Basic fit on small synthetic data
- Verify Z is non-negative after fit and transform
- Verify D shape matches (n_concepts, n_features)
- Roundtrip serialize/deserialize test
- Verify encode produces same output with fixed seed
- Verify mini-batch produces similar D to full-batch on small data

## Non-Goals

- Full-batch fit (infeasible for large data)
- NNDSVD initialization (use random init)
- bf16 precision (numerical stability concerns with multiplicative updates)
- TopK sparsification (use low-rank k instead, like PCA)

## Implementation Notes

### Mini-batch Algorithm

```python
def fit(self, dataloader, n_concepts: int, n_features: int):
    # Initialize D randomly
    D = torch.randn(n_concepts, n_features)
    ridge = 1e-6
    eps = 1e-8
    forget = 0.7
    d_update_every = 10

    # EMA accumulators for D update
    ZtZ_acc = torch.zeros(n_concepts, n_concepts)
    ZtA_acc = torch.zeros(n_concepts, n_features)

    # Cache D-related terms (recomputed only after D updates)
    DDT = D @ D.T
    DDT_pos = pos_part(DDT)
    DDT_neg = neg_part(DDT)
    DDT_reg_inv = torch.linalg.inv(DDT + eps * torch.eye(n_concepts))

    n_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        A_batch = batch["act"]  # (batch_size, n_features)
        n_batches += 1

        # Warm-start Z using ridge-regularized least-squares
        Z = (A_batch @ D.T @ DDT_reg_inv).clamp_min(eps)

        # Refine Z with multiplicative updates
        ATD = A_batch @ D.T
        ATD_pos = pos_part(ATD)
        ATD_neg = neg_part(ATD)

        for _ in range(z_iters):
            numerator = ATD_pos + Z @ DDT_neg
            denominator = ATD_neg + Z @ DDT_pos + eps
            Z = Z * torch.sqrt(numerator / denominator)

        # Update EMA accumulators (downweight stale statistics)
        ZtZ_batch = Z.T @ Z
        ZtA_batch = Z.T @ A_batch
        ZtZ_acc = forget * ZtZ_acc + (1 - forget) * ZtZ_batch
        ZtA_acc = forget * ZtA_acc + (1 - forget) * ZtA_batch

        # Update D every N batches using averaged accumulators
        if (batch_idx + 1) % d_update_every == 0:
            reg = ZtZ_acc + ridge * torch.eye(n_concepts)
            D = torch.linalg.solve(reg, ZtA_acc)
            # Recompute cached D-related terms
            DDT = D @ D.T
            DDT_pos = pos_part(DDT)
            DDT_neg = neg_part(DDT)
            DDT_reg_inv = torch.linalg.inv(DDT + eps * torch.eye(n_concepts))

    # Final D update if not aligned with d_update_every
    if n_batches % d_update_every != 0:
        reg = ZtZ_acc + ridge * torch.eye(n_concepts)
        D = torch.linalg.solve(reg, ZtA_acc)
```

### Helper Functions

```python
def pos_part(X: Tensor) -> Tensor:
    return (X.abs() + X) / 2

def neg_part(X: Tensor) -> Tensor:
    return (X.abs() - X) / 2

def warm_start_z(A: Tensor, D: Tensor, eps: float = 1e-8) -> Tensor:
    """Initialize Z using ridge-regularized least-squares, clamped to positive.

    Uses (D @ D.T + eps*I)^-1 for numerical stability. Clamps to eps (not zero)
    to avoid zeros that multiplicative updates cannot revive.
    """
    DDT = D @ D.T
    DDT_reg = DDT + eps * torch.eye(D.shape[0], device=D.device)
    Z = A @ D.T @ torch.linalg.inv(DDT_reg)
    Z = Z.clamp_min(eps)  # Avoid exact zeros
    return Z
```

## References

- Ding, C., Li, T., & Jordan, M. I. (2008). "Convex and Semi-Nonnegative Matrix Factorizations." IEEE TPAMI.
- Serizel, R., Bisot, V., Essid, S., & Richard, G. (2016). "Mini-batch stochastic approaches for accelerated multiplicative updates in nonnegative matrix factorisation with beta-divergence." MLSP.
- [scikit-learn MiniBatchNMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html)
