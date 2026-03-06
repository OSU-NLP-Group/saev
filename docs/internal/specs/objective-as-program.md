# Objective-As-Program for SAEs

## Summary

- Problem: Matryoshka SAEs currently require a special SAE subclass and a special Objective, plus training-loop conditionals.
- Goal: Remove the need for a Matryoshka-specific SAE class and training-loop branching by letting the Objective orchestrate how to run the model for its loss.
- Approach: Change the Objective API so it has authority to call `sae.encode` and `sae.decode` and produce its own reconstructions (including multiple reconstructions in the Matryoshka case). The training loop asks the Objective to compute the loss for a given `sae` and `x` and backpropagates.

## Context

The current abstraction splits responsibilities as:

- Activations: how hidden codes are produced (ReLU, TopK, BatchTopK) baked into the SAE.
- Objective: how loss terms are computed given `x`, `f_x`, `x_hat`.
- Matryoshka: implemented as both a different Objective and a different SAE subclass, because Matryoshka needs multiple reconstructions from prefix subsets of the hidden code.

This creates coupling and special-casing:

- The training loop checks if the Objective is Matryoshka to choose a different model or a different forward path.
- A Matryoshka SAE subclass exists only to expose a special forward for prefix decodes.

Key observation: given the standard SAE with `encode` and `decode`, Matryoshka reconstructions can be computed without a subclass by masking the hidden code and calling `decode` repeatedly:

```
f_x = sae.encode(x)
mask_i = build_prefix_mask(i, d_sae)
x_hat_i = sae.decode(f_x * mask_i)
```

Therefore, the Objective can fully own this policy (how many prefixes, how to sample them, how to reduce MSE across them) without touching the model or the training loop.

## Design

### New Objective API

Change Objective.forward to accept the `sae` and input batch `x`, and return the `Loss` (and optionally intermediate tensors for metrics).

Old signature:

```
def forward(self, x, f_x, x_hat) -> Loss
```

Proposed signature:

```
def forward(self, sae, x) -> Loss
```

Notes:

- The Objective is a small program that can decide how to run the SAE.
- This keeps Activations orthogonal and removes the Matryoshka SAE subclass.
- The training loop simplifies to a single call path for all objectives.

### VanillaObjective implementation sketch

```
class VanillaObjective(Objective):
    def forward(self, sae, x):
        f_x = sae.encode(x)
        x_hat = sae.decode(f_x)
        mse = mean_squared_err(x_hat, x).mean()
        l0 = (f_x > 0).float().sum(dim=1).mean(dim=0)
        l1 = f_x.sum(dim=1).mean(dim=0)
        sparsity = self.cfg.sparsity_coeff * l1
        return VanillaLoss(mse, sparsity, l0, l1)
```

### MatryoshkaObjective implementation sketch

```
class MatryoshkaObjective(Objective):
    def forward(self, sae, x):
        f_x = sae.encode(x)  # shape: (B, S)
        cuts = sample_prefix_cuts(S=f_x.shape[-1], n_prefixes=self.cfg.n_prefixes)
        # Build masks for each cut; mask[i] keeps [0:cuts[i])
        masks = [
            torch.zeros_like(f_x)
            .bool()
            .index_fill_(dim=1, index=torch.arange(cut, device=f_x.device), value=True)
            for cut in cuts
        ]
        # Decode each prefix
        x_hats = [sae.decode(f_x * mask.float()) for mask in masks]  # list of (B, D)
        # Stack and reduce over prefixes
        x_hats_PBD = torch.stack(x_hats, dim=0)
        mse_per_prefix = [mean_squared_err(x_hat, x).mean() for x_hat in x_hats]
        mse = torch.stack(mse_per_prefix).mean()
        l0 = (f_x > 0).float().sum(dim=1).mean(dim=0)
        l1 = f_x.sum(dim=1).mean(dim=0)
        sparsity = self.cfg.sparsity_coeff * l1
        return MatryoshkaLoss(mse, sparsity, l0, l1)
```

Implementation detail:

- `sample_prefix_cuts` can reuse the Pareto sampler we already have; it just moves into the Objective or into a small util.
- Masks can be broadcasted as `(1, S)` per cut if memory is tight; multiply with `f_x` before decoding.
- Bias handling is correct because we call the model’s standard `decode`. No repeated bias addition.

### Training loop changes

Current (simplified):

```
if isinstance(objective, MatryoshkaObjective):
    x_hat, f_x = sae.matryoshka_forward(x, cfg.n_prefixes)
else:
    x_hat, f_x = sae(x)
loss = objective(x, f_x, x_hat)
```

Proposed:

```
loss = objective(sae, x)
loss.loss.backward()
```

- Instantiate only `SparseAutoencoder`.
- Remove the `MatryoshkaSparseAutoencoder` subclass and the forward branching.
- Keep scheduling logic that mutates `objective.sparsity_coeff` as-is.

## Migration Plan

1. Add the new Objective.forward signature accepting `sae` and `x`.
2. Update VanillaObjective and MatryoshkaObjective to the new signature.
3. Adapt `train.py` to call `objective(sae, acts_BD)` and remove Matryoshka-specific branches.
4. Delete the Matryoshka SAE subclass and any dead code paths.
5. Ensure serialization remains unchanged, since the model type is now uniform.

Optional staged migration:

- Temporarily support both Objective signatures with an adapter in the training loop during refactor, then remove the old path once objectives are updated.

## Tests

Unit tests:

- Matryoshka masks produce non-increasing L0 with increasing prefix: for a fixed `f_x`, apply two cuts `c1 < c2`, assert `L0(f_x * mask(c1)) <= L0(f_x * mask(c2))`.
- Matryoshka reconstruction equals decode-of-mask: for random `f_x` and random `W_dec`, check `decode(f_x * mask)` equals the specialized path (there is no specialized path after refactor; this guards against accidental bias addition or indexing mistakes if you later optimize).
- Objective API: create a tiny SAE (e.g., `d_model=4`, `d_sae=8`), a tiny batch, run both VanillaObjective and MatryoshkaObjective; assert `loss.loss` is finite and gradients exist for `W_enc`, `W_dec`, `b_enc`, `b_dec` after backward.
- Prefix sampler: for large `S`, generate many cuts; assert they are sorted and within `[1, S]`, and that smaller cuts occur more often when using Pareto.

End-to-end tests:

- Train parity sanity: train for a few steps on a deterministic tiny dataset with VanillaObjective and with MatryoshkaObjective using `n_prefixes=1`. Loss curves should be comparable (not necessarily identical due to sampling but the same order of magnitude and trend). This tests that `n_prefixes=1` roughly reduces to vanilla.
- Exploding/NaN guard: run 200 steps with both objectives on random data scaled like real activations; assert no NaNs in loss and parameters.

Manual tests:

- Visual inspection of reconstructions: encode a batch once, create three prefix cuts (e.g., 5%, 20%, 100%), decode and compute MSEs; confirm monotonic improvement with larger prefixes on average.
- Performance sampling: profile time per step for `n_prefixes=1, 5, 10, 20`; confirm roughly linear cost with `n_prefixes` and acceptable throughput for your target.

## Pitfalls

- Double bias addition: avoid any custom partial decoding that adds `b_dec` per block and then sums; that would add the bias multiple times. Always compute prefix reconstructions via `decode(f_x * mask)`.
- Gradient masking: using boolean masks requires converting to float before multiplication. Failing to do so can throw dtype errors or silently cast on CPU/GPU divergently.
- Memory blow-up: stacking all `x_hat` for many prefixes and big batches can blow memory. Prefer looped reduction (accumulate MSE in a running sum) or decode in chunks if needed.
- Scheduler coupling: the training loop updates `objective.sparsity_coeff`. Ensure the new Objective still exposes the same attribute so the scheduler works. Consider making the scheduler call a setter if you later change the internal name.
- Metric computation: if you previously logged `x_hat` directly from the loop, you no longer have it. Either have the Objective expose optional extras for logging, or recompute a single `x_hat_full = sae.decode(f_x)` locally when needed for diagnostics.
- API skew in saved configs: if configs or checkpoints record old Objective signatures or model class names, migrate readers accordingly. Model serialization should remain stable because only the Objective API changes.
- TopK vs L1 sparsity: Matryoshka with BatchTopK often makes L1 term irrelevant. Ensure you don’t double-penalize sparsity and that configs reflect the intended regime.

## Implementation Tips

- Keep the mask construction simple and O(S) per prefix cut. A 1D mask with broadcasting is usually enough: shape `(1, S)` multiplied with `f_x`.
- If you want weighted prefixes (e.g., Pareto weights), keep a `weights` tensor and compute a weighted mean of the per-prefix MSEs.
- Keep `sample_prefix_cuts` deterministic under a passed `seed` for repeatable tests.
- Preserve current mean-squared error function semantics (normalization flag, scaling) so training curves remain comparable.

## Rollback Plan

- If any regressions appear, re-enable the old code path guarded by a feature flag in the training loop and objective, then iterate on the new path behind the flag until parity is restored. Once stable, remove the flag.
