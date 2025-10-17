# Ill-Conditioned Probe1D Newton Steps

## Summary
- We are fitting per-latent logistic probes with a Newton-style solver that streams sparse activations.
- Repeated sweeps show the Newton system becoming ill-conditioned: the zero-activation slice dominates the gradient, logits saturate, `mu(1-mu)` collapses, and the Hessian determinant approaches zero.
- When that happens the solver emits enormous `db`/`dw` updates (10⁶–10⁷) and hits the iteration cap before gradients fall under `tol`.

## Evidence
- Training logs from 2025-10-16 (e.g. `2657235_0_log.out`, `2657817_0_log.out`, `2658166_0_log.out`) show:
  - `g0_zero` in the 10^6 range while `g0_nz` is <=10^5.
  - Hessian diagonals (`h0`, `h2`) below 1e-1, leading to determinants <1e-2.
  - Newton updates saturating at ~3e7 even after ridge values up to 1e-2.
- Diagnostic logging confirms the ridge term barely affects the maximally unstable coordinates because the raw curvature is so small.

## Current Mitigation
- Added a configurable `hessian_floor` (default `1e-4`) that clamps `h0` and `h2` before solving the 2×2 Newton system. This is equivalent to a lightweight Levenberg–Marquardt damping.
- Keeping ridge at `1e-8` for now; larger ridge smoothed iteration 0 but did not solve late-iteration blow-ups.

## Next Steps
- Verify the Hessian floor in new runs (see job `2658498`) and track whether the largest updates now stay bounded.
- If updates remain too large, consider:
  1. A trust-region step scaler (cap `|db|`, `|dw|` or perform a backtracking line search).
  2. Per-latent clipping of intercepts when `mu_0` saturates to the clamp epsilon.
  3. A more adaptive damping schedule tied to the observed determinant statistics.

## Update (2025-10-17)
- Implemented a per-coordinate trust region inside `_solve_lm_step` that enlarges LM damping until the Newton step falls below `lm_max_update`, and records when steps are clipped. This keeps intercept updates bounded (~O(10)) while still allowing aggressive progress where curvature is reliable. Logging now surfaces how many coordinates were clipped each iteration so we can inspect slabs that repeatedly hit the trust radius.
