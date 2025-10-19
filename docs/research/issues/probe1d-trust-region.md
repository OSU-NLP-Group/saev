# Goal

For each latent l and class c, minimize the ridge-regularized logistic NLL in 2 params (b, w).
We work in class slabs and stream CSR nonzeros; zeros are handled in closed form.

# One iteration (per slab)

1. **Intercept baseline and ridge.** Start from (b=logit(\pi_c)), (w=0). Ridge keeps solutions finite and conditions the 2×2 system. (Standard TR/LM for logistic is well-studied.) [Journal of Machine Learning Research](https://www.jmlr.org/papers/v9/lin08b.html)

2. **Accumulate Newton stats with sparsity.**
   From nonzeros:
   (g_1=\sum (\mu-y)x,; h_0=\sum s,; h_1=\sum s x,; h_2=\sum s x^2) with (s=\mu(1-\mu)).
   Fold zeros via (n_0\mu_0) and (n_0 s_0). Add ridge: (g_0 += \lambda(b-b_{base}),; g_1 += \lambda w,; h_0 += \lambda,; h_2 += \lambda).

3. **Levenberg–Marquardt step (damped Newton).**
   Solve ((H+\lambda_k I)\Delta=g) (closed-form 2x2), then compute the **predicted reduction** (pred=g^\top \Detla - \frac12,\Delta^\top H \Delta). This is the classic TR/LM quadratic-model decrease. [UCI Mathematics](https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)

4. **Trust region = logit-budget.**
   Pick a global logit budget (\delta_{logit}) (e.g., 8). For each latent l, estimate a robust nonzero magnitude (q_x[l]) (e.g., 95th percentile of |x|). Enforce
   [
   |\Delta b|\le \delta_{logit},\qquad
   |\Delta w|\le \delta_{logit}/q_x[l],
   ]
   by scaling (\Delta) if needed. This is just a **scaled trust region** (elliptical norm) with a per-parameter diagonal (D=diag(1,;q_x[l])): "keep typical logit changes bounded." (Scaled/elliptical TR with a diagonal (D) is standard; the logit budget is an interpretable instantiation.) [UCI Mathematics](https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)

5. **Accept/adapt damping (with lagged rho).**
   Within the iteration, require (pred>0), finite step, and inside the logit budget; otherwise increase (\lambda_k) and retry (a few attempts).
   Also track a **lagged ratio**
   [
   \rho^{(t)}=\frac{L_t - L_{t+1}}{pred_t}
   ]
   using the loss you already compute next iteration. Use rho to adapt (\lambda_k) for the following step:

* if rho >= 0.75 and the previous step wasn’t clipped → (\lambda_k \leftarrow \lambda_k \cdot \text{shrink}) (more Newton-like);
* if rho <= 0.25 or the step was clipped/unstable → (\lambda_k \leftarrow \lambda_k \cdot \text{grow}) (more GD-like).
  Trust-region updates based on the ratio of **actual vs. predicted** reduction are the textbook rule; using approximate or delayed information is consistent with the literature on **inexact/gradient-only TR** when function evaluations are costly. [UCI Mathematics](https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)

6. **Update and stop.**
   Apply (\theta \leftarrow \theta - \Delta). Stop when max(|grad|,|\Detla|) < tol or max iters. (TR Newton for logistic is a strong default.) [Journal of Machine Learning Research](https://www.jmlr.org/papers/v9/lin08b.html)
