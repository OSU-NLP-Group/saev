# Goal

For each latent l and class c, minimize the ridge-regularized logistic NLL in 2 params (b, w).
We work in class slabs and stream CSR nonzeros; zeros are handled in closed form.

# One iteration (per slab)

1. **Intercept baseline and ridge.** Start from (b=logit(\pi_c)), (w=0). Ridge keeps solutions finite and conditions the 2×2 system. (Standard TR/LM for logistic is well-studied.) [Journal of Machine Learning Research](https://www.jmlr.org/papers/v9/lin08b.html)

2. **Accumulate Newton stats with sparsity (mean loss).**
   Form the per-sample averages rather than raw sums so the quadratic model lives on the same scale as the mean logistic loss we optimize. While streaming CSR events maintain mean-scaled statistics (with (s=\mu(1-\mu))):
   [
   g_0=\frac1n\left(\sum_{j \in \mathcal{NZ}} (\mu_j-y_j) + n_0 \mu_0 - y_{0,\text{count}}\right),
   ]
   [
   g_1=\frac1n\sum_{j \in \mathcal{NZ}} (\mu_j-y_j)x_j,\quad
   h_0=\frac1n\left(\sum_{j \in \mathcal{NZ}} s_j + n_0 s_0\right),\quad
   h_1=\frac1n\sum_{j \in \mathcal{NZ}} s_j x_j,\quad
   h_2=\frac1n\sum_{j \in \mathcal{NZ}} s_j x_j^2.
   ]
   Here (n_0) is the zero count, (\mu_0=\sigma(b)), (s_0=\mu_0(1-\mu_0)), and (y_{0,\text{count}}=n_{\text{pos}}-\sum_{\mathcal{NZ}}y_j). Equivalently, (g_0 = (\mu_{\text{sum,all}} - n_{\text{pos}})/n) with (\mu_{\text{sum,all}} = \sum_{\mathcal{NZ}}\mu_j + n_0\mu_0). Add ridge (`wd`) using the same mean scaling: (g_0 += wd(b-b_{base})), (g_1 += wd\,w), (h_0 += wd), (h_2 += wd). This keeps gradients/predicted reductions comparable to the actual drop in the averaged loss.

3. **Levenberg–Marquardt step (damped Newton).**
   Solve ((H+\text{damp}_k D^\top D)\Delta=g) (closed-form 2×2) with (D=\operatorname{diag}(1,q_x[l])). Always compute the **predicted reduction** as (pred=g^\top \Delta - \frac12 \Delta^\top H \Delta) using the ridge-only Hessian; recompute it after any clipping so the quadratic model matches the step actually taken. [UCI Mathematics](https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)

4. **Trust region = logit-budget.**
   Pick a global logit budget (\delta_{logit}) (e.g., 8). For each latent l, compute the root-mean-square nonzero magnitude (q_x[l]) = \sqrt{\mathbb{E}[x_{\text{nz}}^2]} with a fallback of 1.0 for empty latents, then clamp it to (\[q_{x,\min}, q_{x,\max}\]) with (q_{x,\min}\approx10^{-6}). Enforce the scaled 2-norm constraint
   [
   \left\|\begin{bmatrix}\Delta b \\ q_x[l]\,\Delta w\end{bmatrix}\right\|_2 \le \delta_{logit},
   ]
   i.e. scale (\Delta) so (|\Delta b|\le\delta_{logit}) and (|\Delta w|\le\delta_{logit}/q_x[l]). Record whether the step was clipped so subsequent damping updates stay conservative. This is a **scaled trust region** with diagonal metric (D) that keeps typical logit changes bounded. [UCI Mathematics](https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)

5. **Accept/adapt damping (with lagged rho).**
   Within the iteration, require (pred>0), finite step, and inside the logit budget; otherwise increase (\text{damp}_k) and retry up to a small cap (e.g., 5 attempts). After exhausting retries, fall back to a tiny gradient step projected into the logit budget so progress continues.
   Also track a **lagged ratio**
   [
   \rho^{(t)}=\frac{\bar{L}_t - \bar{L}_{t+1}}{pred_t}
   ]
   where (\bar{L}) denotes the mean loss. Because the quadratic model is built from the same averages, the ratio stays numerically stable. Use rho to adapt (\text{damp}_{k+1}) for the following step:

* if rho >= 0.75 and the previous step wasn’t clipped → (\text{damp}_{k+1} = \max(\text{damp}_k \cdot \text{shrink}, \text{damp}_{\min})) (more Newton-like);
* if rho <= 0.25 or the step was clipped/unstable → (\text{damp}_{k+1} = \min(\text{damp}_k \cdot \text{grow}, \text{damp}_{\max})) (more GD-like).
  Trust-region updates based on the ratio of **actual vs. predicted** reduction are the textbook rule; using approximate or delayed information is consistent with the literature on **inexact/gradient-only TR** when function evaluations are costly. [UCI Mathematics](https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)

6. **Update and stop.**
   Apply (\theta \leftarrow \theta - \Delta). Stop when max(|grad|,|\Delta|) < tol or max iters. (TR Newton for logistic is a strong default.) [Journal of Machine Learning Research](https://www.jmlr.org/papers/v9/lin08b.html)
   In practice we now use three concrete checks tied to the mean-loss formulation plus two additional scale-aware guards:
   - Gradient norm: stop when (\max(|g_0|,|g_1|) \le \text{tol}_{\text{grad}}) (e.g. 1e-3 in float32).
   - Predicted reduction: if the quadratic model predicts less than (\text{tol}_{\text{pred}}) mean-loss drop (e.g. 1e-6), accept the current parameters.
   - Curvature guard: when the average sigmoid curvature (\bar{s}=\mathbb{E}[\mu(1-\mu)]) falls below (\text{tol}_{\text{curv}}) (e.g. 1e-6), the coordinate is in the logistic tail and further Newton steps are fruitless.
   - Relative decrease: if (pred \le \text{tol}_{\text{pred,rel}}(|\bar{L}| + 1e-8)), stop to avoid absolute-scale sensitivity.
   - Monotonicity: accepted steps must not increase (\bar{L}); if they do, automatically grow (\text{damp}_k) and retry.
   These criteria are standard in trust-region/IRLS treatments and avoid relying solely on the noisy lagged (\rho).
