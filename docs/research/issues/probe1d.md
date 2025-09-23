# Probe1D: Per‑feature x per‑class 1D logistic probes on SAE activations

Given a trained SAE over ViT patch activations and a dataset with patch‑level semantic labels (ADE20K, 150 classes or FishVista, 10 labels), estimate how predictive each single SAE feature is for each class. For every `(feature f, class c)` pair, fit a binary 1D logistic regression using only activation `a_f` to predict `y_c \in {0,1}` for a patch. Report per‑pair train/val probe loss (NLL) and accuracy. For each class, surface the best latent (lowest validation loss) and export artifacts for downstream analysis.

This is intentionally simple and convex: each model has only two parameters `(w_{f,c}, b_{f,c})`. We will train n_features x n_classes independent probes (e.g., 16,000 x 150 = 2.4M models). The key engineering challenge is IO/throughput and parallelization under GPU memory limits.

## Inputs

* A trained SAE checkpoint (e.g., SAE‑16K) compatible with `saev` inference.
* Dataset with per‑patch labels for ADE20K (150 classes). We assume a function mapping each ViT patch to a single class id via majority pixel label.
* Access to A100 40GB GPU(s). Single‑GPU is the baseline; Slurm array is optional but recommended.

## Outputs:

* `parquet` table of per (feature, class) probe metrics (+ learned `w`, `b`).
* `parquet` table of best latent per class (`class_id`, `best_feature`, `val_loss`, `val_acc`).

## Details

* We include extremely minimal L2 regularization (ridge) to avoid divergence on linearly separable pairs.
* We use balanced weighting or `pos_weight` to address class imbalance.

SAE inference will be done once at the start from ViT patch activations (no multi‑TB dumps) but stored as a sparse matrix in memory.
We assume an average L0 of 400 per image. That means that we have n_imgs * n_patches_per_img * 400 non-zero elements + n_classes class binary labels.

For example, for ADE20K with 256 patches/image, that's 22K x 256 x 400 = 2.25B elements. Since we need a lookup integer into our list of 22K x 256 x 32K (180.2B) possible entries, we'll use int64 offsets and fp32 for the values. Thus, 22K x 256 x 400 x 12 bytes = 27.0 GB, which can fit on our 40GB A100.
The labels are 22K x 256 x 151 x 1 byte = 850.4 MB.

So we will first iterate through the entire dataset of ViT activations, run SAE inference, then keep that sparse SAE feature matrix in memory.

Then for each (feature, class) pair, we need to store two parameters, $w$ and $b$, each scalars.
Then, for each (feature, class) pair, we need 5 scalars: G0, G1, S0, S1 and S2.
These scalars can be calculated via streaming over the non-zero elements in the sparse SAE feature matrix.

So we will have 2 parameters + 5 scalars + 2 parameter updates + 1 loss + 1 accuracy = 11 fp32 values per (feature, class) pair, for a total of 11 x 32K x 151 x 4 bytes = 212.6 MB.
Again, this can be stored on the GPU trivially.
Thus, we calculate some sums over the sparse feature matrix, calculating the 5 scalars per (feature, class) pair, and update our estimates for each parameter.

One challenge will be calculating these sums quickly; due to the sparse layout of the feature matrix, it might require some tricks. I'm not sure. Don't use torch.nn or torch's autograd for anything; calculate all these terms by hand.

## Math

We have N samples indexed by j.

* Feature: $x_j \in \R$ (this is the activation of one SAE latent for sample j).
* Label: $y_j \in \{0,1\}$.
* Parameters: intercept $b$, slope $w$.
* Logit and mean:

Define the logit $\eta_j = b + w x_j$ and probability $\mu_j = \sigma(\eta_j) = \frac{1}{1+e^{-\eta_j}}$.

Negative log-likelihood (binary cross-entropy):
$$
L(b,w) = \sum_{j=1}^{N}[-y_j\log \mu_j - (1-y_j)\log(1-\mu_j)]
$$

Define the residual $r_j = \mu_j - y_j$.
Define a weight term $s_j = \mu_j(1-\mu_j)$.

Sum over j and collect five statistics:

$$
S_0 = \sum_j s_j \\
S_1 = \sum_j s_j x_j \\
S_2 = \sum_j s_j x_j^2 \\
G_0 = \sum_j (\mu_j - y_j) \\
G_1 = \sum_j (\mu_j - y_j) x_j
$$

with $s_j = \mu_j(1-\mu_j)$ as above.

Let $\det H = S_0 S_2 - S_1^2$.

Then
$$
\Delta b = \frac{ S_2 G_0 - S_1 G_1}{\det H} \\
\Delta w = \frac{-S_1 G_0 + S_0 G_1}{\det H}
$$

and the update is

$$
\boxed{\;b \leftarrow b - \Delta b,\qquad w \leftarrow w - \Delta w.\;}
$$

Given current $(b,w)$:

1. Compute logits and means: $\eta_j=b+w x_j,\; \mu_j=\sigma(\eta_j)$.
2. Residuals and weights: $r_j=\mu_j-y_j,\; s_j=\mu_j(1-\mu_j)$.
3. Accumulate:
   * $G_0=\sum r_j$, $G_1=\sum r_j x_j$.
   * $S_0=\sum s_j$, $S_1=\sum s_j x_j$, $S_2=\sum s_j x_j^2$.
4. Form $\det H=S_0 S_2 - S_1^2$ (optionally add a tiny ridge $\lambda$: $S_0{+}\lambda, S_2{+}\lambda$).
5. Compute $\Delta b,\Delta w$ as above; update $b,w$.


At $w=0$, the optimal $b$ makes $\sum_j(\mu_j - y_j)=0$ with $\mu_j=\sigma(b)$ constant across j. Hence

$$
\sigma(b) = \frac{1}{N}\sum_j y_j \quad\Rightarrow\quad
\boxed{\,b_0 = \operatorname{logit}\left(\frac{\sum_j y_j}{N}\right),\; w_0=0.\,}
$$

* Clip $\mu_j$ to $[1e{-}6,1-1e{-}6]$ to avoid $\log 0$.


## Sparse Update

Next, we can swap in sparsity: the formulas stay the same, but only nonzeros contribute to $G_1,S_1,S_2$; zeros still contribute to $G_0,S_0$ via closed forms.

* Total samples $N$, labels $y_j\in\{0,1\}$.
* Feature values $x_j$; most are $0$.
* Let $\mathcal{Z}=\{j: x_j=0\}$, $\mathcal{NZ}=\{j: x_j\neq 0\}$.
* Let $n=\lvert\mathcal{NZ}\rvert$ (nnz for this latent), so $\lvert\mathcal{Z}\rvert=N-n$.
* Parameters $b,w$; logits $\eta_j=b+w x_j$; probs $\mu_j=\sigma(\eta_j)$; IRLS weights $s_j=\mu_j(1-\mu_j)$.
* Define the "zero-logit" and "zero-weight" once per iteration:
$$
\mu_0=\sigma(b)\\
\s_0=\mu_0(1-\mu_0).
$$

For all $j\in\mathcal{Z}$, $\eta_j=b,\ \mu_j=\mu_0,\ s_j=s_0$.

Also precompute once per class: $\pi=\sum_{j=1}^N y_j$ (total positives). You will reuse $\pi$ for every latent.

For **zero** entries ($x_j=0$):

* Gradient pieces w\.r.t. $w$ and mixed Hessian terms carry factors of $x_j$ or $x_j^2$ and therefore vanish.
* Only the intercept terms contribute, and they are constant across $\mathcal{Z}$.

That gives closed forms for the zero group:

$$
\begin{aligned}
G_{1,\text{zero}}&=\sum_{j\in\mathcal{Z}}(\mu_j-y_j)\,x_j=0,\\
S_{1,\text{zero}}&=\sum_{j\in\mathcal{Z}} s_j x_j=0,\\
S_{2,\text{zero}}&=\sum_{j\in\mathcal{Z}} s_j x_j^2=0,\\
S_{0,\text{zero}}&=\sum_{j\in\mathcal{Z}} s_j=(N-n)\,s_0,\\
G_{0,\text{zero}}&=\sum_{j\in\mathcal{Z}} (\mu_j-y_j)=(N-n)\,\mu_0-\sum_{j\in\mathcal{Z}}y_j.
\end{aligned}
$$

You don’t want to iterate $\sum_{j\in\mathcal{Z}} y_j$ explicitly. Use $\sum_{j\in\mathcal{Z}}y_j=\pi-\sum_{j\in\mathcal{NZ}}y_j$ to eliminate it.

## Nonzero group (stream over events)

For **nonzeros** ($j\in\mathcal{NZ}$), compute as in the dense case, but only on the events you store:

$$
\begin{aligned}
G_{0,\text{nz}}&=\sum_{j\in\mathcal{NZ}}(\mu_j-y_j),\\
G_{1,\text{nz}}&=\sum_{j\in\mathcal{NZ}}(\mu_j-y_j)\,x_j,\\
S_{0,\text{nz}}&=\sum_{j\in\mathcal{NZ}} s_j,\\
S_{1,\text{nz}}&=\sum_{j\in\mathcal{NZ}} s_j x_j,\\
S_{2,\text{nz}}&=\sum_{j\in\mathcal{NZ}} s_j x_j^2.
\end{aligned}
$$

While streaming these events, it’s convenient to also accumulate either

* the **sum of labels** on nonzeros, $Y_{\text{nz}}=\sum_{j\in\mathcal{NZ}}y_j$, or
* the **sum of probabilities**, $M_{\text{nz}}=\sum_{j\in\mathcal{NZ}}\mu_j$.

Both give a simple closed form for $G_0$ without touching zeros 

Note that $G_{0,\text{nz}}=\sum_{j\in\mathcal{NZ}}(\mu_j-y_j)=M_{\text{nz}}-Y_{\text{nz}}$. Plugging that and simplifying cancels $Y_{\text{nz}}$, yielding

$$
\boxed{\,G_0 = M_{\text{nz}} + (N-n)\mu_0 - \pi\,}
$$

with the same formulas for $G_1,S_0,S_1,S_2$ as above. This version is often nicer because you’re already computing $\mu_j$ for $s_j$.

Either way, **only the nonzero events** are ever iterated; the zeros are collapsed into two constants depending on $(b,w)$ only through $\mu_0$ and $s_0$.

## Newton step (unchanged)

Once you have the five scalars, it's the same as in the dense case.

## One sparse Newton iteration—checklist

1. Compute $\mu_0=\sigma(b)$, $s_0=\mu_0(1-\mu_0)$.
2. Initialize accumulators: $M_{\text{nz}},G_{1,\text{nz}},S_{0,\text{nz}},S_{1,\text{nz}},S_{2,\text{nz}} \leftarrow 0$; $n\leftarrow 0$.
3. Stream each nonzero event $(x_j\neq 0)$:

   * $\eta=b+w x_j$; $\mu=\sigma(\eta)$; $s=\mu(1-\mu)$.
   * $M_{\text{nz}}{+}{=}\mu$; $G_{1,\text{nz}}{+}{=}(\mu-y_j)x_j$.
   * $S_{0,\text{nz}}{+}{=}s$; $S_{1,\text{nz}}{+}{=}s x_j$; $S_{2,\text{nz}}{+}{=}s x_j^2$.
   * $n{+}{=}1$.
4. Finish:

   * $G_0 = M_{\text{nz}} + (N-n)\mu_0 - \pi$.
   * $G_1 = G_{1,\text{nz}}$.
   * $S_0 = S_{0,\text{nz}} + (N-n)s_0$.
   * $S_1=S_{1,\text{nz}},\ S_2=S_{2,\text{nz}}$.
5. Solve for $\Delta$ and update $b,w$.

## Edge cases you’ll hit

* **nnz = 0**: Then $G_1=S_1=S_2=0$. The slope is unidentifiable; the model collapses to intercept-only. Keep $w=0$; set $b$ to the logit of prevalence (or run a 1D Newton on $b$ alone). If you still solve 2×2, add a small ridge to $S_2$.
* **Near-separable**: If probabilities saturate (either in the zero group via $\mu_0$ or in nonzeros), $S_0$ (or parts of it) shrinks and $H$ can be ill-conditioned. Use a tiny damping/L2 and, if needed, clip logits or compute with “BCE-with-logits” style numerics.

That’s the whole change: same Newton, but you **never** touch the zero entries; you add just two constants, $(N-n)\mu_0$ and $(N-n)s_0$, and everything else is streamed over the latent’s nonzero events.

## Dataset & Patch Labeling

1. Patch extraction: use the same ViT patch geometry as the SAE’s training ViT. Read existing code for how we match up SAE patches with image patches (contrib/trait_discovery/scripts/visuals.py).
2. Labeling rule (default):
   * Assign a patch the class `c*` whose pixel count inside the patch is maximal. Ignore "void/background" if present; if only void, mark the patch as a negative example for all classes.
   * Produce `Y[:,c]` as one‑hot for the chosen class; all other classes are 0.
3. Record per‑class counts and class‑imbalance stats.

This is mostly already implemented with the `labels.bin` file for saved ViT activations (see src/saev/data/protocol.md).


## Splits & Evaluation Protocol

* All reported metrics are on the validation split.
* Per `(f,c)` compute: `nll`, `accuracy@0.5`, `balanced_accuracy`, optional `AUROC`, `AUPRC`.
* For each `c`, choose best feature by min val nll (tie‑break by highest val AUROC).


## Metrics, Artifacts & Formats

* Per‑pair parquet schema (`metrics_pair.parquet` shards per class):

  * `class_id:int16, feature_id:int32, w:float32, b:float32, n_train:int32, n_val:int32, pos_train:int32, pos_val:int32, nll_train:float32, nll_val:float32, acc_train:float32, acc_val:float32, auroc_val:float32, auprc_val:float32`.
* Best latent per class (`best_per_class.parquet`): `class_id, feature_id, nll_val, acc_val, auroc_val`.


## Testing & Validation

Unit tests (small synthetic):

* Generate `x ~ N(0,1)`, `y ~ Bernoulli(sigmoid(w*x+b))` for 8 features x 3 classes; ensure recovered `(w,b)` within tolerance and metrics reasonable.
* Degenerate cases: `all y=0`, `all y=1`, separable data; verify L2 prevents divergence and code exits cleanly with NaNs guarded.


## Style & Quality Bar

* Follow repo norms: typed dataclasses, `beartype`, `tyro` CLI, no hard‑wrapping comments, `uvx ruff format/check`, add tests, and docstrings with examples. Prefer early returns. Keep modules small.
* Logging: `helpers.progress`; log per‑class counts and timing.
* Repro: fix seeds
