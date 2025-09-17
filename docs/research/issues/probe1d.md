# Probe1D: Per‑feature x per‑class 1D logistic probes on SAE activations

## 0) Problem Statement (TL;DR)

Given a trained SAE (e.g., SAE‑16K) over ViT patch activations and a dataset with patch‑level semantic labels (ADE20K, 150 classes), estimate how predictive each single SAE feature is for each class. For every `(feature f, class c)` pair, fit a binary 1D logistic regression using only activation `a_f` to predict `y_c \in {0,1}` for a patch. Report per‑pair train/val probe loss (NLL) and accuracy. For each class, surface the best latent (lowest validation loss) and export artifacts for downstream analysis.

This is intentionally simple and convex: each model has only two parameters `(w_{f,c}, b_{f,c})`. We will train n\_features x n\_classes independent probes (e.g., 16,000 x 150 = 2.4M models). The key engineering challenge is IO/throughput and parallelization under GPU memory limits.


## 1) Scope & Assumptions

* Inputs:

  * A trained SAE checkpoint (e.g., SAE‑16K) compatible with `saev` inference.
  * Dataset with per‑patch labels for ADE20K (150 classes). We assume a function mapping each ViT patch to a single class id via majority pixel label.
  * Access to A100 40GB GPU(s). Single‑GPU is the baseline; Slurm array is optional but recommended.

* Outputs:

  * `parquet` table of per (feature, class) probe metrics (+ learned `w`, `b`).
  * `parquet` table of best latent per class (`class_id`, `best_feature`, `val_loss`, `val_acc`).

* Assumptions (explicit):

  * We include L2 regularization (ridge) to avoid divergence on linearly separable pairs.
  * We use balanced weighting or `pos_weight` to address class imbalance.
  * SAE inference will be done on‑the‑fly from ViT patch activations (no multi‑TB dumps).


## 2) Definitions

* `X \in \R^{NxF}`: SAE activations for N patches and F features (e.g., F=16,000). We only ever materialize feature blocks of X.
* `Y \in {0,1}^{NxC}`: one‑vs‑rest labels for C ADE20K classes (C=150). `Y[:,c]` is reused for all features.
* For each `f,c`, model: `p(y=1|x_f) = sigmoid(w_{f,c} ⋅ x_f + b_{f,c})`. Loss is BCEWithLogits with optional class weights.


## 3) Dataset & Patch Labeling (ADE20K)

1. Patch extraction: use the same ViT patch geometry as the SAE’s training ViT (e.g., 14x14 grid for 224x224). For variable‑sized ADE20K images, center‑crop/resize like the ViT preprocessing.
2. Labeling rule (default):
   * Assign a patch the class `c*` whose pixel count inside the patch is maximal. Ignore "void/background" if present; if only void, mark the patch as ignore.
   * Produce `Y[:,c]` as one‑hot for the chosen class; all other classes are 0.
3. Filtering: discard ignored patches before training; record per‑class counts and class‑imbalance stats.

This is mostly already implemented with the `labels.bin` file.


## 4) Splits & Evaluation Protocol

* All reported metrics are on the entire task. We assume L2 prevents overfitting.
* Per `(f,c)` compute: `nll`, `accuracy@0.5`, `balanced_accuracy`, optional `AUROC`, `AUPRC`.
* For each `c`, choose best feature by min val nll (tie‑break by highest val AUROC).


## 5) Optimization Strategy

Why not naively train 2.4M probes at once? `B x F x C` logits explode memory (orders of billions). We instead block by features and iterate over classes.

* Each probe is 2‑parameter convex logistic regression; Newton updates are cheap if we can stream gradients/Hessians.
* For a given class c and a feature block `f \in [f0:f0+F_blk)`, over a mini‑batch of patches `[b0:b0+B)`:

  * Let `a \in R^{BxF_blk}` be activations; `y \in {0,1}^{B}` labels for class `c`.
  * Predictions: `p = sigmoid(a * w_f,c + b_f,c)` where `w_f,c, b_f,c \in R^{F_blk}` and `*` is elementwise multiply and broadcast add.
  * Accumulate across batches:

    * `g_w = \sum (p − y) ⊙ a` (sum over batch dimension)
    * `g_b = \sum (p − y)`
    * `h_ww = \sum p(1−p) ⊙ a^2`
    * `h_wb = \sum p(1−p) ⊙ a`
    * `h_bb = \sum p(1−p)`
  * Add L2: `g_w <- g_w + lambda w`; `h_ww <- h_ww + lambda`.
  * Solve independent 2x2 systems per feature via closed‑form for the Newton step:
    * `H = [[h_ww, h_wb], [h_wb, h_bb]]`, `g = [g_w, g_b]`  => `[delta w, delta b] = H^{-1} g`.
  * Update with a global damped Newton: `[w,b] <- [w,b] − lr * [delta w, delta b]` with `lr \in (0, 1]` (start 1.0, halve on NLL increase).
* Iterate for K passes over the data (default K=5–10). Stop early if `max(|delta w|, |delta b|) < tol` and NLL reduced < epsilon.

LBFGS fallback: For sanity/regression parity, support per‑class LBFGS on the same feature blocks. Since PyTorch LBFGS optimizes a scalar objective, wrap `w,b` as length‑`2*F_blk` vector and compute loss as mean NLL over the block. (Slower; use only for debugging or small runs.)

### 5.3 Class imbalance & separability

* Compute `pos_weight = (N_neg/N_pos)` per class for BCEWithLogits.
* Add L2 (`lambda=1e−3` default) to avoid weight blowup for separable cases; tuneable per class.

---

## 6) Throughput, Memory, and Parallelization

* Default dtypes: activations in `bfloat16` (A100‑friendly), parameters/accumulators in `float32`.
* Block sizes (good starting point): `B=8192` patches, `F_blk=2048` features, one class at a time (so logits tensor is `BxF_blk`).
* Streaming: iterate batches; never materialize full `X` or `X@W`.
* Caching: If ViT->SAE inference is the bottleneck, persist a rolling NVMe cache of most‑recent feature blocks as `memmap`/`zarr` shards.
* Parallelism plan (choose 1 or combine):

  1. Slurm array over classes: 150 jobs, each trains all features for its class on one GPU. Output is later merged.
  2. Single‑GPU, inter‑feature blocks: multithread loader + CUDA streams; loop classes sequentially.
  3. Multi‑GPU: shard classes across GPUs; identical code path as (1).

Runtime sanity: Parameters per class = `Fx2` ~= 32K floats ⇒ tiny. The heavy part is scanning patches & computing `σ`/sums.


## 7) Layout Design

```
contrib/trait_discovery/
  src/tdiscovery/
    probe1d.py    # training (GPU)
  scripts/
    probe1d.py    # Tyro CLI entrypoint
```

Use `tyro` for CLI (consistent with repo). Examples:

```
cd contrib/trait_discovery
uv run scripts/probe1d.py --sae-ckpt /path/sae_16k.pt
```


## 8) Metrics, Artifacts & Formats

* Per‑pair parquet schema (`metrics_pair.parquet` shards per class):

  * `class_id:int16, feature_id:int32, w:float32, b:float32, n_train:int32, n_val:int32, pos_train:int32, pos_val:int32, nll_train:float32, nll_val:float32, acc_train:float32, acc_val:float32, auroc_val:float32, auprc_val:float32`.
* Best latent per class (`best_per_class.parquet`): `class_id, feature_id, nll_val, acc_val, auroc_val`.


## 9) Testing & Validation

Unit tests (small synthetic):

* Generate `x ~ N(0,1)`, `y ~ Bernoulli(sigmoid(w*x+b))` for 8 features x 3 classes; ensure recovered `(w,b)` within tolerance and metrics reasonable.
* Degenerate cases: `all y=0`, `all y=1`, separable data; verify L2 prevents divergence and code exits cleanly with NaNs guarded.


## 10) Risks & Mitigations

* IO bottleneck (SAE inference).
* GPU memory spikes: keep class loop outermost to avoid `BxF_blkxC` tensors; cap `F_blk` adaptively.
* Class imbalance: large `pos_weight` can destabilize Newton; clip `pos_weight` to max (e.g., 100) and increase lambda.
* Separability: explicit L2 and damped Newton.


## 11) Suggested Milestones

* M0: Land `probe1d_spec.md`, skeleton dirs, Tyro stub, tests with synthetic data.
* M1: ADE20K patch label builder + splits (image‑grouped).
* M2: Newton trainer end‑to‑end on tiny subset; metrics written.
* M3: Slurm array + merging; runs at ADE20K scale class‑sharded.
* M4: Summary tables + basic notebook for exploration.
* M5: Cleanup, docstrings, `just` targets, README section.


## 12) Style & Quality Bar

* Follow repo norms: typed dataclasses, `beartype`, `tyro` CLI, no hard‑wrapping comments, `uvx ruff format/check`, add tests, and docstrings with examples. Prefer early returns. Keep modules small.
* Logging: `helpers.progress`; log per‑class counts and timing.
* Repro: fix seeds
