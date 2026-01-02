# Proposal Audit Evaluation

## Goal

Compare TopK Matryoshka SAEs (with AuxK as an auxiliary loss) under the proposal-audit framework for ADE20K scene prediction and FishVista habitat prediction tasks. We will max pool over SAE patch-level activations and evaluate a sparse linear method, a conditional tree method, and a univariate correlation baseline for feature proposal.

## Scope

- Datasets: ADE20K (scene labels -> semantic segmentation features) and FishVista (habitat labels -> semantic segmentation features).
- Class subset: ADE20K top 50 scene classes by frequency on the training split (assert no ties; cutoff has 47 samples for rank 51). Use all FishVista habitats.
- Models: all trained DINOv3 SAEs from the 6 layers used throughout this project (layers 13, 15, 17, 19, 21, 23; 0-indexed).
- Pooling: max over patches only.
- Proposal methods: sparse linear, decision tree, and univariate correlation
- Audit methods: semantic segmentation labels, measuring how many of the top proposed features are grounded in the semantic concepts.

## Outputs

Results table with one row per dataset and layer, with classifier, classifier hyperparameters, and tracked metrics.

Proposed columns (subject to change):

- dataset, split, n_classes
- model_family, backbone_name, layer, feature library method (library_key), feature library hparams
- clf_key, clf hparams
- metrics (include all proposal + audit metrics listed below)

## Metrics (Feature Proposal Stage)

- Accuracy (top-1) for a simple sanity check.
- Balanced accuracy to handle class imbalance.
- Macro F1 for class-level balance.
- Log loss for calibration.
- Sparsity diagnostics: number of nonzero coefficients (SparseLinear) or number of features used / n_nodes (DecisionTree).

## Metrics (Audit Stage)

- Yield@B for B in {3, 10, 30, 100, 300, 1000}.
- AUC_B (area under Yield-vs-Budget).
- Best-class AP per feature, with tau threshold for grounded vs not grounded.

The results table should include all metrics listed above.

## Assumptions (to confirm)

- ADE20K provides a single scene label per image and we will treat this as a multi-class classification problem.
- FishVista habitat labels are binary per class for the Pearson correlation baseline.
- Max pooling uses the maximum activation over all ViT patches per image for each SAE latent.
- The evaluation is per-layer and per-SAE, with no cross-layer ensembling.

## Proposal-Audit Framework

We evaluate feature libraries with a fixed propose–audit protocol under a browsing budget.
A frozen ViT backbone $f$ maps images $x_i$ to patch tokens $t_{i,p}$; a library method $g$ (SAE or baseline decomposition) is trained unsupervised on tokens to produce per-patch activations $z_{i,p}\in\mathbb{R}^m$, and each image is summarized by a fixed pooling $X_i=\mathrm{pool}_p(z_{i,p})$.
We then fit a sparse predictor for proposal labels $K$ (e.g., habitat / condition) on the training split using L1-regularized logistic regression (multinomial or one-vs-rest), yielding weights $W$; per-feature importance is computed by aggregating magnitude across classes, e.g. $s_j=\sum_c |W_{c j}|$ (or via bootstrap selection frequency), inducing a ranking $\pi$.
Critically, proposal labels $K$ and audit labels $H$ (e.g., body-part segmentations or semantic masks) are disjoint: feature selection is driven only by $K$, while interpretability is assessed only via $H$, so the protocol measures whether task-relevant features align with orthogonal structure rather than merely predicting the training task.
For a budget $B$, we select $S_B=\{j:\pi(j)\le B\}$ and audit each $j\in S_B$ on held-out images by treating its activation map $a_{i,j}(p)=z_{i,p,j}$ as a score field and computing alignment to all audit classes $h\in H$ (threshold-free AP); we define $\mathrm{AP}_j=\max_h \mathrm{AP}(a_{i,j},\mathbf{1}[h])$ and declare a feature anatomically grounded if $\mathrm{AP}_j\ge\tau$ for a fixed $\tau$.
Discovery yield is $\mathrm{Yield}(B)=\frac{1}{B}\sum_{j\in S_B}\mathbf{1}[\mathrm{AP}_j\ge\tau]$ evaluated at $B\in\{3,10,30,100,300,1000\}$ with scalar summary $\mathrm{AUC}_B=\sum_B \mathrm{Yield}(B)$ (area under the Yield-vs-budget curve).
Uncertainty is estimated by bootstrapping images; validity is assessed with a proposal-stage permutation null (shuffle $K$, refit, rerank, recompute yield).
All libraries (SAE variants and non-SAE baselines such as raw units, PCA, NMF, clustering) share identical $f$, pooling, proposal model, audit metric, budgets, and $\tau$, isolating the effect of the library on discovery yield.

\begin{enumerate}
\item \textbf{Inputs.} A frozen backbone $f$, a feature-library method $g$ (SAE or baseline), a proposal label set $K$ (e.g., habitat/condition), an audit label set $H$ (e.g., part/semantic masks), browsing budgets $\mathcal{B}$ (the number of features a scientist can feasibly inspect), and a grounding threshold $\tau$ (default $\tau=0.3$). Crucially, $K$ and $H$ must be disjoint: feature selection uses only $K$, while interpretability is assessed only via $H$.

\item \textbf{Learn a feature library (unsupervised).} Train $g$ on patch tokens $t_{i,p}=f(x_i)$ to obtain per-patch activations $z_{i,p}\in\mathbb{R}^m$.

\item \textbf{Pool to image level.} For each image, pool per-patch activations into a fixed summary $X_i=\mathrm{pool}_p(z_{i,p})$.

\item \textbf{Proposal (rank features using only $K$).} Fit a sparse predictor of $K$ from $\{X_i\}$ (default: L1-logistic). Convert the fitted model into a per-feature importance score (e.g., sum of absolute weights across classes) and rank features by importance.

\item \textbf{Audit (measure grounding using only $H$).} For each budget $B$, take the top-$B$ ranked features. For each selected feature, treat its activation as a spatial score map and compute alignment to the audit masks (primary: best-class AP across all $h\in H$).

\item \textbf{Discovery yield.} Mark a selected feature as grounded if its best-class AP exceeds $\tau$, and report $\mathrm{Yield}(B)$ as the fraction of top-$B$ features that are grounded. Summarize across budgets by the area under the Yield-vs-budget curve.

\item \textbf{Uncertainty and nulls.} Bootstrap over audit images for confidence intervals, and run a proposal-stage permutation null by shuffling $K$ and repeating the protocol end-to-end.

\item \textbf{Qualitative/expert interpretation.} Inspect representative grounded features (including ``unexpected anatomy'') and consult domain experts for biological interpretation; this step is explicitly separated from the quantitative yield metric.
\end{enumerate}

## Decisions from discussion

- ADE20K class frequency ordering uses the training split only; assert there are no ties.
- Use the same ADE20K splits and DINOv3 layer choices as prior runs; use `contrib/trait_discovery/sweeps/003_auxk/probe1d_metrics.py` as the source of truth.
- Candidate ADE20K TopK + AuxK Matryoshka SAEs are defined in `contrib/trait_discovery/sweeps/003_auxk/probe1d_metrics.py`.
- Candidate FishVista TopK + AuxK Matryoshka SAEs are defined in `contrib/trait_discovery/sweeps/004_fishbase/probe1d_metrics.py`.
- Image-level score for a latent is the max activation across all patches in an image; rankings are computed over image-level scores per class/label.
- Tree definitions and sparsity constraints likely live in `contrib/trait_discovery/src/tdiscovery/classification.py`.
- Sparse linear method should use `SparseLinear` (L1-logistic) from `contrib/trait_discovery/src/tdiscovery/classification.py`.
- Tree method should use `DecisionTree` (sklearn DecisionTreeClassifier) from the same module.
- Univariate correlation will use raw Pearson (keep sign; labels are binary) to preserve directionality and avoid selecting anti-class features.
- We need at least a sparse linear method, a tree method with conditionality, and a univariate method.
- DINOv3 layer indices to evaluate are 13, 15, 17, 19, 21, 23 (0-indexed, 23 is the last layer).
- Existing shard paths:
  - FishVista train: `/fs/scratch/PAS2136/samuelstevens/saev/shards/e65cf404` (from `contrib/trait_discovery/sweeps/004_fishbase/probe1d_metrics.py`)
  - FishVista val: `/fs/scratch/PAS2136/samuelstevens/saev/shards/b8a9ff56` (from `contrib/trait_discovery/sweeps/004_fishbase/probe1d_metrics.py`)
  - ADE20K train: `/fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0` (from `contrib/trait_discovery/sweeps/003_auxk/probe1d_metrics.py`)
  - ADE20K val: `/fs/scratch/PAS2136/samuelstevens/saev/shards/3802cb66` (from `contrib/trait_discovery/sweeps/003_auxk/probe1d_metrics.py`)
- FishVista proposal labels use the `habitat` column (see `contrib/trait_discovery/sweeps/004_fishbase/cls_train_tree.py`).

## Open questions

- For FishVista, identify the audit segmentation labels and dataset; segfolder2 appears to live at `/fs/scratch/PAS2136/samuelstevens/derived-datasets/fish-vista-segfolder2`.
- Confirm whether the results table should include both ImageNet-trained and ADE20K-trained SAE runs from `contrib/trait_discovery/sweeps/003_auxk/probe1d_metrics.py` for ADE20K evaluation.
- If ADE20K shards lack the `scene` label column, create a `format_ade20k.py` script to re-format the raw ADE20K dataset to match segfolder conventions.

## Implementation Plan for `eval_cli`

### Overview

The audit stage evaluates whether classifier-selected features are "grounded" in semantic segmentation concepts. For each proposed feature, we treat its patch-level activation as a score field and compute AP against each segmentation class, then take the max AP across classes. Features with max AP >= tau are considered grounded.

### Data Flow

```
Inputs:
  - Trained classifier checkpoints (from train_cli): cls_{task}_{agg}_{cls_params}.pkl
  - SAE patch-level activations: token_acts.npz (sparse CSR, shape: n_patches x d_sae)
  - Segmentation labels: labels.bin (uint8 memmap, shape: n_images x patches_per_image)

Outputs:
  - Per-feature AP scores: ap_s.npy (shape: d_sae)
  - Best class per feature: best_class_s.npy (shape: d_sae, dtype: int)
  - Yield@B metrics for B in {3, 10, 30, 100, 300, 1000}
  - Feature rankings from classifier weights
```

### EvalConfig Fields

```python
@dataclasses.dataclass(frozen=True)
class EvalConfig:
    cls_checkpoint: pathlib.Path
    """Path to trained classifier checkpoint from train_cli."""
    test_shards: pathlib.Path
    """Test shards directory with token_acts.npz and labels.bin."""
    run: pathlib.Path
    """SAE run directory for loading SAE activations."""
    tau: float = 0.3
    """Grounding threshold: feature is grounded if best-class AP >= tau."""
    budgets: tuple[int, ...] = (3, 10, 30, 100, 300, 1000)
    """Browsing budgets for Yield@B computation."""
```

### Algorithm

1. **Load classifier checkpoint** and extract feature rankings:
   - SparseLinear: importance_j = sum_c |W_cj|, rank by descending importance
   - DecisionTree: importance_j = feature_importances_, rank by descending importance

2. **Load patch-level data**:
   - SAE activations: scipy.sparse.load_npz(token_acts.npz) -> shape (n_patches, d_sae)
   - Segmentation labels: np.memmap(labels.bin) -> shape (n_images, patches_per_image)

3. **Compute best-class AP for each SAE latent** (training-free):
   - For each latent j in {0, ..., d_sae-1}:
     - Extract activation scores: acts_n = token_acts[:, j].toarray().flatten()
     - For each segmentation class c:
       - Binary labels: labels_c_n = (seg_labels.flatten() == c).astype(float)
       - Compute AP(acts_n, labels_c_n) using vectorized implementation
     - best_ap_j = max over c of AP
   - This can be vectorized: compute AP for all (latent, class) pairs simultaneously

4. **Vectorized AP computation** (adapted from `tdiscovery/metrics.py` lines 193-206):
   ```python
   # For a single latent j, compute AP against all classes simultaneously
   acts_n = token_acts[:, j].toarray().flatten()  # (n_patches,)
   sort_idx = np.argsort(acts_n)[::-1]  # descending
   labels_sorted_nc = seg_labels_one_hot[sort_idx]  # (n_patches, n_seg_classes)

   tp_nc = labels_sorted_nc.cumsum(axis=0)
   ranks = np.arange(1, n_patches + 1, dtype=np.float32).reshape(-1, 1)
   precision_nc = tp_nc / ranks
   n_pos_c = seg_labels_one_hot.sum(axis=0)
   recall_nc = tp_nc / np.clip(n_pos_c, 1.0, None)
   recall_shift_nc = np.concatenate([np.zeros((1, n_seg_classes)), recall_nc[:-1]], axis=0)
   delta_recall_nc = recall_nc - recall_shift_nc
   ap_c = (precision_nc * delta_recall_nc).sum(axis=0)
   best_ap_j = ap_c.max()
   ```

5. **Compute Yield@B**:
   - For each budget B:
     - Select top-B features by classifier importance ranking
     - grounded_count = sum(best_ap[top_B_indices] >= tau)
     - Yield@B = grounded_count / B

6. **Compute AUC_B** (area under Yield-vs-Budget curve):
   - Simple sum: AUC_B = sum(Yield@B for B in budgets)

### Memory Considerations

- SAE activations are sparse (CSR), so memory is manageable
- Segmentation labels are uint8, one per patch
- For FishVista: 10 segmentation classes, ~4k images, 256 patches/image = ~1M patches
- For ADE20K: 150 segmentation classes, varies by split

### Batching Strategy

To avoid memory issues when computing AP for all (latent x seg_class) pairs:

1. **Batch over latents**: Process L latents at a time (e.g., L=1000)
2. **Batch over images** (optional): If labels don't fit in memory, process image batches

### Output Files

Save to `{run.inference}/{test_shards.name}/`:
- `audit_ap_s.npy`: Best-class AP for each SAE latent
- `audit_best_class_s.npy`: Which seg class gave best AP
- `audit_metrics.json`: Yield@B values, AUC_B, tau, budgets used
- `audit_feature_ranks.npy`: Feature indices sorted by classifier importance

### Self-Review Notes

**Issues to address:**

1. **EvalConfig is incomplete**: Need to add `slurm_acct`, `slurm_partition`, `n_hours`, `log_to`, `mem_gb`, `debug` fields for Slurm execution (copy from TrainConfig).

2. **Classifier checkpoint loading**: The checkpoint format from `train_cli` is JSON header + pickle. Need to handle both `SparseLinear` (LogisticRegression with `.coef_`) and `DecisionTree` (`.feature_importances_`).

3. **Feature ranking from DecisionTree**: `feature_importances_` only gives non-zero values for features actually used in splits. For unused features, importance is 0. Need to decide: rank all 0-importance features arbitrarily, or exclude them from Yield@B?

4. **Segmentation labels location**: RESOLVED - `labels.bin` is already stored in the shards directory (verified: 1071872 bytes = 4187 images x 256 patches for FishVista train).

5. **Background class handling**: Segmentation class 0 is background. Should we include background in AP computation? Probably not - exclude class 0 from the seg classes used for audit.

6. **Sparse column extraction**: `token_acts[:, j].toarray()` for sparse CSR is slow because CSR is row-major. Consider:
   - Converting to CSC format for column access
   - Or iterating over rows and accumulating (but this loses the vectorized AP benefit)

7. **Tie-breaking in rankings**: When multiple features have equal importance (common for DecisionTree), need deterministic tie-breaking (e.g., by feature index).

8. **Validation of classifier-shard compatibility**: The classifier was trained on specific shards. Need to verify that the test_shards used for audit match the test_shards used in training.

**Potential simplification:**

Since we only need AP for the top-B features (max B=1000), we could skip computing AP for all 16k+ SAE latents and instead only compute AP for the ranked top-1000 features. This would be much faster and more memory-efficient.
