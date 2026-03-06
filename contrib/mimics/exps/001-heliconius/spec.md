# Mimic Pair Feature Discrimination

Find SAE features that maximally discriminate between mimic pairs in the Cambridge butterfly dataset.

## Background

The Cambridge Heliconius dataset contains 8 mimic pairs (erato subspecies, melpomene subspecies) that co-evolved similar wing patterns in the same geographic regions. The goal is to find individual SAE features (or small sets of features) that fire on one species in a pair but not the other, revealing what the SAE has learned about subtle morphological differences between mimics.

## Deliverables

### 1. cls_train sweep for 007_cambridge_butterflies

**File**: `contrib/trait_discovery/sweeps/007_cambridge_butterflies/cls_train.py`

Create a `make_cfgs()` sweep that runs L1-penalized logistic regression (SparseLinear) on each mimic pair using the 007 Cambridge infrastructure.

**Parameters**:
- **Shards**: v1.6 640-patch only (`/fs/scratch/PAS2136/samuelstevens/saev/shards/79239bdd`)
- **Layers**: 21 and 23
- **Pareto-optimal runs (v1.6, 640p)**:
  - Layer 21: `zhul9opa`, `gz2dikb3`, `3rqci2h1`, `r27w7pmf`, `x4n29kua`
  - Layer 23: `pnsi8yhe`, `onqqe859`, `rd8wc24d`, `vends70d`, `pa5cu0mf`
- **Aggregation**: MAX only (patches -> image)
- **Classifier**: SparseLinear with C in {0.001, 0.01, 0.1}
- **Tasks**: 8 binary LabelGrouping tasks, one per mimic pair:
  - `lativitta_vs_malleti`, `cyrbia_vs_cythera`, `notabilis_vs_plesseni`, `hydara_vs_melpomene`, `venus_vs_vulcanus`, `demophoon_vs_rosina`, `phyllis_vs_nanna`, `erato_vs_thelxiopeia`
- **Label format**: `source_col="subspecies"` with bare subspecies names. v1.6 labels.csv has columns `stem,subspecies` with values like `"lativitta"`, `"malleti"`, etc. Groups use these bare names:
  ```python
  groups={"erato": ["lativitta"], "melpomene": ["malleti"]}
  ```
- **Hybrids**: Excluded automatically because hybrid labels like `"malleti x bellula"` won't match the exact bare name `"malleti"` in `apply_grouping`'s exact string matching.
- **Train/test shards**: Same directory (same as 005 pattern). This means accuracy numbers are on training data, not generalization metrics. This is acceptable because the goal is feature discovery (which features separate these species), not building a generalizable classifier. The notebook will document this.

**Total configs**: 10 runs x 3 C values x 8 pairs = 240 jobs. Each is CPU-only logistic regression on a few hundred to a few thousand images with ~16k features, so very fast (1-5 min each).

**Known data limitations in v1.6** (from actual label counts):
- `phyllis=0`, `nanna=0` (completely missing, jobs will produce empty results)
- `demophoon=2`, `thelxiopeia=2` (near-degenerate)
- `erato=23`, `rosina=120` (small)
- Viable pairs: lativitta/malleti, cyrbia/cythera, notabilis/plesseni, hydara/melpomene, venus/vulcanus

The sweep runs all 8 pairs. The notebook flags insufficient-data pairs.

Design so v1.7 shards and runs can be added later by just appending another dict.

### 2. Rename existing notebook

Rename `contrib/trait_discovery/notebooks/007_cambridge.py` to `contrib/trait_discovery/notebooks/007_cambridge_sae.py`.

### 3. Marimo analysis notebook

**File**: `contrib/trait_discovery/notebooks/007_cambridge_mimicry.py`

A marimo notebook that loads cls_train results and produces three main outputs:

**Part A: Sparsity-accuracy tradeoff**

For each mimic pair, plot balanced accuracy vs number of nonzero features as C varies. Show raw accuracy as secondary. Include majority-class baseline for each pair. Aggregate across the 10 SAE runs (mean +/- std, or show all points).

Flag pairs with insufficient data (< 50 samples in either class).

Key question answered: "How many SAE features do you need to tell these mimics apart?"

**Part B: Feature ranking table**

For each mimic pair, extract the top-K features with nonzero weights from the best L1 model (highest balanced accuracy). Show:
- Feature index (SAE latent ID)
- Weight magnitude and sign (sign interpretation: sklearn binary coef_ is for class 1; class ordering is lexicographic from `apply_grouping`, so "erato"=0, "melpomene"=1; positive weight = feature favors melpomene)
- Parse SAE run ID and layer from the output file path (since `extract_feature_ranking()` only returns indices and scores)

**Part C: Cross-pair comparison**

Summary table: for each pair, show (a) best balanced accuracy, (b) number of features needed for >90% balanced accuracy, (c) best single feature and its balanced accuracy, (d) majority-class baseline accuracy, (e) sample counts per class. Rank pairs by difficulty.

**Hybrid handling**: Excluded from classification by exact label matching. In future visualization work, show where hybrids land in feature space as a sanity check.

## Implementation notes

- The notebook loads cls_train output files: JSON header with `cfg`, `test_acc`, `n_classes`, `class_names`, plus pickled `classifier`, `test_pred`, `test_y`
- Raw `test_acc` in the header is overall accuracy. Balanced accuracy must be recomputed from saved `test_pred` and `test_y` in the notebook.
- Use `extract_feature_ranking()` from `tdiscovery.classification` to get ranked features from trained classifiers
- The notebook should be version-agnostic: parameterize shard versions so v1.7 results slot in when ready
- This is analysis only; no training happens in the notebook

## Non-goals (for now)

- Visualization of feature activations on images (heatmaps, overlays). Deferred to later.
- probe1d (that's for patch-level labels; we only have image-level species labels here)
- Decision trees (keeping it focused on L1 logistic regression)
- 384-patch configurations (640-patch only)
- MEAN aggregation (MAX only)
- Cross-validation or separate train/test splits (feature discovery, not generalization)
