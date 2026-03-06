# Cambridge Mimicry: Feature Validation Spec

## Overview

A single marimo notebook that lets you triage and visually validate SAE features selected by sparse linear classifiers for Heliconius mimic pair discrimination. The notebook handles filtering, rendering, and browsing in one place.

## Inputs

User provides a **task name** (e.g. `lativitta_dorsal_vs_malleti_dorsal`). The task name encodes the subspecies pair and view: `{erato_ssp}_{view}_vs_{melp_ssp}_{view}`. The notebook parses this to construct the `LabelGrouping` (source_col=`subspecies_view`, groups mapping `erato` -> `["{erato_ssp}_{view}"]`, `melpomene` -> `["{melp_ssp}_{view}"]`).

The notebook also needs:
- `run_root`: path to `runs/` directory (e.g. `/fs/ess/PAS2136/samuelstevens/saev/runs`).
- `run_ids`: list of SAE run IDs to search across (e.g. the 5 layer-21 runs).
- `shards_dpath`: path to the image shards (e.g. `/fs/scratch/PAS2136/samuelstevens/saev/shards/79239bdd`).

Data sources:
- Classifier pkl files: `runs/{run_id}/inference/{shard_id}/cls_{task_name}_max_C{c}.pkl`. The pkl contains a JSON header with `test_acc` and a cloudpickle payload with the sklearn `LogisticRegression` object, `test_pred`, and `test_y`. Feature indices are extracted via `np.where(np.any(classifier.coef_ != 0, axis=0))[0]`, weights from the corresponding `coef_` entries. Balanced accuracy must be computed from `test_pred` and `test_y`.
- SAE activations: `runs/{run_id}/inference/{shard_id}/token_acts.npz` (scipy sparse CSR matrix, shape [n_tokens, d_sae]).
- Image labels: loaded via `tdiscovery.classification.load_image_labels(shards_dpath)`, which reconstructs the dataset in shard order and extracts the `subspecies_view` column. Do NOT read `labels.csv` directly; row order in the CSV does not match dataset/token order.
- Image data: loaded via `saev.data.shards` dataset, which reads shard binaries and metadata.
- Shard metadata: `shards/{shard_id}/metadata.json` for `content_tokens_per_example`, model family, checkpoint, etc. Note: `patch_size` is NOT in shard metadata. It comes from the ViT model object (`vit.patch_size`), loaded via `saev.data.models.load_model_cls(md.family)(md.ckpt)`.

The `shard_id` is the name of the shards directory (e.g. `79239bdd`). It appears both in the shards path and as a subdirectory under `runs/{run_id}/inference/`.

## Workflow

### Step 1: Filter classifiers

Load all classifier pkl files for the given task across all specified run_ids and C values. Display a Polars DataFrame with columns: run_id, C, n_features, balanced_acc. User filters by sparsity range (e.g. 1-30 features) and sorts by balanced_acc to pick the top-K checkpoints.

### Step 2: Collect features

From selected classifiers, collect the union of (run_id, feature_id, weight, favored_class) tuples. Pool features across all selected runs. Display as a Polars DataFrame with columns: run_id, feature_id, weight, favors (erato/melpomene), abs_weight.

Features appear once per run (if feature 48 is selected by multiple C values in the same run, deduplicate by keeping the entry from the highest-accuracy classifier). Same feature index in different runs is treated as distinct.

### Step 3: Render per-class images

For each (run_id, feature_id) pair, render highlighted images split by class. The notebook does this itself (not a separate script), but checks disk for already-rendered images before re-rendering.

**Processing order:** Process one run at a time. Load the run's `token_acts.npz`, convert to CSC format (`token_acts.tocsc()`), extract all features for that run in one batch (`token_acts_csc[:, feature_ids]`), then free the matrix before moving to the next run. This avoids O(nnz) per-column scans (CSR column slicing) and keeps memory bounded to one run's data at a time.

**Rendering logic per run:**
1. Load `token_acts.npz` for the run. Convert CSR to CSC. Extract the columns for all features needed from this run as a dense `(n_tokens, n_features)` array in one operation.
2. Load image labels via `load_image_labels(shards_dpath)`, apply `LabelGrouping` to get class assignments (erato vs melpomene) per image.
3. For each feature:
   a. Reshape feature column from `[n_tokens]` to `[n_images, tokens_per_image]`. This gives the per-patch activation grid needed for both ranking and rendering.
   b. Compute max-patch activation per image: `max(axis=1)`. Use this for ranking.
   c. Split images by class.
   d. Per class, sort by max-patch activation descending. Take top-N (highest activation) and bottom-N (lowest non-zero activation). If fewer than N non-zero images exist in a class, take all available. This is expected for highly discriminative features that fire predominantly on one class.
   e. For each selected image, render the SAE highlighted image using `saev.viz.add_highlights` with the per-patch activation vector from step (a). This avoids a second sparse matrix access.

**Output directory:**
```
runs/{run_id}/inference/{shard_id}/images/{feature_id}/cambridge-mimics/{subspecies_view}/
    0_sae_img.png          # top (highest non-zero)
    1_sae_img.png
    ...
    bottom/
        0_sae_img.png      # bottom (lowest non-zero)
        ...
```

The `{subspecies_view}` in the path is the fine-grained label (e.g. `lativitta_dorsal`), not the class group name (`erato`). The browsing step groups these by class for display.

N = 5-10 images per class per direction (configurable, default 8).

**Cache check:** Before rendering a feature, check for a sentinel file `images/{feature_id}/cambridge-mimics/_done.json`. The sentinel records `{"n_per_class": N, "task": "...", "timestamp": "..."}`. If the sentinel exists and `n_per_class` matches the current N, skip rendering. On render completion, write the sentinel as the last step. This handles partial renders (no sentinel = incomplete) and N changes (sentinel N mismatch = re-render).

### Step 4: Browse

Display all features in a scrollable view. Step 4 only reads images from disk (written by Step 3). No in-memory image passing between steps.

Each feature gets a row showing:
- Feature metadata: run_id, feature_id, weight, favored class.
- Per-class image strips: for each class (erato, melpomene), show the top-N highlighted images in a horizontal row. Each image annotated with its max-patch activation value and subspecies_view label (e.g. `lativitta_dorsal`).

Bottom-N images shown below the top-N strip for each class.

The view pools features from all selected runs for the given task. Features can be sorted by abs(weight), run_id, or feature_id.

## Non-goals (v1)

- Discrimination stats (per-class activation rate, single-feature AUC, histograms). Can add later.
- Spatial average heatmaps across specimens.
- Cross-run feature alignment by activation similarity.
- Annotation export (JSON/CSV of validated features).
- Feature interaction plots.
- Cache invalidation on upstream data changes (re-inference). Delete `_done.json` manually if needed.

## Implementation notes

- Single marimo notebook: `contrib/cambridge_mimicry/notebooks/validate.py`.
- Use existing `saev.viz.add_highlights` for rendering highlighted images. This function takes `(img, tokens, patch_size, upper=...)`. The `upper` parameter controls highlight scaling; for cross-image comparability within a feature, use the same `upper` for all images of that feature (e.g. the global max activation across both classes).
- Use existing `saev.data.shards` for loading image data.
- Use `tdiscovery.classification.load_image_labels` for label loading (correct dataset order). Use `LabelGrouping` for class assignment, parsed from the task name.
- `patch_size` comes from the ViT model, not shard metadata. Load via `saev.data.models.load_model_cls(md.family)(md.ckpt).patch_size`. At img_scale=1.0, just use `vit.patch_size` directly.
- Load `token_acts.npz` as scipy sparse CSR, convert to CSC for column access, batch-extract all features for a run in one slice. CSC column slicing is ~15x faster than CSR for this matrix shape (confirmed by benchmark: ~5.8s vs ~88s for 200 features).
- Images rendered at original resolution (img_scale=1.0). No segmentation overlays.
- No GPU needed. token_acts.npz is precomputed. Image rendering is CPU-only (PIL). PNG writing dominates render time (~0.08s/image).
- Memory budget: one run's dense feature matrix at a time. For 2000 images x 196 patches x 60 features (generous upper bound per run), that's ~94 MB. Full CSC matrix for a top-k=16 run is ~0.59 GB; evict after extracting needed columns.
- Marimo reactivity caveat: gate the rendering cell so it only runs on explicit trigger (e.g. button click), not on every upstream change. Otherwise filter changes will re-render all images.
