# Issue: Keep Top-K Activations Small

## 0) Problem Statement (TL;DR)

`src/saev/scripts/activations.py` currently materializes a `(d_sae, top_k, n_patches_per_img)` tensor (`top_values_p`) to keep per-feature heatmaps for downstream visuals. With `d_sae=16_384`, `top_k=128`, and `n_patches_per_img=1_920`, this buffer alone is ~15.7 GB in float32 (or 7.8 GB in float16) and causes OOM on both GPU and host RAM. We can keep the same qualitative outputs while shrinking the footprint to a few megabytes by storing only the top activation score and the corresponding `(image_i, patch_i)` for each slot, then reconstructing the full patch grid later by re-running the SAE on demand.


## 1) Scope & Constraints

* Applies to the unified activation writer (`src/saev/scripts/activations.py`) and the trait-discovery visualiser (`contrib/trait_discovery/scripts/visuals.py`). All other tooling can adapt after this change or via a follow-up.
* Maintain one-pass processing over the dataset during extraction; no extra per-image SAE passes while writing.
* Visualisation can afford to re-run the SAE for dozens of images per latent; batching via the indexed activations dataset is (probably) acceptable.
* Preserve existing numerics for sparsity, mean activations, percentile estimation, and distributions.
* Output file format may change (e.g., new filenames) as long as it is well-documented in the spec; downstream scripts should be updated accordingly.


## 2) Desired Outputs After Extraction

For each SAE feature and top-k slot we need to persist:

* `top_values_sk`: shape `(d_sae, top_k)`, float32, storing the activation score that ranked the slot. Initialise to `-inf` on CPU and keep updates CPU-side.
* `top_img_i_sk`: shape `(d_sae, top_k)`, int64, the global image index.

All other statistics already emitted by `dump_activations` stay as-is.

## 3) Changes in `src/saev/scripts/activations.py`

1. Drop the giant `top_values_p` initialisation. Introduce three tensors on CPU: `top_values_sk` (fill with `-torch.inf`) and `top_img_i_sk` (zeros, dtype `torch.int64`).
2. When we flatten `sae_acts_sb` to `[d_sae, total_patches]`, call `torch.topk(..., dim=1)` once to obtain both top candidate scores and flattened indices. No `gather_batched` of full patch grids.
3. Derive candidate image indices: `candidate_img = i_im[flat_idx // n_patches]`.
4. Merge candidates with existing top-k buffers: concatenate current scores with candidate scores along dim=1, call `torch.topk` on the concatenated scores to keep the best `cfg.top_k`, and use the resulting indices to gather from the concatenated `(scores, img_i)` tensors. Keep all arrays on CPU. The logic mirrors the current gather of dense patch grids but with tiny tensors.
5. Keep the computation of `img_acts_ns` (mean of top-3 patches) using `values_p` since it already depends on the reshaped per-image activations; this can stay unchanged.
6. Save artifacts: write `top_values_sk` to `cfg.top_values_fpath` and `top_img_i_sk` to `cfg.top_img_i_fpath` as before.
7. Confirm dtype choices (`float32` for scores, `int64` for indices) and device placements (CPU only) to avoid GPU memory pressure.


## 4) Changes in `contrib/trait_discovery/scripts/visuals.py`

1. Adjust loading logic in `main` to expect `top_values.pt` and `top_img_i.pt`.
2. Construct an indexed activations dataset for replay (via `saev.data.indexed.Dataset`). Use the stored `metadata` and shard root to instantiate it; reuse configuration for patch geometry.
3. Load the SAE checkpoint lazily (only once) and move it to the configured device (`cuda` if available, else CPU`).
4. When iterating features, build the candidate image list from `top_img_i[feature]`. For each unique `img_i`:
   * Query the indexed dataset for all patches of that image (or use a helper to fetch the full `[n_patches_per_img, d_vit]` tensor).
   * Run the SAE forward pass (`sae(vit_acts)`), obtain the sparse activations, and cache them in a dictionary keyed by `img_i` to reuse across multiple features in the same run.
5. Reconstruct the patch heatmap for each (feature, image) pair by taking the cached activations, reshaping to `[n_patches_per_img]`, and using the stored `patch_i` as the argmax location for ordering. The full `patch` grid comes directly from the SAE output, so no dense tensor was ever saved.
6. When building the `Example` list for visualisation, replace `patches=values_p` with the freshly computed per-image activation vector. Preserve the existing upper-bound scaling logic by using the cached activations to compute max values.
7. Ensure batching or caching keeps replay work lightweight: shapes are small, but note in code comments that repeated SAE inference is acceptable and should stay on CPU if GPUs are unavailable.


## 5) Validation Plan

* Run the unified activations script on a small shard (e.g., a handful of images) and verify the produced tensors:
  * Shapes match `(d_sae, top_k)` and contain expected indices.
  * `top_values` monotonically decrease along dim=1.
* Execute the visual script on the same small shard; confirm it regenerates highlight images identical (within tolerance) to the previous pipeline by eyeballing a sample latent.
* Optional regression: add an assertion in the visual script that re-derived `top_values` from the replayed activations match the saved `top_values` for the top slot to catch mismatches early.


## 6) Follow-ups / Open Questions

* Other tooling (e.g., `contrib/interactive_interp`) may still load `top_values.pt` expecting the old shape. File a follow-up issue.
* Consider storing `top_values` as float16 if disk usage becomes an issue; for now stick to float32 for parity.
