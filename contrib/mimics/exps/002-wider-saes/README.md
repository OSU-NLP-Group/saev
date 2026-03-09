# 002-wider-saes

Train 16K and 32K-latent SAEs on Cambridge butterflies (384p, v1.6, DINOv3 ViT-L/16).
Goal: more latents for finer-grained mimic pair discrimination.

## Train

```bash
uv run launch.py train --sweep contrib/mimics/exps/002-wider-saes/train_16k.py --slurm-acct PAS2136 --slurm-partition nextgen --max-parallel 4
uv run launch.py train --sweep contrib/mimics/exps/002-wider-saes/train.py --slurm-acct PAS2136 --slurm-partition nextgen --max-parallel 4
```

Each sweep is 40 configs: 2 layers (21, 23) x 4 k values (16, 32, 64, 128) x 5 learning rates.
TopK + AuxK + Matryoshka, datapoint init, 100M tokens, 8h each on nextgen.
WandB tags: `mimics-16k-384p-v1.6`, `mimics-32k-384p-v1.6`.

## Pareto selection

`notebook.py` pulls runs from WandB, marks Pareto-optimal runs per (layer, d_sae) by sorting on L0 and keeping cumulative-min NMSE. 16 Pareto runs total (4 k values x 2 widths x 2 layers). All best LRs landed interior to the sweep range [1e-4, 1e-2].

## Inference

```bash
uv run launch.py inference --sweep contrib/mimics/exps/002-wider-saes/inference.py --slurm-acct PAS2136 --slurm-partition nextgen --n-hours 4
```

Produces `token_acts.npz` (sparse CSR, shape [n_tokens, d_sae]) per run. ~45 min each on nextgen.

## Scoring

Score every latent in every Pareto-optimal run for every mimic pair task. The goal is a parquet file per run that lets us triage latents without rendering images.

### Input

Per run: `token_acts.npz` (sparse CSR, [n_images * tokens_per_image, d_sae]). Max-pool over patches to get image-level activation per latent: `max_acts[i, j] = max over patches of token_acts for image i, latent j`.

### Scores

All scores are computed per (latent, task) pair, where a task is a binary mimic pair (erato subspecies vs melpomene subspecies). Scores are defined in `contrib/mimics/src/mimics/consistency.py`.

Task-level scores (one value per latent per task):

- auroc: ROC AUC using max-patch activation as the predictor and binary erato/melpomene label as the target. Measures raw discriminative power of a single latent.
- selectivity: `clip(2 * |auroc - 0.5|, 0, 1)`. Rescales AUROC so 0 = chance, 1 = perfect separation, regardless of which class the latent favors.
- support_overall: fraction of task images (both classes) where max-patch activation > 0. A latent that never fires is useless.

Per-class scores (one value per latent per task per class, stored as `{score}_{class}`):

- support: fraction of class images where max-patch activation > 0.
- strength: `clip(1 - CV(top_k_activations), 0, 1)` where CV is the coefficient of variation of the top-k non-zero activation values. High strength means the latent fires at a consistent magnitude on its top images, not just on one outlier.
- topk_stability: mean pairwise Jaccard similarity of top-k image sets across bootstrap resamples. High stability means the same images consistently rank highest; low stability means the ranking is fragile and depends on which images happen to be sampled.
- n_nonzero: count of class images with any non-zero activation.


### Output

One parquet file per run at `runs/{run_id}/inference/{shard_id}/cambridge-mimics.parquet`. Columns: `run_id, task, feature_id, selectivity, auroc, support_overall, support_{class}, topk_stability_{class}, strength_{class}, n_nonzero_{class}`.

For a 32K SAE with 5 viable tasks, that's ~160K rows per run (32768 latents x 5 tasks). At ~14 float columns, roughly 160K x 14 x 8 bytes = 18 MB uncompressed, much smaller with parquet compression.

### Triage workflow

1. Score all latents across all tasks.
2. Filter by `consistency > threshold` and `selectivity > threshold`.
3. For surviving latents, render per-class image grids (top/bottom activating images with SAE heatmap overlays).
4. Browse in a marimo viewer notebook.
