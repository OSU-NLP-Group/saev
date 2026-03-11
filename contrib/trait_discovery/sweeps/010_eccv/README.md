# 010_eccv

This folder is for rerunning selected experiments for the ECCV 2026 submission.

Current focus:

- backfill missing ImageNet-1K validation NMSE metrics for historical ReLU SAE runs
- avoid writing large inference artifacts when only `metrics.json` is needed

## Sweep files

- `relu_nmse.py`: runs SAE inference on IN1K val (`3e27794f`) for the selected ReLU runs with `save: False`, so inference writes `metrics.json` without `token_acts.npz`.
- `matryoshka_relu_nmse.py`: same as above for the selected Matryoshka ReLU runs.
- `matryoshka_topk_nmse.py`: same as above for the selected Matryoshka TopK runs.
- `baselines_nmse.py`: runs baseline inference on IN1K val (`3e27794f`) for `kmeans`, `pca`, and `semi_nmf` figure run IDs with `save: False`, writing metrics only.

## Job status (updated 2026-03-01 10:46 EST)

- [ ] `3827766_[0-13%14]` (`relu_nmse.py`, `--force-recompute`)
  - Goal: regenerate IN1K val `metrics.json` for the 14 ReLU runs with the new schema (`mse_per_dim`, `mse_per_token`, `normalized_mse`, `baseline_mse_per_dim`, `baseline_mse_per_token`, `sse_recon`, `sse_baseline`, `n_tokens`, `d_model`, `n_elements`).
  - Current state: running/pending.

- [ ] `3827767_[0-15%16]` (`matryoshka_relu_nmse.py`, `--force-recompute`)
  - Goal: regenerate IN1K val `metrics.json` for the 16 Matryoshka ReLU runs used in `figures.py`.
  - Current state: running/pending.

- [ ] `3827768_[0-2%3]` (`matryoshka_topk_nmse.py`, `--force-recompute`)
  - Goal: regenerate IN1K val `metrics.json` for TopK runs `flqkcam7`, `s3pqewz1`, `l8hooa3r`.
  - Current state: pending.

- [ ] `3827769_[0-12%13]` (`baselines_nmse.py`, `--force`)
  - Goal: regenerate IN1K val `metrics.json` for 13 baseline figure run IDs (`kmeans + pca + semi_nmf`) using the same schema.
  - Current state: pending.

- [ ] `3837777_[0-76%77]` (`matryoshka_relu_layers_nmse.py`, `--force-recompute`)
  - Goal: backfill IN1K val `metrics.json` for 77 Matryoshka ReLU layer-comparison frontier runs in `figures.py` (layers 14/16/18/20/22), disk metrics only.
  - Current state: pending.
## Example command

```bash
uv run python launch.py inference \
  --sweep contrib/trait_discovery/sweeps/010_eccv/relu_nmse.py \
  --slurm-acct PAS2136 \
  --slurm-partition nextgen
```

```bash
uv run python launch.py inference \
  --sweep contrib/trait_discovery/sweeps/010_eccv/matryoshka_relu_nmse.py \
  --slurm-acct PAS2136 \
  --slurm-partition nextgen
```

```bash
uv run python launch.py inference \
  --sweep contrib/trait_discovery/sweeps/010_eccv/matryoshka_topk_nmse.py \
  --slurm-acct PAS2136 \
  --slurm-partition nextgen
```
