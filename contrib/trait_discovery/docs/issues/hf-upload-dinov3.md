# Upload DINOv3 SAE Checkpoints to Hugging Face

## Overview

Upload Pareto-optimal DINOv3 ImageNet SAE checkpoints to Hugging Face, organized as multi-checkpoint repos with one repo per (model_size, activation_type) combination.

## What Gets Uploaded

### ReLU Matryoshka SAEs (3 repos)

| Model | Layers | d_model | d_sae | Checkpoints per layer | Total |
|-------|--------|---------|-------|-----------------------|-------|
| ViT-S/16 | 6, 7, 8, 9, 10, 11 | 384 | 16384 | up to 6 | up to 36 |
| ViT-B/16 | 6, 7, 8, 9, 10, 11 | 768 | 16384 | up to 6 | up to 36 |
| ViT-L/16 | 13, 15, 17, 19, 21, 23 | 1024 | 16384 | up to 6 | up to 36 |

All use Matryoshka objective with ReLU activation + L1 sparsity.

### TopK Matryoshka SAEs (1 repo)

| Model | Layers | d_model | d_sae | Checkpoints per layer | Total |
|-------|--------|---------|-------|-----------------------|-------|
| ViT-L/16 | 13, 15, 17, 19, 21, 23 | 1024 | 16384 | up to 6 | up to 36 |

The 003_auxk sweep has 6 runs per layer (k in {16, 64, 256} x LR grid) across all 6 layers = 36 ImageNet runs total. We select up to 6 per layer from the Pareto frontier, same as the ReLU ones.

TopK activation with AuxK auxiliary loss, Matryoshka objective.

## HF Repo Naming

- `osunlp/SAE_DINOv3_ViT-S-16_IN1K`
- `osunlp/SAE_DINOv3_ViT-B-16_IN1K`
- `osunlp/SAE_DINOv3_ViT-L-16_IN1K`
- `osunlp/SAE_DINOv3_TopK_ViT-L-16_IN1K`

New HF collection: "Towards Open-Ended Visual Scientific Discovery with Sparse Autoencoders" linking all 4 repos.

## Directory Layout Within Each Repo

```
layer_<N>/
  <run-id>/sae.pt
  <run-id>/sae.pt
  <run-id>/sae.pt
  <run-id>/sae.pt
  <run-id>/sae.pt
README.md
```

Users download individual checkpoints via:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("osunlp/SAE_DINOv3_ViT-L-16_IN1K", "layer_23/<run-id>/sae.pt")
```

Or download everything:
```python
from huggingface_hub import snapshot_download
snapshot_download("osunlp/SAE_DINOv3_ViT-L-16_IN1K")
```

## Checkpoint Selection: Pareto Frontier + Log-Spaced L0

For each (model, layer):

1. Read ALL run IDs (including commented-out ones) from existing sweep files. Commented IDs indicate runs that have already completed inference/probing; they are valid candidates.
2. Filter by objective type (must have `n_prefixes` in W&B config for Matryoshka) and activation type to avoid mixing vanilla/Matryoshka or ReLU/TopK runs.
3. Query W&B API for each run's `summary/eval/l0` and `summary/eval/normalized_mse`.
4. Sort by L0 ascending.
5. Always include the sparsest and densest endpoints. Fill interior with log-L0 quantiles to get up to 6 total. Tie-break by lower NMSE, then later run date.
6. Preflight: load every selected checkpoint with `saev.nn.load()` and verify it loads without error.

For TopK ViT-L, use the `in1k_run_ids` from `003_auxk/inference.py` (36 runs across 6 layers, 6 per layer). Same selection process: query W&B, compute Pareto frontier, pick up to 6 log-spaced by L0.

## Run ID Sources

| Model | Sweep file with Pareto run IDs |
|-------|-------------------------------|
| ViT-S | `contrib/trait_discovery/sweeps/vits_ade20k_inference.py` |
| ViT-B | `contrib/trait_discovery/sweeps/vitb_ade20k_inference.py` |
| ViT-L (ReLU) | `contrib/trait_discovery/sweeps/vitl_ade20k_inference.py` |
| ViT-L (TopK) | `contrib/trait_discovery/sweeps/003_auxk/inference.py` |

Checkpoint paths: `/fs/ess/PAS2136/samuelstevens/saev/runs/<run_id>/checkpoint/sae.pt`

## Model Card (README.md per repo)

Each repo gets a README.md with:

1. YAML frontmatter (`license: mit`)
2. Title and description
3. Links (homepage, code, preprint, demos, contact)
4. Run-id-to-metrics table mapping each `<run-id>` directory to its layer, fractional L0, and NMSE so users can pick the right checkpoint
5. Usage example showing `hf_hub_download` + `saev.nn.load()`

Example table:

| Run ID | Layer | L0 | NMSE | Path |
|--------|-------|----|------|------|
| abc123 | 23 | 31.7 | 0.042 | `layer_23/abc123/sae.pt` |
| def456 | 23 | 128.3 | 0.018 | `layer_23/def456/sae.pt` |
| ... | ... | ... | ... | ... |

## Upload Script

New script at `contrib/trait_discovery/scripts/push_dinov3.py`.

### Responsibilities

1. Parse run IDs from sweep files (or accept them as config).
2. Query W&B API for L0/MSE metrics per run.
3. For each (model, layer) group, sort by L0 and select up to 5 evenly log-spaced.
4. Copy/symlink `sae.pt` files into a staging directory with the target directory layout.
5. Generate README.md with the checkpoint table.
6. Generate `manifest.jsonl` at repo root with machine-readable metadata per checkpoint (run_id, layer, L0, NMSE, activation config, sha256) for programmatic access.
7. Upload each repo to HF using `huggingface_hub.HfApi.upload_folder()`.
8. Create the new HF collection and add repos to it.
9. Post-upload smoke test: `hf_hub_download` one checkpoint per repo, load it with `saev.nn.load()`.

### Interface

```
HF_TOKEN=<token> uv run contrib/trait_discovery/scripts/push_dinov3.py \
  [--dry-run]  # show what would be uploaded without actually uploading
```

Uses `HF_TOKEN` env var (not CLI arg) to avoid leaking token into shell history.

### Steps (implementation order)

1. **Data gathering**: Extract run IDs from sweep files, query W&B for metrics.
2. **Selection**: Compute the 5 log-spaced checkpoints per (model, layer).
3. **Staging**: Build the directory tree in a temp dir, copying sae.pt files.
4. **Model card generation**: Render README.md from the metrics table.
5. **Upload**: Push to HF, create collection, run smoke test.
6. **Docs update**: Update `docs/src/users/inference.md` with nested-path `hf_hub_download` examples for the new multi-checkpoint repos.

## Resolved

- **Schema-4 `seed` bug**: Fixed in `ea10c0f`. `_normalize_cfg_kwargs` now strips `seed` from checkpoint headers. Test added.
- **Safetensors conversion**: No. The custom binary format (JSON header + torch.save) must be preserved because the JSON header is essential for loading.
- **Commented-out run IDs**: All commented IDs in sweep files are valid (they indicate already-completed inference/probing runs). The script will parse both commented and uncommented IDs.
- **Run type filtering**: The script must filter by W&B config (objective type, activation type) to avoid mixing vanilla/Matryoshka or ReLU/TopK runs from shared sweep files.
- **SAE-V collection**: Yes, also add repos to the existing `osunlp/sae-v` collection.
