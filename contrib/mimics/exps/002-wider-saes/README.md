# 002-wider-saes

Train 32K-latent SAEs on Cambridge butterflies (384p, v1.6, DINOv3 ViT-L/16).
Goal: more latents for finer-grained mimic pair discrimination.

## Train

```bash
uv run launch.py train --sweep contrib/mimics/exps/002-wider-saes/train.py --slurm-acct PAS2136 --slurm-partition nextgen --max-parallel 4
```

40 configs: 2 layers (21, 23) x 4 k values (16, 32, 64, 128) x 5 learning rates.
TopK + AuxK + Matryoshka, datapoint init, 100M tokens, 8h each on nextgen (4 SAEs/GPU).

## After training

1. Pick Pareto-optimal runs (NMSE vs L0) from WandB tag `mimics-32k-384p-v1.6`.
2. Run inference to get `token_acts.npz`.
3. Score all latents and browse in the triage notebook.
