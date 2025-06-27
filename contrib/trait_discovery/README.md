# Trait Discovery

This folder contains all the stuff specific to our [trait discovery work]() (doesn't exist yet).

## CUB-200-2011

First, you have to train/fit the baseline methods.

```sh
uv run fit_baselines.py \
  --method random \
  --data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/900da851ddfb6085f76db3c7a75a62c2f6c4ee60ca64556cb6eefa47f7cd6c6e \
  --data.layer 23 \
  --sweep sweeps/baselines.toml \
  --slurm-acct PAS2136 \
  --slurm-partition nextgen \
  --n-hours 2
```

This fits a `baselines.RandomVector` on the ViT activations at layer 23.
It sweeps over all the configs in `sweeps/baselines.toml`.
It runs a slurm job for each config for 2 hours on the `nextgen` partition using the PAS2136 slurm account.

## Testing

```sh
uv run pytest tests/ \
  --cub-root /fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder/
```
