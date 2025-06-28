# Trait Discovery

This folder contains all the stuff specific to our [trait discovery paper]() (doesn't exist yet, link doesn't go anywhere).

See [CONTRIBUTING.md](/contrib/trait_discovery/CONTRIBUTING.md) for more details.

## CUB-200-2011

First, you have to train/fit the baseline methods.

```sh
uv run train_baseline.py \
  --method random \
  --data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/900da851ddfb6085f76db3c7a75a62c2f6c4ee60ca64556cb6eefa47f7cd6c6e \
  --data.layer 23 \
  --sweep sweeps/train-baselines.toml \
  --slurm-acct PAS2136 \
  --slurm-partition nextgen \
  --n-hours 2
```

This fits a `baselines.RandomVector` on the ViT activations at layer 23.
It sweeps over all the configs in `sweeps/baselines.toml`.
It runs a slurm job for each config for 2 hours on the `nextgen` partition using the PAS2136 slurm account.

Then you can actually evaluate a baseline checkpoint:

```sh
uv run dump_cub200_scores.py \
  --ckpt checkpoints/random__n_prototypes-128__seed-17.bin \
  --train-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/900da851ddfb6085f76db3c7a75a62c2f6c4ee60ca64556cb6eefa47f7cd6c6e/ \
  --train-data.layer 23 \
  --test-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/9c29c95d5663c77b69069dc55bbb72e1de9ec0cbc9392f067d57b41f4b769980/ \
  --test-data.layer 23 \
  --cub-root /fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder \
  --n-train 32
```

This evaluates the random vector baseline at `checkpoints/random__n_prototypes-128__seed-17.bin`.
It uses the data in `--train-data.shard-root` to pick the best prototypes for each trait in CUB 200, using 32 random images from the training set, which is configured with `--cub-root`.
Then it uses the `--test-data.shard-root` to score each test image for each trait, and these scores are dumped to `--dump-to` (by default something like `./data`).

## Testing

```sh
uv run pytest tests/ \
  --cub-root /fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder/
```
