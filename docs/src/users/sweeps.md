# Sweeps

Hyperparameter sweeps in `saev` train multiple SAE configurations in parallel on a single GPU, amortizing the cost of loading activation data from disk across all models.
Furthermore, sweeps make it easy to train multiple SAEs with one command across multiple GPUs using Slurm.

## Quick Start

Create a Python file defining your sweep:

```python
# sweeps/my_sweep.py

def make_cfgs() -> list[dict]:
    cfgs = []

    # Grid search over learning rate and sparsity
    for lr in [3e-4, 1e-3, 3e-3]:
        for sparsity in [4e-4, 8e-4, 1.6e-3]:
            cfg = {
                "lr": lr,
                "objective": {"sparsity_coeff": sparsity},
            }
            cfgs.append(cfg)

    return cfgs
```

Run the sweep:

```bash
uv run train.py --sweep sweeps/my_sweep.py \
  --train-data.layer 23 \
  --val-data.layer 23
```

This trains 9 SAEs (3 learning rates x 3 sparsity coefficients) in parallel.

## Why Parallel Sweeps?

SAE training is bottlenecked by disk I/O, not GPU computation. Loading terabytes of pre-computed ViT activations from disk is the slowest part. By training multiple SAE configurations on the same batch simultaneously, we amortize the I/O cost:

```
┌────────────────────────┐
│ ViT Activations (disk) │
└───────────┬────────────┘
            │ (slow I/O, once per batch)
            ▼
      ┌──────────┐
      │  Batch   │
      └─────┬────┘
            ├─────────┬─────────┬─────────┐
            ▼         ▼         ▼         ▼
         SAE #1    SAE #2    SAE #3     ...
        (lr=3e-4) (lr=1e-3) (lr=3e-3)
```

## Sweep Configuration

### Python-Based Sweeps

Python sweeps give you full control over config generation. Your sweep file must define a `make_cfgs()` function that returns a list of dicts.

**Grid search example:**

```python
def make_cfgs():
    cfgs = []

    for lr in [1e-4, 3e-4, 1e-3]:
        for d_sae in [8192, 16384, 32768]:
            cfg = {
                "lr": lr,
                "sae": {"d_sae": d_sae},
            }
            cfgs.append(cfg)

    return cfgs
```

**Paired parameters (not a grid):**

```python
def make_cfgs():
    cfgs = []

    # Grid over lr x sparsity
    for lr in [3e-4, 1e-3, 3e-3]:
        for sparsity in [4e-4, 8e-4, 1.6e-3]:
            # Paired layers (train and val use same layer)
            for layer in [6, 7, 8, 9, 10, 11]:
                cfg = {
                    "lr": lr,
                    "objective": {"sparsity_coeff": sparsity},
                    "train_data": {"layer": layer},
                    "val_data": {"layer": layer},
                }
                cfgs.append(cfg)

    return cfgs
```

This generates 54 configs (3 x 3 x 6) where each train/val pair uses the same layer, avoiding the 162 configs you'd get from a full grid (3 x 3 x 6 x 6).

**Conditional sweeps:**

```python
def make_cfgs():
    cfgs = []

    for d_sae in [8192, 16384, 32768]:
        # Use different LR for different SAE widths
        lrs = [1e-3, 3e-3] if d_sae <= 16384 else [3e-4, 1e-3]

        for lr in lrs:
            cfg = {
                "lr": lr,
                "sae": {"d_sae": d_sae},
            }
            cfgs.append(cfg)

    return cfgs
```

## Command-Line Overrides

Command-line arguments override sweep parameters with deep merging. The precedence order is: **CLI > Sweep > Default**.

```bash
uv run train.py --sweep sweeps/my_sweep.py \
  --lr 5e-4  # Overrides all LRs in the sweep
```

Override nested config fields with dotted notation:

```bash
uv run train.py --sweep sweeps/my_sweep.py \
  --train-data.layer 23 \
  --val-data.layer 23 \
  --sae.d-sae 16384
```

Deep merging means that when you override a nested field, only that specific field is replaced—other fields in the nested config are preserved from the sweep or default values.

## Parallel Groups

Not all parameters can vary within a parallel sweep. Parameters that affect data loading (like `train_data`, `n_train`, `device`) must be identical across all configs in a parallel group.

When configs differ in these parameters, they're automatically split into separate Slurm jobs:

```python
def make_cfgs():
    cfgs = []

    # These will run in 2 separate jobs
    for layer in [6, 12]:  # Different data loading
        for lr in [1e-4, 3e-4]:  # Can parallelize
            cfg = {
                "lr": lr,
                "train_data": {"layer": layer},
            }
            cfgs.append(cfg)

    return cfgs
```

This creates 2 parallel groups:
- Job 1: layer=6, lr=[1e-4, 3e-4]
- Job 2: layer=12, lr=[1e-4, 3e-4]

!!! note "Implementation detail"
    See `CANNOT_PARALLELIZE` in `train.py` for the full list of parameters that split parallel groups. The `split_cfgs()` function handles grouping automatically.

## Module Loading

Your sweep file is executed as a Python module, so you can use imports and helper functions:

```python
def make_cfgs():
    cfgs = []

    # You can use helper functions
    base_layers = list(range(6, 24, 2))

    for layer in base_layers:
        for lr in [1e-4, 3e-4]:
            cfg = {
                "lr": lr,
                "train_data": {"layer": layer, "n_threads": 8},
                "val_data": {"layer": layer, "n_threads": 8},
                "sae": {"d_model": 1024, "d_sae": 16384},
            }
            cfgs.append(cfg)

    return cfgs
```

!!! note "Import mechanics"
    The sweep file is loaded with `importlib.import_module()`, so it must be importable as a Python module. Place sweep files in a location where Python can find them (typically the project root or a `sweeps/` subdirectory).

## Slurm Integration

When running with `--slurm-acct`, each parallel group becomes a separate Slurm job:

```bash
uv run train.py --sweep sweeps/large.py \
  --slurm-acct PAS2136 \
  --slurm-partition nextgen \
  --n-hours 24
```

The system automatically:
- Groups configs that can parallelize
- Submits one Slurm job per group
- Waits for all jobs to complete
- Reports results

## Seed Management

Seeds are automatically incremented for each config to ensure reproducibility:

```python
# Base config has seed=42
# Sweep generates 9 configs with seeds: 42, 43, 44, ..., 50
```

Override the base seed on the command line:

```bash
uv run train.py --sweep sweeps/my_sweep.py --seed 100
```

## Examples

**Simple grid:**

```python
# sweeps/simple.py
def make_cfgs():
    return [
        {"lr": lr, "objective": {"sparsity_coeff": sp}}
        for lr in [1e-4, 3e-4, 1e-3]
        for sp in [4e-4, 8e-4, 1.6e-3]
    ]
```

**Layer sweep with paired train/val:**

```python
# sweeps/layers.py
def make_cfgs():
    cfgs = []

    for layer in range(6, 24, 2):  # Layers 6, 8, 10, ..., 22
        for lr in [3e-4, 1e-3]:
            cfg = {
                "lr": lr,
                "train_data": {"layer": layer},
                "val_data": {"layer": layer},
            }
            cfgs.append(cfg)

    return cfgs
```

**Architecture sweep:**

```python
# sweeps/architecture.py
def make_cfgs():
    cfgs = []

    architectures = [
        ("small", 8192, 1e-3),
        ("medium", 16384, 5e-4),
        ("large", 32768, 3e-4),
    ]

    for name, d_sae, lr in architectures:
        cfg = {
            "lr": lr,
            "sae": {"d_sae": d_sae},
            "tag": name,
        }
        cfgs.append(cfg)

    return cfgs
```
