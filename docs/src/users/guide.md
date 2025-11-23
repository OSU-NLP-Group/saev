# Guide

This guide explains how to transition from the ADE20K demo to using `saev` with your own custom datasets.

Here are the steps:

1. [Save ViT activations to disk](#save-vit-activations-to-disk)
2. [Train SAEs on activations](#train-saes-on-activations)
3. [Evaluate the SAE checkpoints](#evaluation)
3. [Visualize Learned Features](#visualize-learned-features)

!!! note
    `saev` assumes you are running on NVIDIA GPUs. On a multi-GPU system, prefix your commands with `CUDA_VISIBLE_DEVICES=X` to run on GPU X.

## Save ViT Activations to Disk

To save activations to disk, we need to specify:

1. Which model we would like to use
2. Which layers we would like to save.
3. Where on disk and how we would like to save activations.
4. Which images we want to save activations for.

The `saev/framework/shards.py` script does all of this for us.

Run `uv run scripts/launch.py shards --help` to see all the configuration.

In practice, you might run:

```sh
uv run scripts/launch.py shards \
  --shards-root /fs/scratch/PAS2136/samuelstevens/saev/shards \
  --family clip \
  --ckpt ViT-B-16/openai \
  --d-model 768 \
  --layers 6 7 8 9 10 11 \
  --content-tokens-per-example 196 \
  --batch-size 512 \
  --slurm-acct PAS2136 \
  --slurm-partition nextgen \
  data:img-seg-folder \
  --data.root /fs/scratch/PAS2136/samuelstevens/datasets/ADEChallengeData2016/ \
  --data.split training
```

This will save activations for the CLIP-pretrained model ViT-B/16, which has a residual stream dimension of 768, and has 196 patches per image (224 / 16 = 14; 14 x 14 = 196).
It will save the last 6 layers.
It will write 2.4M patches per shard, and save shards to a new directory `/fs/scratch/PAS2136/samuelstevens/saev/shards`.


!!! note
    A note on storage space: A ViT-B/16 on ImageNet-1K will save 1.2M images x 197 patches/layer/image x 1 layer = ~240M activations, each of which take up 768 floats x 4 bytes/float = 3072 bytes, for a **total of 723GB** for the entire dataset. As you scale to larger models (ViT-L has 1024 dimensions, 14x14 patches are 224 patches/layer/image), recorded activations will grow even larger.

This script will also save a `metadata.json` file that will record the relevant metadata for these activations, which will be read by future steps.
The activations will be in `.bin` files, numbered starting from 000000.

To add your own models, see the guide to extending in `saev.activations`.

## Train SAEs on Activations

To train an SAE, we need to specify:

1. Which activations to use as input.
2. SAE architectural stuff.
3. Optimization-related stuff.

The `train.py` script handles this.

Run `uv run train.py --help` to see all the configuration.

The most important options are:

- `--runs-root`: where to store runs.
- `--train-data` and `--val-data`: How to load the training and validation data. You probably want to specify both `--{train,val}-data.shards` (the shard directory) and `--{train,val}-data.layer` (which layer to use).
- `sae.activation`: `sae.activation:relu` to use the ReLU activation.

This is a full example:

```sh
uv run train.py \
  --runs-root /fs/ess/PAS2136/samuelstevens/saev/runs \
  --lr 4e-3 \
  --sae.exp-factor 16 \
  --sae.d-model 1024 \
  --tag ade20k-v0.1 \
  --n-train 100_000_000 \
  --slurm-acct PAS2136 \
  --slurm-partition nextgen \
  --train-data.shards /fs/scratch/PAS2136/samuelstevens/saev/shards/51567c6c \
  --train-data.layer 11 \
  --val-data.shards /fs/scratch/PAS2136/samuelstevens/saev/shards/3e27794f \
  --val-data.layer 11 \
  sae.activation:relu \
  objective:matryoshka \
  --objective.sparsity-coeff 1e-3 \
```

This will train one (1) sparse autoencoder on the data.
See the section on sweeps to learn how to train multiple SAEs in parallel using one or more GPUs.

### Loader Entropy Metrics

The training loop logs additional loader diagnostics derived from `calc_batch_entropy` in `train.py`. Every batch contributes two entropy measurements in natural log units:

- `loader/example_entropy` and `loader/example_entropy_normalized` summarize how evenly the shuffled loader samples example indices. Normalization divides the raw entropy by `ln(metadata.n_examples)` so perfectly uniform sampling is 1.0.
- `loader/token_entropy` and `loader/token_entropy_normalized` do the same for patch indices using `ln(metadata.content_tokens_per_example)` as the normalizer.
- `loader/example_coverage` and `loader/token_coverage` report the fraction of distinct example or patch indices seen in the current batch relative to their theoretical support.

All eight metrics appear alongside the existing `loader/read_mb` counters, helping spot skewed sampling or under-covered patches mid-run.

## Evaluation

After training an SAE, you probably want to *use* the SAE.
While you can use the SAE as a regular PyTorch `torch.nn.Module` in combination with a `saev.data.OrderedDataLoader` or `saev.data.IndexedDataset`.

However, most SAEs are evaluated with a similar set of metrics (normalized MSE, L0, etc).
The `saev/framework/inference.py` script calculates these metrics.
You can run `uv run scripts/launch.py inference --help` to see all the options.

The most important options are:

- `--run`: The path to the SAE run directory.
- `--data`: The options for the OrderedDataLoader. Specifically, you need to set `--data.shards` and `--data.layer`, just like for training.

```sh
uv run scripts/launch.py inference \
  --run /fs/ess/PAS2136/samuelstevens/saev/runs/z55bntm1/ \
  --data.shards /fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0 \
  --data.layer 11
```

## Visualize Learned Features

Now that you've trained an SAE, you probably want to look at its learned features.
One way to visualize an individual learned feature is by picking out images that maximize the activation of feature.
We use the saved sparse `token_acts.npz` file from the previous inference step.

!!! warning

    Because there are so many different ways to visualize SAE features, I moved it to `contrib/trait_discovery` (used for our preprint ["Towards Open-Ended Visual Scientific Discovery with Sparse Autoencoders"]()).


The most important options:

- `--run`: The path to the SAE run directory.
- `--shards`: The shards directory.
- `--latents`: The 0-indexed latents to save images for.
- `--n-latents`: The number of randomly selected latents to save images for.

So first, move into the `contrib/trait_discovery`:

```sh
cd contrib/trait_discovery
```

Then run the script that generates highlighted images:

```sh
uv run scripts/launch.py visuals \
  --run /fs/ess/PAS2136/samuelstevens/saev/runs/unu6dbfb \
  --shards /fs/scratch/PAS2136/samuelstevens/saev/shards/3802cb66 \
  --latents 0 1 2 3 4 5 6 7 8 9 49 56 57 125 202 \
  --n-latents 20 \
```

!!! note

    Because of limitations in the SAE training process, not all SAE latents are equally interesting. Some latents are dead, some are *dense*, some only fire on two images, etc. Typically, you want neurons that fire very strongly (high value) and fairly infrequently (low frequency). You might be interested in particular, fixed latents (`--include-latents`). **I recommend using `saev/interactive/metrics.py` with marimo to figure out good thresholds.**


## Sweeps

> tl;dr: basically the slow part of training SAEs is loading vit activations from disk, and since SAEs are pretty small compared to other models, you can train a bunch of different SAEs in parallel on the same data using a big GPU. That way you can sweep learning rate, lambda, etc. all on one GPU.

### Why Parallel Sweeps

SAE training optimizes for a unique bottleneck compared to typical ML workflows: disk I/O rather than GPU computation.
When training on vision transformer activations, loading the pre-computed activation data from disk is often the slowest part of the process, not the SAE training itself.

A single set of ImageNet activations for a vision transformer can require terabytes of storage.
Reading this data repeatedly for each hyperparameter configuration would be extremely inefficient.

### Parallelized Training Architecture

To address this bottleneck, we implement parallel training that allows multiple SAE configurations to train simultaneously on the same data batch:

<pre class="mermaid">
flowchart TD
    A[Pre-computed ViT Activations] -->|Slow I/O| B[Memory Buffer]
    B -->|Shared Batch| C[SAE Model 1]
    B -->|Shared Batch| D[SAE Model 2]
    B -->|Shared Batch| E[SAE Model 3]
    B -->|Shared Batch| F[...]
</pre>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
</script>

This approach:

- Loads each batch of activations **once** from disk
- Uses that same batch for multiple SAE models with different hyperparameters
- Amortizes the slow I/O cost across all models in the sweep

### Running a Sweep

The `train` command accepts a `--sweep` parameter that points to a TOML file defining the hyperparameter grid:

```bash
uv run python -m saev train --sweep configs/my_sweep.toml
```

Here's an example sweep configuration file:

```toml
[sae]
sparsity_coeff = [1e-4, 2e-4, 3e-4]
d_model = 768
d_sae = [6144, 12288]

[data]
scale_mean = true
```

This would train 6 models (3 sparsity coefficients × 2 SAE widths), each sharing the same data loading operation.

### Limitations

Not all parameters can be swept in parallel.
Parameters that affect data loading (like `batch_size` or dataset configuration) will cause the sweep to split into separate parallel groups.
The system automatically handles this division to maximize efficiency.
