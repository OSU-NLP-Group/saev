# Guide

This guide explains how to transition from the ADE20K demo to using `saev` with your own custom datasets.

Here are the steps:

1. Record ViT activations and save them to disk.
2. Train SAEs on the activations.
3. Evaluate the SAE checkpoints.
4. Visualize the learned features from the trained SAEs.

!!! note
    `saev` assumes you are running on NVIDIA GPUs. On a multi-GPU system, prefix your commands with `CUDA_VISIBLE_DEVICES=X` to run on GPU X.

## Record ViT Activations to Disk

To save activations to disk, we need to specify:

1. Which model we would like to use
2. Which layers we would like to save.
3. Where on disk and how we would like to save activations.
4. Which images we want to save activations for.

The `scripts/shards.py` script does all of this for us.

Run `uv run scripts/launch.py shards --help` to see all the configuration.

In practice, you might run:

```sh
uv run scripts/launch.py shards \
  --family siglip \
  --ckpt hf-hub:timm/ViT-L-16-SigLIP2-256 \
  --d-model 1024 \
  --n-patches-per-img 256 \
  --no-cls-token \
  --layers 13 15 17 19 21 23 \
  --dump-to /fs/scratch/PAS2136/samuelstevens/cache/saev/ \
  --max-patches-per-shard 500_000 \
  --slurm-acct PAS2136 \
  --n-hours 48 \
  --slurm-partition nextgen \
  data:image-folder \
  --data.root /fs/ess/PAS2136/foundation_model/inat21/raw/train_mini/
```

Let's break down these arguments.



This will save activations for the CLIP-pretrained model ViT-B/32, which has a residual stream dimension of 768, and has 49 patches per image (224 / 32 = 7; 7 x 7 = 49).
It will save the second-to-last layer (`--layer -2`).
It will write 2.4M patches per shard, and save shards to a new directory `/local/scratch/$USER/cache/saev`.


.. note:: A note on storage space: A ViT-B/16 will save 1.2M images x 197 patches/layer/image x 1 layer = ~240M activations, each of which take up 768 floats x 4 bytes/float = 3072 bytes, for a **total of 723GB** for the entire dataset. As you scale to larger models (ViT-L has 1024 dimensions, 14x14 patches are 224 patches/layer/image), recorded activations will grow even larger.

This script will also save a `metadata.json` file that will record the relevant metadata for these activations, which will be read by future steps.
The activations will be in `.bin` files, numbered starting from 000000.

To add your own models, see the guide to extending in `saev.activations`.

## Train SAEs on Activations

To train an SAE, we need to specify:

1. Which activations to use as input.
2. SAE architectural stuff.
3. Optimization-related stuff.

The `saev.training` module handles this.

Run `uv run python -m saev train --help` to see all the configuration.

Continuing on from our example before, you might want to run something like:

```sh
uv run python -m saev train \
  --data.shard-root /local/scratch/$USER/cache/saev/ac89246f1934b45e2f0487298aebe36ad998b6bd252d880c0c9ec5de78d793c8 \
  --data.layer -2 \
  --data.patches patches \
  --data.no-scale-mean \
  --data.no-scale-norm \
  --sae.d-model 768 \
  --lr 5e-4
```

```sh
uv run train.py --sweep configs/preprint/baseline.toml --data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/f9deaa8a07786087e8071f39a695200ff6713ee02b25e7a7b4a6d5ac1ad968db --data.patches image --data.layer 23 --data.no-scale-mean --data.no-scale-norm sae:relu --sae.d-model 1024
```

`--data.*` flags describe which activations to use.

`--data.shard-root` should point to a directory with `*.bin` files and the `metadata.json` file.
`--data.layer` specifies the layer, and `--data.patches` says that want to train on individual patch activations, rather than the [CLS] token activation.
`--data.no-scale-mean` and `--data.no-scale-norm` mean not to scale the activation mean or L2 norm.
Anthropic's and OpenAI's papers suggest normalizing these factors, but `saev` still has a bug with this, so I suggest not scaling these factors.

`--sae.*` flags are about the SAE itself.

`--sae.d-model` is the only one you need to change; the dimension of our ViT was 768 for a ViT-B, rather than the default of 1024 for a ViT-L.

Finally, choose a slightly larger learning rate than the default with `--lr 5e-4`.

This will train one (1) sparse autoencoder on the data.
See the section on sweeps to learn how to train multiple SAEs in parallel using only a single GPU.

## Visualize the Learned Features

Now that you've trained an SAE, you probably want to look at its learned features.
One way to visualize an individual learned feature \(f\) is by picking out images that maximize the activation of feature \(f\).
Since we train SAEs on patch-level activations, we try to find the top *patches* for each feature \(f\).
Then, we pick out the images those patches correspond to and create a heatmap based on SAE activation values.

.. note:: More advanced forms of visualization are possible (and valuable!), but should not be included in `saev` unless they can be applied to every SAE/dataset combination. If you have specific visualizations, please add them to `contrib/` or another location.

`saev.visuals` records these maximally activating images for us.
You can see all the options with `uv run python -m saev visuals --help`.

The most important configuration options:

1. The SAE checkpoint that you want to use (`--ckpt`).
2. The ViT activations that you want to use (`--data.*` options, should be roughly the same as the options you used to train your SAE, like the same layer, same `--data.patches`).
3. The images that produced the ViT activations that you want to use (`images` and `--images.*` options, should be the same as what you used to generate your ViT activtions).
4. Some filtering options on which SAE latents to include (`--log-freq-range`, `--log-value-range`, `--include-latents`, `--n-latents`).

Then, the script runs SAE inference on all of the ViT activations, calculates the images with maximal activation for each SAE feature, then retrieves the images from the original image dataset and highlights them for browsing later on.

.. note:: Because of limitations in the SAE training process, not all SAE latents (dimensions of \(f\)) are equally interesting. Some latents are dead, some are *dense*, some only fire on two images, etc. Typically, you want neurons that fire very strongly (high value) and fairly infrequently (low frequency). You might be interested in particular, fixed latents (`--include-latents`). **I recommend using `saev.interactive.metrics` to figure out good thresholds.**

So you might run:

```sh
uv run python -m saev visuals \
  --ckpt checkpoints/abcdefg/sae.pt \
  --dump-to /nfs/$USER/saev/webapp/abcdefg \
  --data.shard-root /local/scratch/$USER/cache/saev/ac89246f1934b45e2f0487298aebe36ad998b6bd252d880c0c9ec5de78d793c8 \
  --data.layer -2 \
  --data.patches patches \
  images:imagenet-dataset
```

This will record the top 128 patches, and then save the unique images among those top 128 patches for each feature in the trained SAE.
It will cache these best activations to disk, then start saving images to visualize later on.

`saev.interactive.features` is a small web application based on [marimo](https://marimo.io/) to interactively look at these images.

You can run it with `uv run marimo edit saev/interactive/features.py`.


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
d_vit = 768
exp_factor = [8, 16]

[data]
scale_mean = true
```

This would train 6 models (3 sparsity coefficients × 2 expansion factors), each sharing the same data loading operation.

### Limitations

Not all parameters can be swept in parallel.
Parameters that affect data loading (like `batch_size` or dataset configuration) will cause the sweep to split into separate parallel groups.
The system automatically handles this division to maximize efficiency.

## Training Metrics and Visualizations

When you train a sweep of SAEs, you probably want to understand which checkpoint is best.
`saev` provides some tools to help with that.

First, we offer a tool to look at some basic summary statistics of all your trained checkpoints.

`saev.interactive.metrics` is a [marimo](https://marimo.io/) notebook (similar to Jupyter, but more interactive) for making L0 vs MSE plots by reading runs off of WandB.

However, there are some pieces of code that need to be changed for you to use it.

.. todo:: Explain how to use the `saev.interactive.metrics` notebook.

* Need to change your wandb username from samuelstevens to USERNAME from wandb
* Tag filter
* Need to run the notebook on the same machine as the original ViT shards and the shards need to be there.
* Think of better ways to do model and data keys
* Look at examples
* run visuals before features

How to run visuals faster?

explain how these features are visualized

