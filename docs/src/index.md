# saev

![PyPI Downloads](https://static.pepy.tech/badge/saev)
![MIT License](https://img.shields.io/badge/License-MIT-efefef)
![GitHub Repo stars](https://img.shields.io/github/stars/OSU-NLP-group/saev?style=flat&label=GitHub%20%E2%AD%90)

saev is a framework for training and evaluating **S**parse **a**uto**e**ncoders (SAEs) for **v**ision transformers (ViTs), implemented in PyTorch.

## Installation

Installation is supported with [uv](https://docs.astral.sh/uv/).
saev will likely work with pure pip, conda, etc. but I will not formally support it.

Clone this repository, then from the root directory:

```bash
uv run scripts/launch.py --help
```

This will create a virtual environment and display the help for all the provided framework scripts.

## Quick Start

Save some activations to disk:

```bash
uv run scripts/launch.py shards \
  --shards-root /$SCRATCH/saev/shards \
  --family clip \
  --ckpt ViT-B-32/openai \
  --d-model 768 \
  --layers 11 \
  --patches-per-ex 49 \
  --batch-size 256 \
  data:cifar10
```

Read the [guide](users/guide.md) for details.

## Why saev?

There are plenty of alternative libraries for SAEs:

- [Overcomplete](https://github.com/KempnerInstitute/overcomplete), primarily developed by [Thomas Fel](https://thomasfel.me).

However, saev has some benefits:

1. saev is more of a framework, rather than a library. The reason for this is that SAEs require lots of activations to train a relatively small neural network; while you can implement it with a simple inference loop, efficient training requires some caching on disk. This means using saev is a little more like Keras or PyTorch Lightning than Huggingface's Transformers or Datasets libraries.
2. saev offers lots of tools for interacting with sparse autoencoders after training, including interactive notebooks and evaluations.
3. saev includes complete code from preprints in the `contrib/` directory, along with logbooks describing how the authors used and developed saev.
