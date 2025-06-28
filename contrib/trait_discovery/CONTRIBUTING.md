# CONTRIBUTING.md

# Research Goals

This project aims to use sparse autoencoders (SAEs) on vision transformers like BioCLIP and DINOv2 to identify interesting and scientifically meaningful visual morphological traits in living organisms.

**"Sparse autoencoders on vision transformers"**

Sparse autoencoders were recently applied to interpreting large language models by many groups.
[Anthropic's work](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) is probably the most well known, but [OpenAI has some work](https://cdn.openai.com/papers/sparse-autoencoders.pdf) and [Google does too](https://arxiv.org/abs/2408.05147).
I have some prior work ([website](https://osu-nlp-group.github.io/saev/), [arxiv](https://arxiv.org/abs/2502.06755))  that shows that sparse autoencoders can also be applied to vision transformer activations and nice-looking qualitative examples are discovered in ViT activations.

**"Interesting and scientifically meaningful"**

Many novel traits can be trivially described by both humans and multimodal large language models.
However, these traits are "uninteresting" because they are not connected to ecological context or evolutionary history.
We want to discover new traits that connect directly with something interesting to a biologist.

We are going to achieve this via two main groups:

1. Heliconious mimics. These butterflies evolve to look like each other. We want to find traits that are present in one species but not the other. The challenge is doing this with a limited number of images.
2. Equids (horses and zebras). What traits distinguish male and female equids? Are these traits consistent across species? Or unique to a species like "Grevy's zebra".

**"Visual morphological traits"**

We want visual morphological traits: spots, stripes, shapes, edges, colors, etc.

## Experiments

We have two baseline experiments and two novel trait discovery experiments.

1. CUB 200 2011
2. FishVista
3. Heliconious mimics
4. Equids (sexual dimorphism)

Here's how the baseline experiment for CUB 200 works:

We want to evaluate methods that score images for the presence of a particular trait, like "blue wing" or "round tail".
CUB 200 has 312 attributes labeled as present/absent for each image in both train and test splits.

Each image has ViT patch vector representations.
We want something like: $f : \mathbb{R}^{w \times h \times d} \rightarrow \mathbb{R}$.
We achieve this with a method that scores a single patch vector $g : \mathbb{R}^{d} \rightarrow \mathbb{R}$, and then we take the maximum score over all patches in an image.

So, for instance, we take a large set of ViT activations, and calculate the PCA of that entire dataset with 1024 components.
Now we have a matrix $\mathbb{R}^{1024 \times d}$.
For each image, we get 1024 scores by taking the dot product of each component with each patch vector, then taking the max over all patches in an image.

Finally, we pick the single "best" PCA component for each trait by finding the trait that maximizes average precision (AP) for that trait over the training set.
Then we evaluate each trait's AP over the test split and take the mean to get mAP.

We can vary:

* The number of PCA components
* The number of training images used to pick out each trait's best component.

In fact we do this.
See [contrib/trait_discovery/logbook.md](/contrib/trait_discovery/logbook.md) for a discussion of this with random vectors instead of PCA components.

## Repo layout

```
src/
  lib/
    baselines.py  <- Baseline methods
    metrics.py  <- batched AP
    cub200/  <- Helpers for cub200
tests/
  ...
dump_cub200_scores.py
train_baseline.py
```

## Environment

```sh
git clone https://github.com/OSU-NLP-Group/saev
git checkout ring-buffer

# Check that saev/ installed okay.
uv run train.py --help

# Check that trait_discovery installed okay.
cd contrib/trait_discovery
uv run train_baseline.py --help
uv run pytest tests/
```

## Coding Standards

* **Type‑annotations** with `jaxtyping` size‑suffixes (B, D, K, N, T).
* No hard‑wired paths. Use `Config` classes like in other files (`tyro` schema already stubbed).

See CONTRIBUTING.md and AGENTS.md in the saev root.

## Submitting PRs

1. Fork or feature branch.
2. Ensure `pytest` and `pre‑commit run --all-files` are green.
3. Push checkpoints to S3 and include link + config JSON (`wandb` or raw dict dump).
4. Open PR; CI must pass.
5. We’ll import the new checkpoints into `experiments/plot_baselines.ipynb`.

