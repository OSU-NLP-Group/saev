# Reproduce

There are two main experiments to reproduce in our preprint.

First, our qualitative examples.
Second, our quantitative evaluation of pseudo-orthogonality.

## Qualitative

You can reproduce our qualititative examples from our preprint by following these instructions.

1. Train a linear probe on semantic segmentation task using ADE20K.
2. Measure linear probe baseline metrics.
3. Manipulate the activations using the proposed SAE features.
4. Be amazed. :)

Details can be found below.

### Train a Linear Probe on Semantic Segmentation

Train a linear probe on DINOv2 activations from ADE20K.
It's fixed with DINOv2 because of patch size, but the code could be extended to different ViTs.

```sh
uv run python -m contrib.semseg train \
  --sweep contrib/semseg/sweep.toml \
  --imgs.root /$NFS/$USER/datasets/ade20k
```

### Measure Linear Probe Baseline Metrics

Check which learning rate/weight decay combination is best for the linear probe.

```sh
uv run python -m contrib.semseg validate \
  --imgs.root /$NFS/$USER/datasets/ade20k
```

Then you can look in `./logs/contrib/semseg` for `hparam-sweeps.png` to see what learning rate/weight decay combination is best.

### Manipulate the Activations

You need an SAE that's been trained on DINOv2's activations on ImageNet.
Then you can run both the frontend server and the backend server:

**Frontend:**

```sh
uv run python -m http.server
```

Then navigate to [http://localhost:8000/web/apps/semseg/](http://localhost:8000/web/apps/semseg/).

**Backend:**

This is a little trickier because the backend server lives on Huggingface spaces and talks to a personal Cloudflare server.

[TODO]


## Quantitative

We aim to measure the specificity and psuedo-orthogonality of SAE-discovered features by evaluating the impact of feature manipulation on semantic segmentation.

We train an SAE on ImageNet-1K activations from DINOv2 ViT-B/14 ([hosted here on HuggingFace](https://huggingface.co/osunlp/SAE_DINOv2_24K_ViT-B-14_IN1K)).
Then, we train a linear probe on top of DINOv2 for ADE20K following the procedure above.
We define four ways to select a feature vector for a given ADE20K class:

1. Random unit vector in $d$-dimensional space
2. Random SAE feature vector.
3. Automatically selected SAE feature vector.
4. Manually chosen SAE feature vector.

All four are described in more detail below.

Given a feature $i$ and an ADE20K class $c$, for each image in the validation set, we perform semantic segmentation inference using DINOv2 and the trained linear probe.
However, we set feature $i$ to $-2$ its maximum observed value following the description of manipulation in Section 3.3 of our preprint.
We then maintain several counts:

1. Number of patches originally predicted as class $c$ and are now *not* $c$.
2. Number of patches originally predicted as class $c$ and are now *still* $c$.
3. Number of patches originally predicted as *not* class $c$ and are now $c$.
4. Number of patches originally predicted as *not* class $c$ and are now *still not* $c$.

With this, we calculate two percentages:

1. Target change rate: `(Number of original $c$ patches that changed class) / (Total number of original $c$ patches) * 100`
2. Other change rate: `(Number of original not-$c$ patches that changed class) / (Total number of original not-$c$ patches) * 100`

Ideally, we maximize target change rate and minimize other change rate.
We measure mean target change rate across all classes and mean other change rate across all classes.

```sh
uv run python -m contrib.semseg quantify \
  --sae-ckpt checkpoints/public/oebd6e6i/sae.pt \
  --seg-ckpt checkpoints/contrib/semseg/lr_0_001__wd_0_1/ \
  --imgs.root /$NFS/$USER/datasets/ade20k/
```
