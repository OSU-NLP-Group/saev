# Reproduce

This document explains the experiment and the steps to reproduce.

## Experimental Design

A high-level description of the experimental setup is below.

1. Pre-experiment setup:
  * Choose K semantic features to search for (e.g., "wheel", "fur", "window")
  * Create pilot dataset to calibrate methodology and verify processing

2. Positive sample collection:
  * Use web search to find 20 clear examples.
  * Record metadata on why each image was selected.

3. Negative sample collection:
  * Randomly sample 20 ImageNet images
  * Manually verify that they definitely lack the feature; if not, remove and re-sample.
  * Document verification criteria used

4. Data processing:
  * Run both image sets through ViT
  * Extract patch activations
  * Run through SAE
  * For each image, aggregate patch scores (sum across patches)
  * Threshold at 0 for binary predictions

5. Feature scoring:
  * Calculate F1 for each SAE feature
  * Calculate F1 for randomly selected SAE features as baseline
  * Rank by F1 score

6. Validation checks:
  * Statistical significance test comparing top feature F1s vs random baseline

7. Human verification:
  * Look at top 5 features by F1
  * For each feature, examine what patches maximally activate it
  * Document both confirming and disconfirming evidence
  * Record failure modes and edge cases

8. Documentation: for each validated feature, record:
  * F1 score and comparison to random baseline
  * Examples of true positives/negatives, false positives/negatives
  * Hypotheses about what the feature is actually detecting
  * Key limitations or caveats

## Reproduce

Here are the specific scripts you need to run to get these results.

First, record the activations for both CLIP and DINOv2 pre-trained ViTs.

```sh
uv run python -m saev activations \
  --vit-family clip \
  --vit-ckpt ViT-B-16/openai \
  --d-vit 768 \
  --n-patches-per-img 196 \
  --dump-to /local/scratch/$USER/cache/saev \
  data:image-folder-dataset \
  --data.root /$NFS/$USER/datasets/semprobe/test
```

```sh
uv run python -m saev activations \
  --vit-family dinov2 \
  --vit-ckpt dinov2_vitb14_reg \
  --d-vit 768 \
  --n-patches-per-img 256 \
  --dump-to /local/scratch/$USER/cache/saev \
  data:image-folder-dataset \
  --data.root /$NFS/$USER/datasets/semprobe/test
```

To get SAE features for each, you need a pre-trained checkpoint.
I used oebd6e6i for DINOv2 and usvhngx4 for CLIP, which are both trained on ImageNet activations from those ViTs.

```sh
uv run python -m contrib.semprobe score \
  --sae-ckpt checkpoints/usvhngx4/sae.pt \
  --imgs.root /$NFS/$USER/datasets/semprobe/test/ \
  --acts.shard-root /local/scratch/$USER/cache/saev/$SHARDS/
```
