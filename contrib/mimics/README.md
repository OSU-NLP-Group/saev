# Cambridge Mimicry: SAE Feature Validation for Heliconius Butterflies

Tools for validating and exploring SAE features that discriminate Heliconius mimic pairs (erato subspecies vs melpomene subspecies), using the Cambridge Butterfly dataset.

## Problem

Current workflow: train sparse linear classifiers on SAE activations, extract selected feature indices, run `visuals.py` to render top-activating images, then manually interpret.

The interpretation step is where things break down. `visuals.py` shows a feature's top-activating images globally with no connection to the classification task. There are no class labels on images, no per-class breakdown, no discrimination stats. You end up switching between the feature visuals, notebook tables, and raw data, trying to hold context across windows.

The unit of analysis is wrong. It's currently "feature N activates on these images." It should be "feature N captures this morphological difference between these two taxa."

## Ideas

### Tier 1: Close the validation loop

#### 1. Per-class activation grids

For a (feature, task) pair, show images split by class:

```
FEATURE 2391 (weight +0.034, favors melpomene)
single-feature AUC: 0.91 | fires on 89% melpomene, 12% erato

        erato (lativitta dorsal)          melpomene (malleti dorsal)
Top     [img+overlay] [img] [img]         [img+overlay] [img] [img]
Bottom  [img+overlay] [img] [img]         [img+overlay] [img] [img]
```

Each image gets the SAE activation heatmap overlay and the subspecies/view label. "Top" = highest activation within that class, "Bottom" = lowest. Answers:

- Does the feature fire consistently on one class?
- Does it fire on the wrong class?
- When it doesn't fire, what does the specimen look like?
- Is it always the same wing region?

Highest priority item.

#### 2. Feature discrimination stats

For each (feature, task) pair, compute from existing SAE activations (`token_acts.npz` + class labels, no new inference):

- Activation rate per class: % of images where max patch activation > 0.
- Mean activation per class: the gap between classes measures effect size.
- Single-feature AUC: ROC AUC using just this feature. Individual discriminative power.
- Activation distribution: per-class histograms.

Computable for all features at once. Useful as a triage filter before rendering images.

#### 3. Labels on every image

Stamp `lativitta dorsal` or `malleti ventral` on every rendered image. PIL text overlay.

### Tier 2: Deeper understanding

#### 4. Spatial activation heatmaps

Average SAE activation maps across all specimens of a class for a given feature. Averages out per-specimen noise, reveals consistent spatial signal. Butterflies in the Cambridge dataset are pose-normalized (pinned specimens), so spatial averaging should work. The difference map (melpomene avg minus erato avg) shows where on the wing the feature discriminates.

#### 5. Counter-example explorer

For a feature that should distinguish erato from melpomene, find the hard cases:

- Erato specimens where the feature fires strongly (false positives)
- Melpomene specimens where the feature is silent (false negatives)

These reveal whether the feature is picking up on something subtler than expected, whether hybrids are confusing it, or whether a subspecies population is unusual.

#### 6. Feature interaction plots

For a classifier with N features, make 2D scatter plots of pairwise feature activations colored by class. Shows whether features contribute independent signal or are redundant.

#### 7. Morphological series

Sort all specimens by a feature's activation value and display as a filmstrip. The morphological trait should vary smoothly. Computational equivalent of a morphological series in taxonomy. Could be a GIF, a marimo slider, or a static strip of 20 images.

### Tier 3: Cross-run and meta-analysis

#### 8. Cross-run feature alignment

Different SAE runs have different latent indices. Feature 2391 in run A and feature 7739 in run B are unrelated by index, but if they activate on the same images, they capture the same trait.

Compute Jaccard similarity of top-K activated image sets across runs. Build a bipartite matching. If a trait appears in 8/10 runs, it's robust. If only 1 run, likely noise. Addresses the question of biological signal vs. SAE training artifacts.

#### 9. Mimicry precision quantification

For each feature, measure how closely the mimic matches the model:

```
Feature 2391:
  erato (lativitta): mean activation 0.82 +/- 0.15
  melpomene (malleti): mean activation 0.31 +/- 0.12
  mimicry gap: 0.51

Feature 48:
  erato: mean 0.44 +/- 0.08
  melpomene: mean 0.41 +/- 0.09
  mimicry gap: 0.03
```

Large gaps = mimicry breaks down (what the classifier picks up on). Small gaps = precise mimicry (strong selection pressure). Reframes classifier feature weights as a quantitative map of mimicry imprecision.

#### 10. Feature annotation dictionary

Record annotations as you validate: `{run: "r27w7pmf", feature: 2391, trait: "red forewing band extent", confidence: "high", notes: "consistent across dorsal views"}`. Simple JSON/CSV. Builds a reusable lookup from SAE latents to morphological traits.

#### 11. Batch HTML report

Standalone HTML per checkpoint. Each feature gets a card with the per-class grid (1), stats (2), and annotations (10). Inline images as base64, no server needed. For sharing with collaborators.

### Wild cards

#### 12. Hybrid specimen analysis

Run the classifier's selected features on hybrid specimens. See which parent species they resemble per trait. A hybrid might have erato-like forewing band (feature 2391 low) but melpomene-like hindwing spots (feature 48 high). Per-trait decomposition of hybrid phenotypes.

#### 13. Wing region vocabulary

Map features to named wing regions (forewing costa, hindwing discal cell, etc., standard lepidopteran venation nomenclature). Even a rough partition into 4-6 regions anchors ML features in morphological language.

## Priority

Build idea 1 first (per-class activation grids with labels). Compute idea 2 (discrimination stats) alongside it. Idea 3 (labels) is trivial.

Then idea 4 (spatial heatmaps) and idea 5 (counter-examples).

Ideas 8-9 (cross-run alignment, mimicry precision) are what turn this into a quantitative biology contribution rather than an interpretability exercise.
