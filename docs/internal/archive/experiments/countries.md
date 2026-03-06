# SAE Country Precision

**Objective**: measure the concept precision of CLIP- and DINOv2-trained SAEs for cultural features. We hypothesize that a CLIP SAE will have higher precision than DINOv2 because it was trained with language, which semantically links visually dissimilar images (like the German flag and the Brandenburg Gate).

## Variables

Independent Variables: The model that produced the SAE's activations.

Controlled Variables: The dataset (ImageNet), the ViT size (ViT-B), the SAE size (32x expansion = 24K).

Dependent Variables: Cultural precision of SAEs.

## Procedure 

Dataset: We will manually gather this dataset. There are 249 ISO-3166 countries. We'll just pick the top 30 countries by population. For each country, we can pick out five different images of the flag, two of the coat of arms, five relevant landmarks, and three images of the national football/soccer team for a total of 15 images per country, or 450 total images.

Models: ImageNet-1K trained SAEs for both DINOv2 and CLIP, both ViT-B.

Baselines (comparators and rationale):

* Random SAE unit
* Random unit vector
* CAV trained using flags only (TODO: explain why)

Metrics (primary, secondary, error bars, test plan):

* TODO

Steps:

* TODO

Hardware Budget (GPUs, expected hours)

When to Stop (wall-clock or cost cap)

Risks (and mitigations)

Success Definition

Priority (impact, effort, 1–5 each)

Owner

Reviewer

ETA

Release Artifacts (code, logs, weights, notebooks)


---

### Reviewer-requested “country-feature” benchmark — detailed spec

| Step                         | Choice & rationale                                                                                                                                                                                                                                                                                                                                                                 | Practical notes                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Source images**         | **(A) World flags.**  One high-res PNG per ISO-3166 country from Flagpedia or the GitHub “country-flags” repo; guarantees uniform coverage and unambiguous label. <br>**(B) Landmarks.**  Use Google Landmarks v2 meta-data: each landmark has a *country\_code* field.  Sample ≤50 images per country to give variety (architecture, landscapes, people in national dress, etc.). | *A* is 256 PNGs, 50 MB total.  *B* is a 5 M-image dataset; you only need \~12 k images (50×246) so you can stream-filter by country code. |
| **2. Activation pass**       | Run the frozen backbone and your trained SAE on every image (flag or landmark).  Store: country label, SAE index, max activation value in the image.                                                                                                                                                                                                                               | One GPU ≈30 min for flags, ≈6 h for 12 k landmarks.                                                                                       |
| **3. Unit–country matching** | For each SAE unit *u* compute average activation over all images of country *c*.  Assign unit → country = arg max<sub>c</sub> activation if that average ≥1 σ above global mean; else “no country”.                                                                                                                                                                                | Gives at most one “primary country” per unit, mirroring reviewer request.                                                                 |
| **4. Precision test**        | For each matched pair (*u*, *c*): rank all images (across *all* countries) by *u*’s activation and inspect top-100.  **Unit precision** = fraction whose ground-truth country = *c*.                                                                                                                                                                                               | A script suffices, no human labelling because test images are already labelled.                                                           |
| **5. Recall test**           | **Country recall** = fraction of countries that have ≥1 unit with precision ≥0.8 (tunable).                                                                                                                                                                                                                                                                                        | Shows whether the SAE vocabulary covers most nations.                                                                                     |
| **6. Baselines**             | *Random unit* (same norm), *single neuron* in ViT layer 11, *TCAV* vector trained on the flag images for that country.                                                                                                                                                                                                                                                             | Implementable with your existing pipeline + a small TCAV helper.                                                                          |
| **7. Report**                | Table: mean precision\@100, std-dev; country recall.  Add qualitative grids for hits & misses (e.g.\ Argentina vs Uruguay).                                                                                                                                                                                                                                                        | Addresses reviewers’ “run a systematic count” demand.                                                                                     |

#### Assumptions and caveats to note in the paper

* Flags are pristine symbols; a unit that loves “Brazil flag green+yellow” may not fire on Rio street scenes — we treat flags as a *proxy* for national visual concepts.
* Google Landmarks is biased toward tourist hotspots in wealthy countries; recall is therefore an *upper* bound.
* Because we assign each unit to at most one country, polysemantic units will be under-counted.

#### Why this satisfies the critique

* **Quantitative, reproducible, low-cost.**  No MTurk; everything labelled.
* **Directly mirrors reviewers’ “Brazil feature” example** with precision & recall numbers.
* **Provides baseline comparison** to TCAV and random directions, showing the added value of SAEs.

---

### Two more quick quantitative add-ons (optional)

1. **Bird attribute causal score**
   *CUB attribute CSV → suppress top unit → logit drop on species possessing that attribute vs others; report AUC.*

2. **Patch specificity metric**
   *ADE20K “sand” example automated over 30 classes: target-pixel flip % vs other-pixel flip %.*

Both reuse your existing models and strengthen causal-edit claims without new data collection.

