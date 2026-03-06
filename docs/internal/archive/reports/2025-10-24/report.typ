#let title = [Sparse Autoencoders for Morphological Trait Discovery]

#set page(
  paper: "us-letter",
  numbering: "1",
  columns: 1,
)
#set par(justify: true)
#set text(size: 11pt)
#set math.equation(numbering: "(1)")
#set heading(numbering: "1.1.")

#show link: set text(blue)
#show figure.caption: set align(left)

#place(
  top + center,
  float: true,
  scope: "parent",
  clearance: 2em,
)[
  #align(center, text(17pt)[
    *#title*
  ])

  *Abstract*

  #pad(left: 0.5in, right: 0.5in, {
    set text(size: 10pt)
    set align(left)
    [
    Large self-supervised models learn remarkably accurate representations of the world.
    To predict masked patches, next frames, or contrastive similarities, models must discover and encode fundamental patterns governing natural phenomena.
    While purpose-built scientific models like AlphaFold demonstrate AI's potential for discovery, we propose that the representations learned by general self-supervised models constitute an untapped reservoir of scientific knowledge.
    *Can we extract novel scientific insights from what these models implicitly learn about the world?*

    We demonstrate that sparse autoencoders (SAEs) can serve as a bridge between the entangled representations of self-supervised models and human-interpretable scientific concepts.
    Using vision transformers trained solely for reconstruction and contrastive objectives on natural images, we show that SAEs discover morphological traits matching those painstakingly catalogued by biologists over centuries without any supervision for these concepts.
    More remarkably, we identify systematic morphological patterns in our extracted features that were previously unknown to domain experts but prove statistically valid upon investigation.

    This work establishes a new paradigm: using interpretability methods to mine scientific knowledge from the learned representations of large self-supervised models.
    Just as these models compress vast observational data into structured representations for prediction, we can decompress these representations to reveal the organizing principles they have discovered.
    We discuss implications for fields where large-scale observational data exists, from astronomy to materials science, and release our framework for extracting human-interpretable knowledge from pretrained models.
    ]
  })
]

= Story

+ Large models learn about the world through pre-training (see ideas and papers such as emergent abilities \& the platonic representation hypothesis).
+ The learned patterns are represented via vectors, distributed across both parameters _and_ activations.
+ Human scientists cannot readily interpret these patterns.
+ Interpretability methods are typically used to explain model predictions.
+ We leverage such methods to extract meaningful, human-interpretable learned world knowledge.

= Request for Comments

Here, I describe my main experiments, and the tables and figures used to communicate the results.
Notably, I am still missing the hook figure.
*I want feedback on both the abstract/story and the following questions:*
+ Do I have enough main experiments? Notably, I am cutting the discovery of Heliconius for the CVPR deadline.
+ Are all of my ablations necessary? Are there any that could be cut in the interest of time?
+ Is Probe $R$ a good metric? #cite(<gao2025scaling>, form: "prose") just report raw cross entropy numbers. I felt this is not as meaningful as explained variance.

= Experiment: Re-Discovering ADE20K Semantic Segmentations

#figure(
  caption: figure.caption(
    position: top,
    [SAEs for re-discovering ADE20K's semantic segmentation classes. All methods use DINOv3 ViT-L/16's patch-level activations from layer $20$ of $24$ with $16$K prototypes/latents. Probe $R$ is explained variance, as described in Section XX. Purity\@$k$ is precision over the top-$k$ highest-scoring patches for a feature, where $k=16$. Coverage\@$tau$ is the fraction of ADE20K classes with best AP $>= tau$, where $tau=0.3$. *Takeaway:* Matryoshka SAEs demonstrate meaningful improvement over baseline methods on all metrics, and outperform vanilla SAEs on all _downstream_ metrics despite worse _dictionary learning_ metrics (MSE and sparsity).]),
  placement: top,
  scope: "parent",
  table(
    columns: 7,
    column-gutter: 4pt,
    stroke: 0pt,
    align: (left, right, right, right, right, right, right),
    table.hline(stroke: 1pt),
    table.header(
      table.cell(rowspan: 2, align: horizon, [*Method*]), table.cell(colspan: 2, align: center, stroke: (bottom: 0.5pt), [Dictionary $arrow.b$]), table.cell(colspan: 4, align: center, stroke: (bottom: 0.5pt), [Downstream $arrow.t$]),
      [*Recon. MSE*], [*Sparsity*], [*Probe $R$*], [*mAP*], [*Precision\@$k$*], [*Coverage\@$tau$*]
    ),
    table.hline(stroke: 0.5pt),
    [$k$-Means], [], [], [], [], [], [],
    [PCA], [], [], [], [], [], [],
    [NMF], [], [], [], [], [], [],
    [Vanilla SAE], [], [], [], [], [], [],
    [Matryoshak SAE], [], [], [], [], [], [],
    table.hline(stroke: 1pt),
  )
) <tab:ade20k>

*Goal:* Use ADE20K (@zhou2016ade20k @zhou2017ade20k, segmentation task) as a quantitative measure of SAE quality to address past criticisms of no quantitative evaluation. Also compare SAEs to baseline dictionary learning methods like $k$-means, PCA and non-negative matrix factorization (NMF).

*Methodology:* Given a bunch of methods for scoring the presence of a particular "feature" in a specific patch, fit a 1D logistic regression classifier on each pair of ("feature", class) and pick the best latent for each class (same methodology as @gao2025scaling; very defensible in peer review).

Our primary metric of quality is "Probe $R$" (explained variance): $R = 1-cal(L)/cal(L)_pi$, where $cal(L)(c)$ is the minimum binary cross entropy loss achieved across all features for a given class $c$. and $cal(L)_pi (c)$ is the binary cross entropy loss achieved if we only use the class prevalence $pi$ as our bias term, and ignore the input entirely. We take the mean across all classes to measure $R$, which is a measure of how much better our classification improves when considering the "feature" for classification.
We also will report reconstruction MSE, L0 sparsity, mAP across all classes, Precision\@$k$ and Coverage\@$tau$, which are described in @tab:ade20k.

*Results:* We'll report results in a full-width table only, with the layout in @tab:ade20k. We can put qualitative examples in the appendix.

= Experiment: Discovering FishVista Body Parts

*Goal:* Use FishVista @fishvista as another segmentation dataset to demonstrate that domain-specific ViTs lead to qualitatively different traits. This motivates the (future) use of SAEs and interpretability methods on models like AlphaFold and GraphCast.

*Methodology:* Train Matryoshka SAEs on SigLIP 2, BioCLIP 2 and DINOv2 activations, comparing their quality on downstream metrics using the probe methodology from above.

#figure(
  caption: figure.caption(
    position: top,
    [SAEs for re-discovering FishVista's body-part segmentations. All methods use a Matryoshka SAE with $16$K prototypes/latents. Probe $R$ is explained variance, as described in Section XX. Purity\@$k$ is precision over the top-$k$ highest-scoring patches for a feature, where $k=16$. Coverage\@$tau$ is the fraction of ADE20K classes with best AP $>= tau$, where $tau=0.3$. *Takeaway:* Domain-specific ViTs demonstrate meaningful improvement over general-domain ViTs.]),
  placement: top,
  scope: "parent",
  table(
    columns: 7,
    column-gutter: 4pt,
    stroke: 0pt,
    align: (left, right, right, right, right, right, right),
    table.hline(stroke: 1pt),
    table.header(
      table.cell(rowspan: 2, align: horizon, [*ViT*]), table.cell(colspan: 2, align: center, stroke: (bottom: 0.5pt), [Dictionary $arrow.b$]), table.cell(colspan: 4, align: center, stroke: (bottom: 0.5pt), [Downstream $arrow.t$]),
      [*Recon. MSE*], [*Sparsity*], [*Probe $R$*], [*mAP*], [*Precision\@$k$*], [*Coverage\@$tau$*]
    ),
    table.hline(stroke: 0.5pt),
    [CLIP], [], [], [], [], [], [],
    [SigLIP 2], [], [], [], [], [], [],
    [DINOv2], [], [], [], [], [], [],
    [BioCLIP 2], [], [], [], [], [], [],
    table.hline(stroke: 1pt),
  )
) <tab:fishvista>

#figure(
  image("fish-example.png", width: 100%),
  caption: [Example Matyroshka SAE features for "caudal fins" (tail fins) from four different ViTs (CLIP, SigLIP 2, DINOv2 and BioCLIP 2). SAE features were picked by maximizing cross entropy on binary classification for caudal fins, as described in Section XX. *Takeaway:* Domain-specific pre-trained models such as BioCLIP 2 for morphological traits lead to more precise, more useful (re-)discoveries.],
  placement: top,
  scope: "parent",
) <fig:fishvista>

*Results:* We'll report results in a full-width table, with the layout in @tab:fishvista. We will also report qualitative comparisons with the layout in @fig:fishvista.

*Extensions:* Fish also have rich metadata. Does BioCLIP 2 find traits that align with particular across-species patterns, like finding features for "tail fins of bottom feeders" or "mouths of carnivorous river fish", etc.
This is an extension because this data is not natively linked with FishVista, and aligning it might be more challenging than we expect.

// = Experiment: Discovering New Morphological Traits in _Heliconius_ Butterflies

// *Goal:* Discover a new morphological trait that distinguishes mimic pairs in _Heliconius_ butterflies.

= Experiment: Ablations

*Goals:* ablations demonstrate our rigor and explain tradeoffs.
- Vision transformer size: DINOv3 has ViT-S/16, ViT-B/16 and ViT-L/16.
- Image resolution and aspect ratios: DINOv3 with 256x256 (256 total patches), 512x512 (1024 total patches), 640 patches at native aspect ratio, 1280 at native aspect ratio.
- Different ViT layers: Try layers 14, 16, 18, 20, 22 and 24 out of 24-layer ViTs (DINOv3 again).
- Different sparsity tradeoffs: Show a typical MSE vs sparsity tradeoff.
Each ablation will contain either a single-column table or a single-column figure/chart.

#bibliography("../refs.yml")
