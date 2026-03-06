#let title = [SAEs for Scientific Discovery: Status Report September 2025]
// ADVISOR COMMENT: Good framing - emphasizing "scientific discovery" positions
// this as more than a technical contribution. Consider adding subtitle like
// "Automated Morphological Trait Discovery in Biological Vision Models"

#set page(
  paper: "us-letter",
  numbering: "1",
  columns: 2,
)
#set par(justify: true)
#set text(
  size: 11pt,
)

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

  #align(center)[
    Samuel Stevens \
    The Ohio State University \
    #link("mailto:stevens.994@osu.edu")
  ]

  *Abstract*

  #pad(left: 0.5in, right: 0.5in, {
    set text(size: 10pt)
    set align(left)
    [
      // ADVISOR COMMENT: Abstract is concise but could be stronger. Consider:
      // 1) Start with the biological motivation/impact (why this matters for biology)
      // 2) Explicitly state the key technical innovation (Matryoshka SAEs for interpretability)
      // 3) Quantify results where possible (e.g., "discovered X diagnostic traits" or "Y% improvement over baselines")
      // 4) End with broader implications for automated phenotyping/trait discovery
      This report summarizes progress on using sparse autoencoders (SAEs) to discover morphological traits in biological images.
      We implemented Matryoshka SAEs with ReLU activation and trained SAEs on DINOv3 activations for two datasets (FishVista and the Heliconius butterflies).
      We developed aspect-aware resizing for DINOv3 and comprehensive metrics logging for hyperparameter optimization.
      Next steps focus on Heliconius butterfly analysis for novel trait discovery, and systematic feature evaluations using FishVista and ADE20K's pixel-level semantic labels.
    ]
  })
]

= Introduction

// ADVISOR COMMENT: Strong opening but consider restructuring for maximum impact:
// 1) Start with concrete biological problem (trait discovery is manual/subjective)
// 2) Then introduce vision transformers as having learned these traits implicitly
// 3) Finally present SAEs as the bridge to extract this knowledge
// Also: cite key papers (DINOv3, original SAE work, Matryoshka paper)
Large vision transformers learn rich, structured representations of the visual world through pre-training on massive datasets.
These models encode complex patterns---textures, shapes, object parts, and semantic relationships--within their high-dimensional activation spaces.
However, this learned knowledge remains locked in dense, entangled representations that resist human interpretation.
// ADVISOR COMMENT: Define "interpretable" more precisely - what makes a feature interpretable?
// Consider mentioning monosemanticity as the gold standard
Sparse autoencoders (SAEs) offer a promising approach to extract these learned structures into discrete, interpretable features that we can examine through collections of maximally-activating images.

// ADVISOR COMMENT: Good paragraph but needs more technical precision:
// 1) Define "meaningful" - meaningful to whom? Biologists? The model? Evolution?
// 2) Explain WHY sparsity leads to interpretability (cite literature)
// 3) Be specific about what "biological vision models" means (BioCLIP? Fine-tuned?)
We aim to use SAEs to automatically discover and isolate meaningful morphological traits from vision transformer representations.
SAEs decompose dense neural activations into sparse combinations of learned features, where each feature can be understood through the example images (anywhere from 5 to 20) that most strongly activate it.
// ADVISOR COMMENT: Why 5-20 images? Is this empirical or theoretical? Cite or explain
By training SAEs on biological vision models, we aim to create a computational lens for examining what these models have learned about organismal morphology, from wing patterns to body shapes to fine-grained anatomical structures.

// ADVISOR COMMENT: EXCELLENT figure! This is your strongest result. However:
// 1) Add quantitative validation - does this feature actually discriminate between species?
// 2) Show activation heatmaps overlaid on the images to make the correspondence clearer
// 3) Include statistics: how often does this feature fire for notabilis vs plesseni?
// 4) This should be Figure 1 - it's your hook!
#figure(
  [
    #image("images/notabilis-vs-plesseni.jpg")
    #grid(
      columns: (1fr, 1fr),
      image("images/notabilis-1.png"),
      image("images/notabilis-2.png"),
    )
  ],
  caption: [
    *Top:* A visual explanation of a known diagnostic trait (#link("https://cliniquevetodax.com/Heliconius/pages/erato%20notabilis.html")[source]) for distinguishing _H. erato notabilis_ (left) and _H. melpomene plesseni_ (right).
    *Bottom*: Top two images for feature DINOv3-16K/2735, which highlight the same parts of the wing that identifies _notabilis_.
    // ADVISOR COMMENT: Add: "Feature activates in X% of notabilis images vs Y% of plesseni"
  ],
  placement: top,
) <fig:heliconius-example>


// ADVISOR COMMENT: This paragraph identifies THE key challenge - good! But:
// 1) Be more specific about validation strategies you'll use
// 2) Explain why mimicry systems are ideal (convergent evolution = independent validation)
// 3) Consider discussing the "ground truth problem" more deeply
The core challenge is demonstrating that SAE features correspond to biologically meaningful traits rather than arbitrary visual patterns.
// ADVISOR COMMENT: Define "biologically meaningful" - phylogenetically informative? Functionally relevant? Under selection?
Without ground truth labels for most discoverable traits, evaluating features requires creative validation strategies that combine quantitative metrics with expert biological assessment.
This challenge becomes an opportunity when studying mimicry systems, where novel trait discoveries could reveal previously unrecognized convergent features.
// ADVISOR COMMENT: Expand on this - mimicry provides natural experiments where unrelated species
// independently evolve similar traits. Finding these in both lineages validates biological relevance

// ADVISOR COMMENT: Methods preview is good but needs more detail:
// 1) Why these specific vision models? What are their strengths?
// 2) Why these datasets? What makes them complementary?
// 3) What's your hypothesis about what SAEs will find that other methods won't?
We train SAEs on BioCLIP 2 and DINOv3 across three datasets: ADE20K for general visual understanding, FishVista for aquatic morphology, and Heliconius butterflies for studying mimicry.
// ADVISOR COMMENT: State the scale - how many images? How many species? This matters for generalization
We automatically evaluate feature quality using segmentation labels from ADE20K and FishVista, establishing baseline performance against PCA, k-means, and NMF decomposition methods.
// ADVISOR COMMENT: Why these baselines? Consider adding ICA as it also targets interpretability
For Heliconius, we aim to discover novel diagnostic traits between mimetic pairs (_H. melpomene_ and _H. erato_ subspecies), potentially revealing features that have escaped human observation but are captured in the models' learned representations.

= Progress

// ADVISOR COMMENT: Bullet points are fine for a status report, but for a paper you'll need:
// 1) Quantitative comparisons (tables/plots showing Matryoshka vs vanilla SAE)
// 2) Ablation studies (which components matter most?)
// 3) Computational costs (training time, memory, inference speed)

- Jake implemented Matryoshka SAE#footnote[#link("https://arxiv.org/abs/2503.17547")[Learning Multi-Level Features with Matryoshka Sparse Autoencoders]] objective and functions. This leads to qualitatively better features on FishVista.
  // ADVISOR COMMENT: "Qualitatively better" is vague - be specific. Better coverage? More interpretable? Higher activation values?
- We trained and visualized Matryoshka SAEs on DINOv3 activations on FishVista and butterflies.
  // ADVISOR COMMENT: Training details needed - which layers? How many latents? Training time? Hardware used?
- Found one qualitative example of a diagnostic trait in Heliconius mimics between _H. erato notabilis_ and _H. melpomene plesseni_. See @fig:heliconius-example for the diagnostic feature.
  // ADVISOR COMMENT: This is great but needs systematic evaluation - how many features did you examine? What's the false positive rate?
- Developed a supervised linear probe for FishVista semantic segmentation. This should be an upper bound on the mean quality of an SAE feature.
  // ADVISOR COMMENT: Smart baseline! What's the actual performance? How close do SAE features get?
- Added comprehensive metrics logging (MSE, explained variance, L0/L1 sparsity, dead unit percentage) to SAE training.
  // ADVISOR COMMENT: Good engineering. Consider adding: feature activation frequency, feature co-occurrence stats
- Developed aspect-aware image resizing for DINOv3.
  // ADVISOR COMMENT: Why was this needed? What problem does it solve? Show before/after comparisons

#figure(
  table(
    columns: 2,
    stroke: ((_, y) => (
      top: if y == 0 { 1pt } else { 0.5pt },
      bottom: 1pt,
    )),
    align: left + horizon,
    table.header(
      [*Date*], [*Goal*]
    ),
    [September 22, 2025], [Background-filtered visual examples of high-resolution (1920 patches/image) SAE features on DINOv3 with Heliconius butterflies.],
    [September 29, 2025], [Automatic _quantitative_ evaluation of FishVista feature quality for SAEs from different layers of DINOv3.],
    [October 6, 2025], [],
    [November 10, 2025], [_(Probable CVPR 2026 submission deadline)_],
  ),
  caption: [Approximate timeline for every Monday for the rest of 2025.
    // ADVISOR COMMENT: Timeline is too sparse. Add milestones for:
    // - Expert validation sessions
    // - Baseline comparisons complete
    // - Draft circulated to collaborators
    // - Computational experiments finished
  ],
  placement: top,
  scope: "parent",
) <tab:timeline>


= Next Steps & Timeline

// ADVISOR COMMENT: Nature Machine Intelligence is ambitious but achievable. Key requirements:
// 1) Novel biological insight (not just technical contribution)
// 2) Rigorous validation with domain experts
// 3) Open-source code and pretrained models
// 4) Broader impact discussion (how this transforms biological research)

We plan to submit to Nature Machine Intelligence.
In order to do so, I think we must:
+ Describe a new scientific discovery (a novel and meaningful trait, ideally 3-5).
+ Compare against baseline methods on a variety of datasets, using statistical tests to validate improvements over multiple metrics.
+ (Ideally) develop an update to SAEs in order to enable this
See @tab:timeline for a plan on achieving this goal.

// ADVISOR COMMENT: Missing critical pieces:
// 1) Collaboration with butterfly experts for validation
// 2) Computational reproducibility (seeds, hyperparameters)
// 3) Failure cases - when do SAEs not find meaningful traits?
// 4) Theoretical analysis - why should SAEs find biological traits?
