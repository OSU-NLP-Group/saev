#let title = [SAEs for Scientific Discovery: Status Report October 3rd, 2025]

#set page(
  paper: "us-letter",
  numbering: "1",
  columns: 1,
)
#set par(justify: true)
#set text(size: 11pt)
#set math.equation(numbering: "(1)")

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
      This report summarizes progress on using sparse autoencoders (SAEs) to discover morphological traits in biological images.
    ]
  })
]

= Introduction

= Progress

- Trained models on butterfly-patch only wings. There are a couple bugs:
  + Some of the butterflies are missing segmentation masks. I need to iteratively update the reference masks using my GUI until the quality is better.
  + Some of the butterflies have flipped (180 degree rotation) masks. I need to investigate why this is happening (EXIF data??)
- Tried to implement sparse 1D probing for evaluating sparse autoencoder quality in terms of how well the SAEs learn patch-level segmentation labels. This has precendent For instance, 

= Evaluting with 1D Probes

#cite(<gao2025scaling>, form: "prose") suggest using a set of known concepts, like a sentiment in language or a texture in vision, as binary classification tasks for evaluating sparse autoencoder quality.
Each concept has some positive and negative examples; we pick out the SAE hidden feature that provides the best signal for predicting positive vs negative examples of a concept.
That is, we optimize both the weight and bias terms in a logistic probe _and_ the specific SAE feature that is input to the logistic probe, and evaluate an SAE by how good a logistic probe we can find for predicting presence/absence of a particular *concept*.
When we do this for more than one concept, we have a benchmark!
$ limits(min)_(i, w, b) EE [ y log (sigma (w z_i + b)) + (1 - y) log (1 - sigma (w z_i + b)) ] $ <eq:objective>
Specifically, we want to minimize @eq:objective where $z$ is the post-activation vector for the SAE, and $i$ is the the particular SAE latent , but since $i$ is not continuous, we optimize $w$ and $b$ for all possible values of $i$, then choose the optimal $i$.
There is no reference implementation for this particular kind of evaluation.
As a result, I have implemented it myself using sparse matrices on a GPU.
I haven't tried it yet because of some other blockers (see below).

= Refactors

+ #link("https://osu-nlp-group.github.io/saev/api/")[User-facing documentation].
+ [in progress] Update disk layout to make it easier to refer from a particular run to the original sharded activations and the image dataset on disk.
+ [in progress] Removing references to images and vision transformers in order to support non-vision but still bi-directional transformers, like audio or other modalities.

These refactors are taking longer than I thought.
Some of the tests should also be refactored to make them shorter.

Some thoughts on enforcing both code and documentation quality:

+ I could ask Claude Code to find logical bugs in the framework every night.
+ I could ask Claude to train an SAE itself. This seems very possible with the regular desktop Claude application because it has access to a little docker image. Maybe we could do the same thing with Codex Cloud with network access, or Claude Code/Codex CLI in a docker container?

#bibliography("../refs.yml")
