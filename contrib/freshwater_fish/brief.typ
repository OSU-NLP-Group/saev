#set page(paper: "us-letter", margin: (x: 1.5in, y: 1.5in))
#set par(justify: true)

#show link: set text(blue)

#align(center)[
  #text(16pt, weight: "bold")[SAE-Generated Dichotomous Keys for Freshwater Fish]
]

_Dichotomous keys are hand-built decision trees that biologists use to identify species. They take significant expertise to create, are brittle (one wrong answer cascades through the tree), and are hard to update. Recent work suggests that sparse autoencoders (SAEs) can recover known morphological traits from vision transformer (ViT) activations without supervision. This project asks: can we use SAE-discovered features to build a better dichotomous key for Ohio's \~170 freshwater fish species?_

== Motivation

Fish ecomorphology is deeply structured: mouth position predicts feeding strategy (superior #sym.arrow.r surface feeder, inferior #sym.arrow.r bottom feeder), body shape predicts swimming behavior through hydrodynamics, and these traits correlate with fine-grained microhabitats (riffles, mid-stream, stream margins). Existing dichotomous keys#footnote[#link("https://www.macroinvertebrates.org/key")[Example] dichotomous key for macroinvertebrates (bugs).] encode this knowledge as hand-curated binary decisions, but they have practical problems: if a diagnostic feature is damaged, absent, or ambiguous on an individual, the key fails with no error tolerance.

SAEs trained on vision transformer activations learn interpretable features that correspond to specific visual traits. These features could be the basis for keys that are both more robust (tolerating mistakes and missing information) and more informative (connecting visual traits to non-visual biology like diet, reproduction, and habitat preference). Training a decision tree on SAE features is straightforward, but most SAE features are not useful to a human in the field. The core challenge is formalizing what makes a feature both predictive and human-interpretable: which features can a biologist actually observe and act on?

== Goal

Produce a *new dichotomous key* for Ohio freshwater fish, using SAE-discovered features and validated by a domain expert. A secondary goal is a paper describing the method and comparing the generated key to existing ones. A stretch goal is predicting non-visual traits (diet, reproduction style, river size preference) from SAE visual features.

== Data

- \~170 Ohio fish species. Zach can provide metadata like river size and non-visual traits.
- \~500K images of fish from the #link("https://arxiv.org/abs/2505.23883")[BioCLIP 2] training set (#link("https://huggingface.co/datasets/imageomics/TreeOfLife-200M")[Tree of Life 200M]).
- Ohio EPA electrofishing survey data (1987 to present). Access could be requested.
- USGS ecology data as an additional metadata source.
- An IBI (Index of Biotic Integrity) species list.

== Methods

Train SAEs on #link("https://arxiv.org/abs/2508.10104")[DINOv3] vision transformer activations from fish images using the #link("https://github.com/samuelstevens/saev")[`saev`] framework. Extract interpretable features, map them to visual traits, then develop a method for organizing those traits into a decision tree (e.g., greedy optimization over information gain). Validate the resulting key against existing keys with Zach.

== Challenges

+ *Formalizing human-interpretability.* Training a decision tree on SAE features is easy. Deciding which features are both predictive and understandable by a human in the field is the real problem, and formalizing that criterion is a research contribution.
+ *Data curation.* The species list needs structuring, images need filtering, and taxonomy needs updating. This requires close collaboration with Zach.
+ *Feature-to-tree mapping.* Translating SAE features into a sequence of binary decisions that a human can follow in the field is non-trivial, especially when accounting for error tolerance.

*Why this is possible:* The #link("https://github.com/samuelstevens/saev")[`saev`] framework handles SAE training and interactive feature exploration. We have \~500K images and a species list to start from. The scope is manageable (\~170 species, not thousands). Zach has deep expertise in fish taxonomy and existing keys to validate against. The core research question (features to tree) can start with simple approaches (greedy information gain) before getting sophisticated.

=== Possible Extensions

- *Non-visual trait prediction:* predict reproduction, diet, or habitat preference from SAE visual features for species where these traits are unknown.
- *#link("https://www.osc.edu/ywsi_project_ohios_watersheds/sampling/ibi")[Index of Biotic Integrity (IBI)] prediction:* use SAE features to predict stream health scores from fish community images.
- *Freshwater mussels:* many species are federally endangered and existing keys require opening the shell (killing the animal). Same SAE pipeline could apply if images are available (#link("https://mbd.osu.edu/collections/invertebrates/databases")[OSU invertebrate collections]).

== Getting Started

+ Read the two SAE papers linked below#footnote[#link("https://arxiv.org/abs/2511.17735")[Stevens et al. (2025a)] and #link("https://arxiv.org/abs/2502.06755")[Stevens et al. (2025b)].] to understand sparse autoencoders for vision.
+ Install #link("https://github.com/samuelstevens/saev")[`saev`] and work through the user guide to train an SAE on a small image dataset.
+ Meet with Sam and Zach to understand dichotomous keys, see existing ones for Ohio fish, and discuss data availability.
+ Train an SAE on available fish images and explore the learned features using `saev`'s interactive tools.

== Collaborators & Roles

- *#link("https://steflab.weebly.com")[Zachary Steffensmeier]* (OSU, Environment & Natural Resources): Domain expert in stream ecology, fish taxonomy, and aquatic organism identification. Will guide species scoping, data curation, and key validation.
- *#link("https://samuelstevens.me")[Sam Stevens]* (OSU, Computer Science): SAE framework author. Will help onboard and advise on SAE training and feature interpretation.
- *You?*
  - Comfortable writing Python scripts.
  - Some ML background (know what PyTorch is, trained a model, understand gradient descent).
  - Comfortable with command line and SSH.
  - Interest in biology, but no background required. No prior SAE experience needed.
  - At least \~10 hours/week.
  - You will have access to the Ohio Supercomputer Center (OSC) for GPU compute.
