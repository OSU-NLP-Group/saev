# 03/11/2026

Today I want to give everyone an introduction to the project, how I think it can work, and a possible six weeks of concrete steps.

This is just me thinking out loud.

What are the steps?

- Get relevant image data
  - EPA electrofishing (since 1987):  potentially millions of fish images. Zach has EOB collaborator who might provide access. Should also check Tanya can help us get access; might add to ToL-200M.
  - Ohio freshwater fish:  ~170 species. Zach will send metadata including river size and non-visual traits. We likely have images from BioClip 2 ToL-200M training data.
  - FishVista? The images came from "museum collections of fish images publicly available at GLIN [3–6, 8, 9, 16], IDigBio [7], and Morphbank [1] databases. We acquired these images along with their associated metadata including species names and licensing information from the FishAIR [2] repository. In total, we collected 56,481 images from GLIN, 41,505 from IDigBio, and 9,000 from MorphBank." But it seems they did a lot of post-processing.
  - Crop the images to the fish itself using moondream or perceptron
- Record ViT activations.
  - Which ViT? What resolution? What layers? Probably DINOv3, but resolution is a key question.
- Train an SAE sweep using the TopK + AuxK loss.
  - Pretty easy. SAE width is a question, as well as some optimizational stuff (dead latents, learning rate schedules, etc)
- Pick out the pareto frontier
  - Again, pretty easy. Use a notebook.
- Combine with metadata to find "interesting" traits
  - Metadata sources:
    - FishBase
- Train a decision tree for classification
- Train logistic regression (linear probing) and decision tree classifiers on DINOv3 activations directly (an upper bound on dichotomous key)
- Train logistic regression (linear probing) and decision tree classifiers on SAE activations directly (another upper bound on dichotomous key)

Each of these have a bunch of substeps and specific decisions that need to be explored.

