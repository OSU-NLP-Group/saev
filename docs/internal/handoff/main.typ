#let title = [SAEs for Discovery: Handoff Plan]

#set page(
  paper: "us-letter",
  numbering: "1",
  margin: 1in,
)
#set par(justify: true)
#set text(size: 11pt)
#set heading(numbering: "1.1.")

#show link: set text(blue)

#align(center, text(17pt)[*#title*])
#align(center)[Sam Stevens #h(1em) March 2026]

#v(1em)

This document is a handoff plan for the "SAEs for Discovery" project.
It describes the current state, what remains to be done, who might take it forward, and what risks to watch for.
The goal is that a successor can read this document, understand the project, and pick up where I left off.

= Background

Large self-supervised vision models (DINOv2, DINOv3, BioCLIP 2, SigLIP 2) learn rich representations of the visual world.
Sparse autoencoders (SAEs) decompose these entangled representations into individual, human-interpretable features.
This project applies SAEs to biological images to discover morphological traits---both recovering known traits and surfacing novel ones that biologists have not yet catalogued.

The core thesis: *SAEs can serve as a bridge between what vision models implicitly learn and what biologists can use for science.*

= Pipeline Overview

The end-to-end discovery pipeline has six stages.
Each stage has working code, but the stages are not yet connected into a single streamlined workflow.

+ *Record ViT activations.*
  Given a dataset of organism images, extract patch-level activations from a vision transformer (e.g., DINOv3 ViT-L/16).
  Activations are stored on disk in a shard format (`saev` library).
  Entry point: `uv run launch.py shards`.

+ *Train SAEs.*
  Train sparse autoencoders on the recorded activations, sweeping over learning rate and sparsity (top-$k$).
  Entry point: `uv run launch.py train`.

+ *Select Pareto-optimal checkpoints.*
  From the sweep, pick checkpoints on the Pareto frontier of sparsity (L0) vs. normalized MSE.
  This is currently done in Marimo notebooks (e.g., `contrib/trait_discovery/notebooks/007_cambridge.py`).
  There is no automated script for picking the best checkpoints on the frontier; it requires judgment and a bit of brute force search.

+ *Run SAE inference on target dataset.*
  The training data (e.g., ImageNet-1K) may differ from the dataset of interest (e.g., Cambridge butterflies).
  Record ViT activations on the target dataset, then run the Pareto-optimal SAE checkpoints to get per-patch SAE activations.
  Entry points: `uv run launch.py shards` to save activations, like before, then `uv run launch.py inference` to get the SAE activations saved in a sparse array format.

+ *Evaluate with probes.*
  If ground-truth segmentation maps or trait labels exist, fit 1D logistic regression probes to measure how well individual SAE latents predict specific classes.
  Entry point: `uv run contrib/trait_discovery/scripts/launch.py probe1d`.
  Key metric: Probe $R$ (explained variance).

+ *Explore and discover.*
  Combine SAE activations with biological metadata (habitat, trophic level, migratory status, etc.) to find features correlated with biologically meaningful patterns.
  Visualize where features fire on images.
  This is the least codified stage---it happens in notebooks, ad-hoc scripts, and interactive tools.
  See `contrib/mimics/`, `contrib/birdsong` and `contrib/trait_discovery` for examples and context on a proposed interactive exploration framework.

== Key Conventions and Tribal Knowledge

- *Shard layout:* ViT activations are stored in a custom shard format on disk. See `docs/src/developers/disk-layout.md` and `docs/research/issues/big-activations.md`. This is one of the most confusing parts of the codebase for newcomers.
- *Config system:* Configs are Python dataclasses. Sweeps are Python files that define a `make_cfgs()` function returning a list of config dictionaries. The `launch.py` scripts read these and submit SLURM jobs via submitit. Most of the existing sweep files hardcode user-specific run and shard locations. For example:
```python
# contrib/trait_discovery/sweeps/003_auxk/inference.py
run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

dinov3_fishvista_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/5dcb2f75"
dinov3_fishvista_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/8692dfa9"
dinov3_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0"
dinov3_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3802cb66"

```

- *`contrib/` structure:* The core `saev` library lives in `src/saev/`. Downstream experiments (trait discovery, interactive interpretation, beetle traits) live in `contrib/`. Each contrib has its own `scripts/launch.py`, sweep configs, and notebooks.
- *Run hashes:* Activation shard directories are identified by content hashes (e.g., `69e6d6fd`). These appear in sweep configs and must match the actual shard directories on the cluster.

= Current State by Taxon

== Butterflies (Most Promising)

*Dataset:* Cambridge butterfly dataset (Heliconius mimic pairs).

*Task:* Find SAE features that discriminate mimic pairs---butterflies that look nearly identical to humans but are different species. This is the clearest research question because biologists are already interested in this mimic discrimination problem.

*Status:* Furthest along of all three taxa.
- SAEs trained on DINOv3 ViT-L/16 with 640 patches at multiple layers and sparsity levels.
- Some ablations partially explored (resolution, SAE width, encoder choice) but results not saved or documented systematically.

*Biological collaborators:*
- Neil Rosser
- Dan Rubenstein
- Christopher Lawrence

These are the people who would validate whether a discovered feature is biologically real.

*What remains:*
- Additional ablations: higher resolution (more patches), wider SAEs, different vision encoders.
- Biological validation: show candidate features to collaborators and get feedback.
- The core challenge: if a discriminating feature were trivially visible, humans would have found it already. SAEs need to surface something subtle.
- UI/UX for inspecting many features manually. You want SAE features that are spatially coherent, discriminative, have good recall, etc. and you need to flip through visual examples quickly while prioritizing different tradeoffs between these factors.

*Key files:*
- `contrib/trait_discovery/sweeps/007_cambridge_butterflies/` --- training, inference, and visualization configs.
- `contrib/trait_discovery/notebooks/007_cambridge.py` --- Marimo notebook for Pareto analysis.

// TODO: What is the more general version of this task? all similar species. describe it as a method rather than a specific instance of a problem.

// TODO: Organize by general task; go beyond organismal biology.

== Fish

// TODO: Combine ecological/environmental/evolutionary metadata with what's common to X but not Y?

// TODO: get Jake back up to speed.

*Dataset:* FishVista (images with body-part segmentation maps).

*Task:* Combine SAE features with rich metadata (habitat, trophic level, migration patterns) to find features like "tail fins of bottom feeders" or "mouths of migratory fish."

*Status:* Moderate progress.
- DINOv3 and BioCLIP 2 activations recorded. SigLIP 2 activations not yet recorded.
- SAEs trained on DINOv3 and BioCLIP 2. SigLIP 2 SAEs not yet trained.
- Probe1d results available for DINOv3 and BioCLIP 2 on FishVista segmentations.
- The metadata integration step (joining SAE activations with habitat/trophic data) is undeveloped.

*Collaborators:*
- Zach (Ohio State) --- preliminary contact, not yet developed into active collaboration.

*What remains:*
- Complete SigLIP 2 pipeline (activations, SAE training, inference, probes).
- Develop the metadata integration workflow.
- Find a biological champion with a clear research question.

*Key files:*
- `contrib/trait_discovery/sweeps/` --- look for fish-related sweep configs.
- `contrib/trait_discovery/docs/reports/todo.md` --- detailed checklist of completed and remaining tasks.

== Beetles

*Dataset:* ~60K NEON ground beetle (Carabidae) images with collection metadata (location, date, site ID).

*Task:* Fully exploratory---discover unknown morphological traits, potentially correlate with climate/land-use data.

*Status:* Early stage.
- A couple SAEs trained during Funkapalooza (August 2025), but work stopped there.
- Project brief exists at `contrib/beetle_traits/brief.typ`.

*Collaborators:*
- Eric Sokol (NEON) --- was excited about initial results.
- Sydne Record (UMaine)
- Alyson East (UMaine)

*What remains:*
- Essentially everything: systematic SAE training, inference, probe evaluation, and exploration.
- Need to re-engage Eric Sokol or find another biological champion.

= Quantitative Evaluation Framework

The paper-in-progress (see `docs/research/reports/2025-10-24/report.typ`) establishes a quantitative evaluation methodology:

+ *ADE20K re-discovery:* Use ADE20K semantic segmentation as a benchmark. Fit 1D probes on each (feature, class) pair, measure Probe $R$, mAP, Precision\@$k$, and Coverage\@$tau$. Compare Matryoshka SAEs against vanilla SAEs, $k$-means, PCA, and NMF.
+ *FishVista body parts:* Same methodology on a domain-specific dataset. Compare DINOv3, SigLIP 2, BioCLIP 2 to show domain-specific ViTs produce better features.
+ *Ablations:* ViT size, image resolution, ViT layer, sparsity tradeoffs.

= Deliverables Before Departure

Given a two-week window:

+ *This handoff document* (you are reading it).
+ *Code cleanup:* Ensure the most confusing parts have inline comments and that sweep configs are up to date.
+ *In-person meeting:* Present this plan to the successor(s) and advisor. The key decision the meeting must produce: who owns this going forward?

= Ownership and Continuity

*Potential successor:* Jake (2nd-year PhD) could take ownership if he wants to push it forward.

*Staff support:* Net (staff data scientist) is available for engineering/infrastructure work.

*External labs (aspirational, no commitments):*
- David Rolnick's lab
- Sara Beery's lab

*The realistic goal* (from advisor): be satisfied that the process of SAEs for discovery is documented well enough that a computationally savvy biologist (someone who can modify Python code, run SLURM jobs, and interpret results) can use the tool, AND that there is a clear owner to push it forward.

= Risks

Three risks that could stall or kill the project:

+ *No biological champion.* Without a biologist driving research questions, the CS side doesn't know what to look for. The butterfly collaborators (Rosser, Rubenstein, Lawrence) are the strongest existing relationship. For fish and beetles, collaborations are nascent.

+ *Priority drift.* This is exploratory research. Against paper deadlines and other obligations, it can easily lose priority. A clear owner with protected time is essential.

+ *Tooling complexity.* The learning curve to use the pipeline is steep. A new person must understand: the shard format, the config system, the sweep/Pareto selection workflow, the `contrib/` structure, SLURM job submission via submitit, and the ad-hoc notebook-based exploration. Without active mentorship during the transition, this could block progress entirely.

= Infrastructure and Data

- *Code:* Currently under the `OSU-NLP-Group` GitHub org. Needs discussion with Tanya about whether to transfer to the Imageomics GitHub org.
- *Cluster data:* Trained checkpoints, activation caches, and intermediate results live on shared cluster scratch/project space (e.g., `/fs/scratch/PAS2136/`). This data persists but is regenerable from the configs if lost.
- *No personal storage risk:* All important data is on shared infrastructure, not personal directories.

= If Nobody Takes Over

If no clear owner emerges, the project partially survives:
- The core `saev` library continues to work as a standalone tool for training and evaluating sparse autoencoders on vision model activations.
- The biological discovery applications (butterflies, fish, beetles) would stall without an active researcher driving the exploration and collaborator relationships.
- The code and data will persist on shared infrastructure and could be picked up later, but the momentum, context, and collaborator relationships would be lost.

= Recommended Next Steps for a Successor

+ *Start with butterflies.* It has the clearest question (mimic discrimination), the most progress, and active biological collaborators.
+ *Run the remaining ablations:* higher resolution, wider SAEs, different encoders. Configs exist or can be adapted from existing sweep files.
+ *Get biological validation:* Show candidate features to Rosser/Rubenstein/Lawrence. Their feedback determines whether you have a discovery or just an interesting visualization.
+ *Build the metadata integration workflow for fish.* This is the most impactful tooling investment: a systematic way to join SAE activations with structured biological metadata.
+ *Re-engage beetle collaborators* if bandwidth allows. Eric Sokol's enthusiasm is an asset.
+ *Invest in documentation and UX* for the exploration stage. The training and inference pipeline works; what's missing is effective tooling for the "stare at features and form hypotheses" phase.



// 4th task: open set, evidence for a new species
// Maybe a computational version of decision trees/dichotomous keys, tell me what an individual sample has/doesn't have -> candidate for a new species.