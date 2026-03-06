# Mini-batched GPU k-means & PCA baselines

## Summary
- Add GPU-capable k-means and PCA trainers under `contrib/trait_discovery` so we can generate dense trait discovery baselines without blocking on CPU throughput.
- Run a single, carefully chosen configuration per method unless downstream results motivate a sweep; in practice we still need to nail down a few implicit hyperparameters (clusters, components, seeds, batch sizing).
- Produce per-patch centroid distance (or similarity) matrices for ADE20K train and val splits, then feed them into the existing probe1d evaluation after extending it to accept dense inputs.
- Emit reusable tensors and metadata so notebooks can iterate on plots without forcing the trainers to generate figures.

## Objectives & scope
- Training: Launch mini-batched k-means and PCA on Slurm GPUs, reusing the trait discovery job infrastructure (`train_baseline.py`, `scripts/slurm_*.sh`).
- Evaluation: Materialize distances/similarities from every ADE20K patch to every centroid/component for both train and val splits; aggregate summary metrics plus diagnostics that establish convergence quality.
- Integration: Update `probe1d.py` (and any helper readers) so dense activations flow without pretending to be sparse.
- Deliverables: (1) checkpoints for centroids / PCA components, (2) chunked distance tensors, (3) documentation for the run recipe, and (4) sanity plots/tables in `contrib/trait_discovery/docs/reports`.

## Training jobs on Slurm/GPU
- Reuse the existing Slurm launcher pattern: a single `train_kmeans.py` and `train_pca.py` entry point that accept dataset shards, batch size, number of iterations, and logging destinations. Both scripts can share a mini-batch feature iterator to avoid duplication.
- For k-means we still pick `n_clusters`, mini-batch size, learning rate/EMA for centroid updates, and initialization seed. Even if we only run one configuration, we should document defaults (e.g., `n_clusters=4096`, `batch_size=8192` patches, `n_init=4`, `max_iter=200`). PCA minimally needs `n_components`, whether we whiten, and a convergence tolerance.
- Jobs should stream features directly from the ADE20K activation store (probably the same HDF5/Zarr used elsewhere) and keep centroids/components on the GPU. We can interleave host-to-device transfers via pinned-memory DataLoader workers to ensure the GPU stays saturated.
- Logging: capture inertia / reconstruction error per epoch, effective learning rate schedule, and wall-clock throughput inside wandb so we can compare to future baselines.
- Checkpointing: persist centroids/components plus optimizer state every N steps so we can resume if Slurm preempts us. Store artifacts under `contrib/trait_discovery/results/baselines/{kmeans,pca}/` with metadata JSON that records feature spec and seeds.

## Dataset & feature handling
- ADE20K has roughly 20k train and 2k validation images; with 256 patches/img we expect `n_patches_train ~= 20k * 256 = 5.12M` and `n_patches_val ~= 2k * 256 = 512k`. If we store 16k centroid distances per patch in fp32 that is `512k * 16k * 4 bytes ~= 37 GB`, so we must stream and shard outputs.
- Plan: iterate through dataset shards, compute distances on GPU, then flush each shard to disk (probably `.bin` shards with manifest JSON, mirroring the ViT activation protocol). Validation reuses the same pipeline with a different split flag.
- Need to ensure feature normalization stays consistent between training and inference. Probably store the per-feature mean/std (or PCA whitening matrix) and assert at load time that the activation tensor agrees with those stats.
- For PCA we can treat component activations as signed similarities. For k-means we likely want negative euclidean distance or cosine similarity; storing `1 - normalized_distance` keeps values in `[0, 1]`, but we should also support raw squared distances so downstream models can choose their own transformation.
- The protocol in `docs/src/developers/protocol.md` already defines sharded `.bin + metadata.json`; extend it with centroid/component-specific metadata instead of introducing zarr just for these baselines.

## Dense distance materialization strategy
- GPU batching: given centroids `centroids_sd` and patch activations `acts_bd`, compute pairwise distances using `torch.cdist` or a manual `||x||^2 - 2x^T y + ||y||^2` formulation to reuse precomputed norms. Stick with fp16/bfloat16 accumulations but upcast to fp32 for the norm sums to avoid catastrophic cancellation.
- Sparsity heuristic: optionally zero out entries whose similarity falls below a configurable margin (e.g., keep top-k per patch or threshold at zero after shifting by `1 - distance / tau`). Even if we keep dense tensors, the thresholding logic lets us fall back to CSR if the storage blow-up becomes unmanageable.
- Storage: write chunks to `.pt` or `.npz` files with an index manifest that maps `(split, shard_id)` to file paths. Include asserts for expected shapes so later readers do not silently ingest truncated tensors.
- Validation: after each shard write, compute per-cluster utilization histograms to ensure no centroid collapses. Log those histograms to wandb for traceability.

## Notebook outputs and ADE20K table coverage
- All plotting and reporting stays inside `contrib/trait_discovery/notebooks`, so trainers should only dump structured data (distances, activations, metadata JSON). Keep file names deterministic so a notebook can glob a directory and iterate quickly without SQL or wandb.
- The ADE20K baseline table in the paper needs dictionary metrics (MSE, L0, PCA reconstruction error) plus downstream probe metrics (Probe R, mAP, Purity@k=16, Coverage@tau=0.3). Persist these per-layer so notebooks can pick the best layer via the same argmin logic already used for SAE runs.
- PCA must store per-patch reconstruction error (mean over components is fine, but keep the scalar per patch so we can recompute aggregates). k-means should store per-iteration inertia and the distance to the assigned centroid so we can report a comparable MSE entry.
- Emit a small manifest per run (JSON or CSV) that captures the ad-hoc CLI args (cluster counts, component counts, learning rate, seeds). We can document the canonical values inside this `baselines.md` file so readers know exactly how we populated the table.

## Storage format decision
- Continue using the homebrewed numpy shard protocol (binary files plus metadata JSON) for the dense centroid-distance tensors. It already satisfies the requirements from `docs/src/developers/protocol.md`, and sticking with it avoids introducing zarr-specific tooling.
- If we later discover that multi-writer workflows or cloud object storage would benefit from zarr, we can switch, but for now the simplest solution is to reuse the established protocol and add asserts/tests for the new tensor shapes.

## Updating `probe1d.py` for dense activations
- The script currently assumes sparse CSR storage; we need an interface like `ActivationStore` that can hand back either sparse or dense batches transparently.
- Add a code path that skips sparse-only ops (e.g., `torch.sparse.mm`) and keeps everything dense when the source store advertises `format="dense"`.
- Guard existing sparse-specific logic with early returns so dense inputs do not hit CSR-only assertions. Tests should cover both formats.
- Since dense probes may OOM, add CLI flags for batch size, precision, and optional on-the-fly top-k pruning before the probe to reduce memory.

### Dense probing risks
- Memory pressure: a single shard of `5k patches x 4k centroids` in fp16 is ~160 MB; batching multiple shards or enabling fp32 doubles that. `probe1d.py` needs chunked readers and `torch.cuda.Stream` overlap so we never materialize more than a couple hundred MB at once.
- Bandwidth/compute: dense matmuls to apply linear probes scale with `O(n_patches * n_features)`. We should fuse normalization and linear layers into one `torch.addmm` to keep kernel launches down and rely on cublasLt for mixed precision.
- Checkpoint compatibility: existing CSR checkpoints expect `indices/indptr`. We should version stamp new dense checkpoints so older runs fail loudly instead of misinterpreting tensors.
- Metrics parity: dense activations remove the implicit sparsity regularization that SAEs provide. We should document that probe loss comparisons between sparse and dense inputs may need normalization (e.g., divide by feature count) and include that normalization inside the notebook metrics helpers.
- Debuggability: add asserts on NaNs/inf after each dense batch and log max abs activation to wandb so we catch divergence early—harder to eyeball when everything is dense.

## Additional open questions
- Should we keep a second set of centroids/components trained on a different random seed to estimate variance, even if we do not run a full sweep?
- Where do we surface the dense features for downstream trait discovery (e.g., new config entries in `train_baseline.py`)?
- How do we share ADE20K feature statistics with other baselines so they stay consistent (central JSON schema vs. ad hoc CLI args)?
- Can we reuse the upcoming dense distance backend for other datasets (coco-stuff, fish-vista), or should we refactor this into a dataset-agnostic module now?
- Should the ad-hoc CLI defaults (cluster counts, PCA components, thresholds) live here permanently or graduate into checked-in config files once they stabilize?

## Precise Experimental Design

Baseline decompositions for vision activations (k-means and PCA) with probe-based evaluation on semantic segmentation labels.

1. Mini-batch k-means with L0 = 1 coding (one-hot).
2. PCA with a Top-K sparsity sweep: K in {1, 4, 16, 64, 256, 1024}.

The goal is to compare reconstruction (NMSE), sparsity (L0), and downstream segmentation-probe performance to SAEs under matched representational budgets.

## Data and splits

- Upstream fitting data: ImageNet-1K, train split (IN1K/train), same images used to fit SAEs.
- Validation for reconstruction: ImageNet-1K, val split (IN1K/val).
- Downstream probing: ADE20K train and val splits (patch-level labels).

## Sweeps

- Sweep over the same set of DINOv3 layers as used for SAEs (e.g., the last 6 blocks including the final block).
- No other hyperparameters are swept for the k-means or PCA baselines.

## k-means with L0 = 1

Fit:

- K (number of clusters) = 16384 (matches SAE dictionary size).
- Algorithm: MiniBatchKMeans

Encode (L0 = 1):

- One-hot top-1: for each token, find nearest center j* and output a one-hot vector s where s[j*] = 1 and 0 otherwise.

Reconstruction for normalized MSE:

- Nearest-centroid reconstruction: x_hat = c_nn (the nearest center in the standardized space, un-standardize only if you want to report raw MSE).
- Normalized MSE (NMSE) is computed on IN1K/val as: NMSE = mean(||x - x_hat||_2^2) / mean(||x - mu||_2^2), where mu is the per-layer IN1K/train mean prior to standardization and the means are over tokens and images. We compute that mu in a separate script for each dataset.

Downstream probing (ADE20K):

- For each feature j (cluster index) and class c:
  - The feature scalar per token is s_j(x) from the chosen encoding.
  - Fit a 1D logistic probe y ~ sigmoid(b + w * s_j) on ADE20K/train, with:
    - L2 penalty on w matching the SAE probe setting (lambda identical),
    - identical optimizer, convergence checks, and early stopping as for SAEs.
  - Evaluate on ADE20K/val and log:
    - probe cross-entropy (CE),
    - optional AP computed on the logit (b + w * s_j) if enabled.
- Also log a reference baseline CE for each class using an intercept-only model (predicts the training prevalence).

Note: this is already implemented in src/tdiscovery/probe1d.py.

## PCA with Top-K sparsity sweep

Fit:

- PCA fitted once per layer on IN1K/train.
- n_components = d (full-rank PCA for that layer).
- Implement a mini batch PCA.

Component orientation:

- PCA components have arbitrary sign. After fitting, orient each component so that the mean coefficient over IN1K/train is nonnegative: sign_j = sign(mean_z_j), where z = U^T (x - mu). Multiply column U[:, j] and all z_j by sign_j if sign_j < 0. This stabilizes reporting and probing but does not change reconstruction.

Encoding:

- Dense coefficients: z = U^T (x - mu) on standardized inputs.
- Top-K sparsification for K in {1, 4, 16, 64, 256, 1024}:
  - Keep the K entries of z with largest absolute value per token.
  - Form z_K by zeroing all other entries. For probing, use the signed z_K values. For reconstruction, use only the kept entries.
- Implementation detail: the inference CLI exposes this as `--latent-topk`. Passing `--latent-topk 64` writes artifacts under `inference/<hash>/pca-topk-00064/`. Omitting the flag keeps dense PCA codes and stores them under `inference/<hash>/pca-dense/`.

Reconstruction for NMSE:

- Reconstruct with the kept components only:
  x_hat = mu + U_K z_K
  and compute NMSE on IN1K/val as above. Report the average L0 (which equals K).

Downstream probing (ADE20K):

- For each component j and class c:
  - Feature scalar is z_j (or 0 if j not kept by Top-K for that token).
  - Standardize per-feature over ADE20K/train before probing.
  - Fit the same 1D logistic probe as above and evaluate on ADE20K/val.
- Report probe CE (and optional AP) at each K in the sweep.

## Reporting

For each layer and method setting:

Reconstruction:

- IN1K/val NMSE.
- For k-means, report L0 = 1 by construction.
- For PCA, report the K used (L0 = K).

Downstream (ADE20K):

- For each (feature j, class c): train CE, val CE, and optionally AP on val.
- Aggregate views:
  - Per-class best feature (min val CE) and its margin to the baseline CE.
  - Mean CE across all features for each class (useful but noisier).
  - Top-N features per class by AP if AP is enabled.

## Implementation notes

- Tokenization: treat each patch token independently; CLS is discarded.
- Standardization: reuse the exact mu and sigma tensors from SAEs for each layer.
- Distance metric: Euclidean on standardized tokens for nearest-centroid search.
- Inverse-distance coding: use eps = 1e-6; no scaling or temperature.
- Feature standardization for probing: compute mean/std on ADE20K/train, and apply to ADE20K/train and ADE20K/val consistently.
- Class weighting: use the same scheme as SAEs (e.g., inverse-frequency).
- Randomness: set and log random_state for PCA and for k-means; log seed per run.
- No hyperparameter sweeps beyond the specified layer sweep and PCA Top-K values.
- Hardware and batching: both MBKMeans.partial_fit and IncrementalPCA can stream shards of tokens from disk; respect the same shuffled order used for SAEs for reproducibility.
- Ignore or mask void/ignore pixels when deriving patch labels for ADE20K (majority vote within the patch; break ties by preferring non-void if any).

## Rationale for these choices

- One-hot k-means gives a simple, strictly L0 = 1 code with no extra parameters.
- The inverse-distance variant is included only as a parameter-free ablation that adds a notion of confidence; one-hot is the primary baseline.
- PCA Top-K provides a sparsity-controlled dense linear basis to compare against SAEs on equal L0 budgets.
- NMSE is variance-normalized and matches the SAE reconstruction metric, enabling direct comparison across methods and layers.
