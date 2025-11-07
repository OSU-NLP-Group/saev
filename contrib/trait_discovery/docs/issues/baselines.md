# Mini-batched GPU k-means & PCA baselines

## Summary
- Add GPU-capable k-means and PCA trainers under `contrib/trait_discovery` so we can generate dense trait discovery baselines without blocking on CPU throughput.
- Run a single, carefully chosen configuration per method unless downstream results motivate a sweep; in practice we still need to nail down a few implicit hyperparameters (clusters, components, seeds, batch sizing).
- Produce per-patch centroid distance (or similarity) matrices for ADE20K train and val splits, then feed them into the existing probe1d evaluation after extending it to accept dense inputs.

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
- Plan: iterate through dataset shards, compute distances on GPU, then flush each shard to disk (e.g., Zarr chunked by `(patches_per_shard, n_clusters)` with `patches_per_shard ~ 4096`). Validation can reuse the same pipeline with a different split flag.
- Need to ensure feature normalization stays consistent between training and inference. Probably store the per-feature mean/std (or PCA whitening matrix) and assert at load time that the activation tensor agrees with those stats.
- For PCA we can treat component activations as signed similarities. For k-means we likely want negative euclidean distance or cosine similarity; storing `1 - normalized_distance` keeps values in `[0, 1]`, but we should also support raw squared distances so downstream models can choose their own transformation.

## Dense distance materialization strategy
- GPU batching: given centroids `centroids_sd` and patch activations `acts_bd`, compute pairwise distances using `torch.cdist` or a manual `||x||^2 - 2x^T y + ||y||^2` formulation to reuse precomputed norms. Stick with fp16/bfloat16 accumulations but upcast to fp32 for the norm sums to avoid catastrophic cancellation.
- Sparsity heuristic: optionally zero out entries whose similarity falls below a configurable margin (e.g., keep top-k per patch or threshold at zero after shifting by `1 - distance / tau`). Even if we keep dense tensors, the thresholding logic lets us fall back to CSR if the storage blow-up becomes unmanageable.
- Storage: write chunks to `.pt` or `.npz` files with an index manifest that maps `(split, shard_id)` to file paths. Include asserts for expected shapes so later readers do not silently ingest truncated tensors.
- Validation: after each shard write, compute per-cluster utilization histograms to ensure no centroid collapses. Log those histograms to wandb for traceability.

## Updating `probe1d.py` for dense activations
- The script currently assumes sparse CSR storage; we need an interface like `ActivationStore` that can hand back either sparse or dense batches transparently.
- Add a code path that skips sparse-only ops (e.g., `torch.sparse.mm`) and keeps everything dense when the source store advertises `format="dense"`.
- Guard existing sparse-specific logic with early returns so dense inputs do not hit CSR-only assertions. Tests should cover both formats.
- Since dense probes may OOM, add CLI flags for batch size, precision, and optional on-the-fly top-k pruning before the probe to reduce memory.

## Additional open questions
- Do we also need PCA reconstruction error per patch for diagnostics, or are raw component activations enough?
- Should we keep a second set of centroids/components trained on a different random seed to estimate variance, even if we do not run a full sweep?
- Where do we surface the dense features for downstream trait discovery (e.g., new config entries in `train_baseline.py`)?
- How do we share ADE20K feature statistics with other baselines so they stay consistent (central JSON schema vs. ad hoc CLI args)?
- Can we reuse the upcoming dense distance backend for other datasets (coco-stuff, fish-vista), or should we refactor this into a dataset-agnostic module now?
