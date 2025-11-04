# Context

We are training sparse 1D probes on ADE20K shards (runs 2766385_*, 2775328_*, 2776042_*) using `probe1d.py`. Recent jobs repeatedly hit out-of-memory failures despite 40 GB A100 GPUs. Telemetry added to `Sparse1DProbe` now records both GPU and host RSS, so we can pinpoint memory spikes.

# Gap

Even after reducing the solver to 30 iterations and raising the Slurm allocation to 120 GB (jobs 2776042_*), slot `2776042_1` still ends with `slurmstepd: OOM` immediately after the solver prints “Fit probe.”. The post-fit metric step (`loss_matrix_with_aux` + `.npz` saves) is uninstrumented and no longer logged before the crash, so we still lack:

- RSS measurements during one-hot label construction and loss/matrix evaluation.
- A breakdown of how much dense memory the confusion-matrix buffers consume per shard.
- A plan to shrink that footprint so we can return to the default 80 GB limit.

# Promise

With the new `log_host_mem` hooks around CSR loads, CUDA transfers, one-hot creation, and `loss_matrix_with_aux`, the very next rerun will tell us exactly where host memory blows past the cgroup. Once we have that trace we can rework label construction and the evaluation path (e.g., slabbed/batched metrics or CPU streaming) to keep RSS safely under 80 GB.

# Solution

Already in place:

- Added a `stats` logger that emits per-iteration JSON with `rss_gb`, GPU peaks, and solver telemetry.
- Added `log_host_mem` events around CSR loads, GPU transfers, one-hot label creation, and both `loss_matrix_with_aux` calls.

Still pending: restructure the train/test metric computation to avoid the extra tens of gigabytes of dense buffers that trigger Slurm’s OOM killer.

# Insight

- The heaviest shard (`2776042_1`, 2.26 B nnz) peaks at ~44.4 GB RSS during optimization; evaluation likely needs a similar amount again, pushing us over the previous 80 GB limit.
- Host RAM, not VRAM, is the limiting factor—the GPU never exceeds ~30 GB.
- The OOM happens *after* fitting, so the trust-region solver is stable; it is the post-processing pipeline that must be optimized next.

# Operational Notes

- Slurm stdout/err for probe runs land in `contrib/trait_discovery/logs/`; filenames follow `%A_%a_%t_log.out` and the matching `.pkl` files store the submitit payload (use `uv run python -c "…"`, then inspect `DelayedSubmission.args[0]`).
- Older sweeps may also have artifacts in the repo-root `logs/` directory, but the ADE20K probe jobs discussed here all write under `contrib/trait_discovery/logs/`.
- The ADE20K shards for these runs live on OSC scratch at `/fs/scratch/PAS2136/samuelstevens/saev/shards/<shard_id>/`, with the associated inference outputs (CSR activations) under `/fs/ess/PAS2136/samuelstevens/saev/runs/<run_id>/inference/<shard_id>/`.
- The largest failing shard so far is `781f8739` (train) paired with `5e195bbf` (test); its CSR activations reach 2.26 B nnz and drive peak host RSS.
- For the full directory map (where `runs/`, `shards/`, checkpoints, etc. reside), see `docs/src/developers/disk-layout.md`; it complements these ADE20K-specific hints.
