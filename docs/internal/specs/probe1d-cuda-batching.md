# Probe1D CUDA batching

## Context

- Slurm job 2688335 (`contrib/trait_discovery/logs/2688335_0_log.out`) failed on October 23, 2025 because `Sparse1DProbe.fit` tried to keep an 11.4 GiB activation matrix on `cuda:0`.
- `contrib/trait_discovery/logbook.md` documents repeated friction with GPU VRAM limits and noisy device transfers while iterating on the sparse probe.
- The current implementation moves CSR components and dense views to the GPU inside every Levenberg-Marquardt (LM) iteration, so cuda-host synchronization dominates runtime once matrices stop being strongly sparse.
- We need a design that lets `y` live comfortably on the GPU while streaming `x` through the kernel without blowing out memory or thrashing PCIe.

## Problem

- `Sparse1DProbe` assumes the feature design matrix arrives as CSR and fits in device memory, which stops working when activations are nearly dense or the GPU only has 16 GiB.
- Dense runs would construct a `(n_samples, n_latents)` tensor on the GPU per LM iteration, so per-iteration transfers multiply the footprint by the number of iterations.
- The solver mixes CPU and GPU tensors in `_plan_row_chunks`, `_iter_event_chunks`, and `_compute_slab_stats`, making it hard to share logic between sparse and dense paths without first isolating data movement.
- We need deterministic streaming so the LM loop, loss, and confusion helpers all see data in the same order, otherwise results diverge between sparse and dense backends.

## Goals and constraints

- Keep `y` and probe state (intercepts, coefficients, accumulators) on the GPU.
- Stream `x` from host to device in batches sized by available VRAM; avoid staging the full matrix.
- Reuse one abstraction across fit, loss, and confusion computation so results remain bit-identical regardless of storage layout.
- Maintain compatibility with current sparse CSR workloads and preserve their performance.
- Allow offline CPU-only fallbacks for testing when no GPU is present.

## Proposed design

### Interface

- Introduce a `DesignAccessor` protocol that encapsulates layout-specific logic:
  - `compute_latent_stats()` returns `nnz_per_latent` and `sum_sq` without materializing the whole matrix in VRAM.
  - `stream_rows(row_batch_size)` yields deterministic batches containing row indices and a CPU view of the features.
  - `accumulate_stats(...)` writes the LM statistics directly into provided tensors so the caller controls dtype and device.
- Provide two implementations:
  - `CSRDesignAccessor` reuses the existing `_plan_row_chunks` and event iteration logic.
  - `DenseRowMajorAccessor` treats `x` as a dense row-major matrix living on CPU RAM.

### Dense streaming

- Hold the dense activations in pinned host memory. When activations are loaded from disk they land on CPU; signal the accessor to pin once before training.
- Let `DenseRowMajorAccessor.stream_rows` return `DenseRowBatch` objects:
  - `row_start`, `row_end`, and `cpu_view` shaped `(batch, n_latents)`.
  - A batch size heuristic chooses the largest multiple of the warp size that fits in `free_vram * occupancy_factor`. Query `torch.cuda.mem_get_info` at runtime and reserve headroom for slabs and LM buffers.
- `Sparse1DProbe` allocates two device staging buffers (double buffering) sized `(row_batch_size, latent_block_size)` plus two CUDA streams so host-to-device copies overlap with compute.
- The LM loop becomes:

```py
for row_batch in design.stream_rows(batch):
    x_cpu = row_batch.cpu_view
    x_gpu = staging[current].copy_(x_cpu, non_blocking=True)
    y_gpu = y[row_batch.row_start:row_batch.row_end]
    for latent_block in latent_blocks(n_latents, latent_block_size):
        stats = design.accumulate_stats_dense(
            x_gpu[:, latent_block],
            y_gpu,
            intercept[latent_block],
            coef[latent_block],
            slab_stats[latent_block],
        )
    swap(current)
```

- `latent_block_size` keeps register usage under control by limiting the number of latents processed per GEMM-like update, enabling batched matrix-vector operations instead of scalar loops.
- Because dense rows contribute to every latent, `nnz_per_latent = n_samples` and `n_zeros_per_latent = 0`. The accessor precomputes `sum_sq` with a CPU streaming pass so `qx` stays consistent with CSR runs.

### Integration

- Wrap `x` immediately at the top of `Sparse1DProbe.fit` using `make_design_accessor(x, device)` and keep the abstraction for `loss_matrix` and `confusion`.
- Move `_plan_row_chunks` and `_iter_event_chunks` into `CSRDesignAccessor`. Their tensors stay on CPU and only transfer when the iterator asks for them.
- Update `SparseEventsBatch` consumers to operate on an accessor-supplied iterator, so the dense path can share the same LM math without branching inside the hot loop.
- Provide regression tests covering:
  - A sparse run equal to current behavior.
  - A dense `(2_000, 4_096)` synthetic dataset that does not fit on a 16 GiB GPU unless streamed, compared against the reference solver for correctness.
  - A CPU-only dense run to guarantee no accidental CUDA-only code paths.

## Expected impact

- Dense probes stop failing due to VRAM limits and no longer restart because of host/device mismatches.
- Sparse workloads pay only a small constant factor for the accessor indirection while gaining cleaner data movement.
- The probe can tolerate larger class slabs or higher iteration counts because the largest allocation scales with the streaming batch rather than the full dataset.

## Open questions

- What is the best heuristic for choosing `row_batch_size` and `latent_block_size`? We may need to expose configuration knobs or profile-driven tuning.
- Do we need a third accessor for hybrid (block sparse) layouts, or can CSR remain the default sparse path?
- Are there other consumers of `Sparse1DProbe` that expect direct tensor access to `x` and would need shims?

## Next steps

- Draft the `DesignAccessor` protocol and migrate the existing CSR logic into the new structure.
- Prototype the dense streaming path using synthetic data to validate numerical parity.
- Profile the double-buffered loop to tune batch heuristics before changing production jobs.
