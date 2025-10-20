# Probe1D performance

## Context

- Latest Slurm job 2669594 shows each LM iteration for the first class slab (size 8) taking about 35 seconds.
- With 151 classes we run roughly 19 slabs; at 100 iterations this is about 18 hours if nothing speeds up.
- The solver now avoids out of memory errors, but many coordinates still enter the Levenberg–Marquardt fallback loop despite having zero gradients.

## Profiling plan

- Keep using pyinstrument on reduced data sets (for example 1k rows and 256 classes) to profile Sparse1DProbe.fit and loss_matrix_with_aux.
- Set PYTORCH_NO_CUDA_MEMORY_CACHING to 1 during profiling to make GPU allocations obvious.
- Collect one profile with iteration_hook disabled and one with debug logging enabled to understand logging overhead.

## Profiling results (contrib/trait_discovery/profiles/it9.html)

- compute_slab_stats accounts for 99.9 percent of Sparse1DProbe.fit runtime.
- _iter_event_chunks accounts for 64.2 percent of fit.
  - torch.repeat_interleave alone consumes 13.5 percent.
  - Tensor.item calls consume 9.3 percent.
  - Tensor.to inside the loop consumes 2.6 percent.
  - torch.arange consumes 1.9 percent.
- The remaining time is largely absorbed by binary cross entropy and index_add kernels that are launched per chunk.

## Updated priorities

- Rework _iter_event_chunks so it no longer constructs row_idx_chunk with repeat_interleave every pass. Precompute row indices or reuse cached tensors sliced by offsets.
- Eliminate Tensor.item inside the hot loop by operating on whole tensors of crow pointers.
- Hoist Tensor.to out of the per chunk loop by caching values and offsets in compute_dtype once.
- Skip zero-gradient coordinates before the Levenberg–Marquardt step so empty rows do not retry indefinitely.
- Consider fusing consecutive index_add calls or batching updates to reduce kernel launch overhead.

## Open questions

- Can we derive qx from statistics that reuse existing sums (for example running RMS) without extra passes?
- Would a mixed precision strategy (weights float32, accumulators float64) reduce the numerical issues that trigger fallbacks?

## Next steps

- Prototype a cached row index builder shared by compute_slab_stats, _compute_loss_slab, and _compute_confusion_slab so repeat_interleave disappears from the hot path.
- Move all per chunk casts and arange construction out of _iter_event_chunks and re-profile.
- Add a gradient tolerance mask so zero-gradient coordinates exit the LM loop immediately.
- Re-run pyinstrument after these changes and update this issue with the new timing breakdown.
