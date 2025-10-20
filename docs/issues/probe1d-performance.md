# Probe1D Performance

## Context

- Latest Slurm job 2669594 shows each LM iteration for the first class_slab_size = 8 block taking about 35 s.
- With 151 classes we run roughly 19 slabs; at max_iter = 100 that is 19 x 100 x 35 s ≈ 18 hours if nothing speeds up.
- Recent changes prevent OOMs but almost every iteration still loops with n_fail ≈ 935 LM fallbacks where gradients are already zero.

## Profiling plan

- Use pyinstrument's Python API (https://pyinstrument.readthedocs.io/en/latest/guide.html#profile-a-specific-chunk-of-code) to capture Sparse1DProbe.fit (smaller shard such as 1k rows, 256 classes) on a real subset of data.
- Set PYTORCH_NO_CUDA_MEMORY_CACHING=1 during profiling to make GPU allocations obvious in the timeline.
- Collect one profile with `iteration_hook` disabled and one with DEBUG logging to understand logging overhead.

## Suspected bottlenecks before profiling

- _compute_slab_stats and _compute_loss_slab rebuild sparse row indices for every slab via _iter_event_chunks; the chunker allocates new row_idx_chunk tensors each pass.
- The LM step treats zero-gradient coordinates as failures, so the damping loop runs repeatedly and keeps the lambda schedule saturated.
- The code converts the full CSR value tensor to self.compute_dtype on every pass, creating redundant copies.
- Debug logging forces torch.cuda.synchronize, which may stall kernels even when no debug output is emitted.

## Ideas to try

- Skip zero-gradient coordinates before calling _compute_lm_step so inert latents exit early instead of triggering fallbacks.
- Cache row indices per slab and reuse them across _compute_slab_stats, _compute_loss_slab, and _compute_confusion_slab.
- Avoid redundant casts by storing CSR values in self.compute_dtype once after the initial x.to(...).
- Tune class_slab_size and row_batch_size to amortize traversal overhead if memory allows.
- Stop increasing lambda for coordinates that repeatedly fail; mark them converged after a small number of retries.
- Add optional instrumentation hooks that record per-slab timing without the overhead of full profiling.

## Open questions

- Is there a cheaper statistic than RMS for qx that preserves stability but avoids extra passes over the data?
- Would mixed precision (weights float32, accumulators float64) remove the residual numerical issues that trigger LM fallback?

## Next steps

- Implement the zero-gradient short-circuit.
- Profile fit with pyinstrument on a reduced dataset and annotate hotspots.
- Measure slab iteration time again after applying fixes and update this issue with real numbers.
