**Title**: Probe1D end-to-end runtime is uncomfortably long

**Context**
- Latest Slurm job `2669594` shows each LM iteration for the first `class_slab_size=8` block taking ~35 s.
- With 151 classes we run ~19 slabs; at `max_iter=100` that’s 19 x 100 x 35 s ≈ 18 hours if nothing speeds up.
- Recent changes prevent OOMs but we now spend most of the time looping with `n_fail ≈ 935` LM fallbacks where gradients are already zero.

**Profiling Plan**
- Use `pyinstrument` (`uv run pyinstrument contrib/trait_discovery/src/tdiscovery/probe1d.py --args ...`) to capture:
  1. `Sparse1DProbe.fit` on a smaller shard (e.g. 1k rows, 256 classes) to identify hot paths.
  2. `loss_matrix_with_aux` because it replays the sparse traversal post-fit.
- Enable `PYTORCH_NO_CUDA_MEMORY_CACHING=1` during profiling to make GPU allocations obvious in the timeline.
- Capture one profile with `iteration_hook` disabled and one with `DEBUG` logging to validate logging overhead.

**Suspected Bottlenecks (pre-profiling)**
- `_compute_slab_stats` and `_compute_loss_slab` repeatedly recompute sparse row indices for every slab via `_iter_event_chunks`; the current chunker builds fresh `row_idx_chunk` tensors on every pass.
- LM step handles coordinates with zero gradients as “failures,” re-entering the damping loop and logging warnings; this wastes compute and keeps lambda huge.
- Converting entire CSR value arrays to `self.compute_dtype` (float32) every iteration (`values_all = x.values().to(...)`) introduces redundant copies.
- Frequent device synchronizations (`torch.cuda.synchronize` inside debug block) may stall kernels even when DEBUG logging is disabled.

**Obvious Fixes / Mitigations**
1. **Skip zero-gradient coordinates early**: mask `grad_max < tol` before calling `_compute_lm_step` so inactive latents exit immediately, cutting the 935 fallbacks.
2. **Cache row indices**: precompute `row_idx_chunk` once per slab and reuse across `_compute_slab_stats`, `_compute_loss_slab`, and `_compute_confusion_slab` to avoid rebuilding identical tensors.
3. **Avoid redundant casts**: store CSR values in `self.compute_dtype` once after `x = x.to(...)` and reuse the tensor instead of calling `.to()` in every loop.
4. **Tune slab/batch sizes**: try larger `class_slab_size` or `row_batch_size` to amortize sparse traversal overhead if GPU memory allows.
5. **Short-circuit LM fallback**: if failures persist, mark them converged after a few retries and stop increasing lambda; this reduces the number of iterations wasted on hopeless coordinates.
6. **Add instrumentation hooks**: expose timers per slab (`profiling_hook`) so we can track progress without full profiles.

**Open Questions**
- Is there a cheaper statistic than RMS that keeps stability but avoids extra passes over the sparsity structure?
- Would mixed precision (weights in float32, accumulators in float64) remove the residual numerical issues causing LM fallback?

**Next Steps**
- Land the zero-gradient short-circuit.
- Profile `fit` with `pyinstrument` on a reduced dataset; annotate hotspots.
- Re-evaluate slab iteration time after the fixes to update this issue with real numbers.
