# Tracking Batch Entropy

We need a metric in the shuffled data loader that reports the batch entropy of the example index and of the patch index so we can understand how well the shuffling covers each batch.
Please add metrics that compute these two entropies per batch and aggregate them for monitoring during training runs.

## Context

- The shuffled data loader returns batches with activations and associated `example_idx` and `token_idx`, sourced from a shared-memory reservoir that randomizes sample order before yielding to the main process.
- Training wraps `ShuffledDataLoader` with `BatchLimiter`, which forwards attribute access, so loader-side metric state must live on the underlying loader to remain accessible for logging.
- Loader metrics currently logged during training focus on throughput (`loader/read_mb`, `loader/read_mb_s`, `loader/cpu_util`, `loader/buffer_fill`) and do not summarize sampling quality.
- Metadata exposes `n_examples` and `content_tokens_per_example`, which establish the theoretical support for example and patch indices needed to normalize entropy measurements.

## Plan

1. Add a standalone `calc_batch_entropy` helper (likely in `src/saev/metrics.py` or similar) that takes index tensors plus support sizes and returns raw and normalized entropies for example and token indices; keep it torch-or numpy-compatible for easy testing.
2. Update the training loop in `train.py` to call this helper for each batch when metrics are gathered, deriving support sizes from `dataloader.metadata` and `metadata.content_tokens_per_example`.
3. Merge the resulting batch-local stats into `loader_metrics`, logging values like `loader/example_entropy` and `loader/token_entropy` alongside their normalized counterparts.
4. Add targeted unit tests for the helper (new test module under `tests/`) that feed synthetic distributions and assert the entropy calculations, including edge cases like repeated indices or empty bins.
5. Document the per-batch entropy logging flow in the shuffled dataloader README or training docs so users understand that the metrics are computed in `train.py` and how normalization is defined.

## Questions

- Q: Should entropy be reported in natural logarithm units or base-2 bits for better interpretability alongside other metrics?
  - A: Natural log units is fine.
- Q: Do we also want to surface coverage ratios (unique indices over support) alongside entropy for quicker diagnostics, or is normalized entropy sufficient?
  - A: Include coverage ratios and normalized entropy.
