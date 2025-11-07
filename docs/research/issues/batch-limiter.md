# Context

`ShuffledDataLoader` waits for its shared-memory reservoir to reach `cfg.min_buffer_fill` before it yields any samples. The code currently measures fill as `reservoir.fill()` which normalizes by the ring buffer capacity (`buffer_size * batch_size`). That works when we have plenty of activations, but it breaks down for tiny datasets and for iterators that deliberately cap the number of consumed samples via `BatchLimiter`.

# Reproduction

- Create two fake images (`datasets.FakeImg(n_examples=2)`) with 16 content tokens each.
- Configure `ShuffledDataLoader` with `batch_size=16`, the default `buffer_size=64`, and `min_buffer_fill=0.2`.
- Either iterate over the loader directly (`test_min_buffer_fill_handles_small_dataset`) or wrap it in `BatchLimiter` (`test_min_buffer_fill_with_batch_limiter`).
- The manager threads read every activation (32 total) and then exit. Because the reservoir capacity is `buffer_size * batch_size = 1024`, the measured fill ratio is 32 / 1024 = 0.031, which never reaches the requested 0.2 threshold. `_wait_for_min_buffer_fill` spins until it notices the manager died and raises `RuntimeError("Manager process died while waiting for reservoir fill.")`.

# Diagnosis

`cfg.min_buffer_fill` is intended to express "require X% of the available activations to be in the buffer before yielding batches." Instead, we compare against the nominal reservoir capacity, which is unrelated to how many samples the manager can possibly enqueue:

- When the dataset is smaller than the reservoir, we physically cannot hit the requested fraction even though every activation is buffered.
- When a wrapper wants fewer samples than the dataset (e.g., `BatchLimiter(..., 128)`), we still insist on filling the entire reservoir before yielding anything, even though we will promptly drop most of those batches once the limiter stops.
- The failure mode is brutal: iteration never yields a batch, the manager exits successfully, and the main process treats this as a crash.

# Proposed Fix

Treat the warmup threshold as a requirement on *available* samples instead of raw buffer slots.

1. Compute an "effective capacity" each time we wait for warmup: `effective_capacity = min(reservoir.capacity, self.n_samples)`. This is the most data the manager can ever place in the buffer before the current epoch ends.
2. Convert `cfg.min_buffer_fill` to an absolute count with `target = math.ceil(cfg.min_buffer_fill * effective_capacity)` and wait until `reservoir.qsize()` reaches that count.
3. Record `_last_reservoir_fill` as `reservoir.qsize() / effective_capacity` so diagnostics remain meaningful.
4. If `effective_capacity == 0` (empty dataset), skip the wait entirely.
5. Add a debug log when the raw reservoir capacity exceeds `self.n_samples` so we can spot cases where we are over-buffering.

With this change the loader acknowledges that a tiny dataset can still satisfy a 20% warmup by buffering every available example, and the limiter can no longer strand the manager just by capping the consumer's appetite.
