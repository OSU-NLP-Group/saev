# Bug Report: BatchLimiter Incorrectly Counts Samples

## Description

The `BatchLimiter` class in `src/saev/utils/scheduling.py` incorrectly counts the number of samples seen during iteration, leading to premature termination when the actual batch size is smaller than the expected batch size.

## Location

`src/saev/utils/scheduling.py:114`

## Root Cause

In the `__iter__` method, the code always increments `self.n_seen` by `self.batch_size`:

```python
self.n_seen += self.batch_size
if self.n_seen > self.n_samples:
    return
```

However, the actual batch yielded might have fewer samples than `self.batch_size`, particularly:
1. For the last batch when `drop_last=False`
2. For dataloaders with uneven dataset sizes

This causes the limiter to overcount samples, terminating the iterator before yielding all requested samples.

## Expected Behavior

The `BatchLimiter` should count the actual number of samples in each batch, not assume all batches have size `self.batch_size`.

## Actual Behavior

The limiter terminates early when it *thinks* it has seen `n_samples`, even though it may have yielded fewer samples than requested.

## Example

If we have:
- A dataloader with 105 samples
- `batch_size = 32`
- `drop_last = False`
- `n_samples = 100` (what we want from BatchLimiter)

The batches would be: [32, 32, 32, 9]

But the counter would be: [32, 64, 96, 128]

When the counter hits 128 > 100, it returns, but we've only yielded 105 samples total, not the expected ~100. Even worse, if we had exactly 100 samples and wanted all 100, we'd get: [32, 64, 96] = 96 samples counted, then the next batch would push us to 128, causing early termination after only 96 samples.

## Proposed Fix

Change line 114 to count the actual batch size:

```python
self.n_seen += len(batch["image"]) if isinstance(batch, dict) else len(batch)
```

Or more generally, determine the actual batch size from the yielded data structure.

## Test Case

See `tests/test_batch_limiter.py` for a unit test that demonstrates this bug.
