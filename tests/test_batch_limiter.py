"""
Tests for BatchLimiter to demonstrate the sample counting bug.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from saev.utils.scheduling import BatchLimiter


class SimpleMockDataLoader:
    """A minimal mock dataloader that satisfies the DataLoaderLike protocol."""

    def __init__(self, data, batch_size, drop_last=False):
        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i : i + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch


def test_batch_limiter_with_uneven_batches():
    """
    Test that BatchLimiter correctly counts samples when batches have uneven sizes.

    This test demonstrates the bug: BatchLimiter assumes all batches have size
    `batch_size`, but the last batch may be smaller when `drop_last=False`.
    """
    # Create a dataset with 105 samples
    data = list(range(105))

    # Create a dataloader with batch_size=32, drop_last=False
    # This will produce batches: [32, 32, 32, 9]
    dataloader = SimpleMockDataLoader(data, batch_size=32, drop_last=False)

    # Request exactly 100 samples via BatchLimiter
    limiter = BatchLimiter(dataloader, n_samples=100)

    # Collect all yielded samples
    seen_samples = []
    for batch in limiter:
        seen_samples.extend(batch)

    # BUG: We expect to see ~100 samples, but due to the counting bug,
    # the limiter terminates early.
    #
    # What happens:
    # - Batch 1: 32 samples yielded, counter = 32
    # - Batch 2: 32 samples yielded, counter = 64
    # - Batch 3: 32 samples yielded, counter = 96
    # - Batch 4: 9 samples yielded, counter = 128 > 100, so return
    #
    # Total yielded: 105 samples (more than requested!)
    # But if we had exactly 96 samples total and wanted 100, we'd only get 96.

    # The correct behavior would be to track actual samples seen
    # and stop when we hit the limit, not when we *think* we've hit it.
    print("Requested: 100 samples")
    print(f"Actually seen: {len(seen_samples)} samples")

    # This assertion will FAIL, demonstrating the bug
    # The limiter yields all 105 samples because the counter check happens
    # AFTER yielding, and the last batch pushes the counter over the limit
    assert len(seen_samples) <= 100, (
        f"Expected at most 100 samples, but got {len(seen_samples)}"
    )


def test_batch_limiter_early_termination():
    """
    Test that BatchLimiter correctly stops when requested samples exceed available data.

    When the dataloader has fewer samples than requested, it should loop and provide
    exactly the requested number (or as close as possible without exceeding).
    """
    # Create a dataset with exactly 96 samples
    data = list(range(96))

    # Create a dataloader with batch_size=32, drop_last=False
    # This will produce exactly 3 batches: [32, 32, 32]
    dataloader = SimpleMockDataLoader(data, batch_size=32, drop_last=False)

    # Request 100 samples (more than available in one pass)
    limiter = BatchLimiter(dataloader, n_samples=100)

    # Collect all yielded samples
    seen_samples = []
    iterations = 0
    for batch in limiter:
        seen_samples.extend(batch)
        iterations += 1

    # With the fix:
    # - Batch 1: 32 samples, counter = 32
    # - Batch 2: 32 samples, counter = 64
    # - Batch 3: 32 samples, counter = 96
    # - Dataloader restarts (while loop continues)
    # - Check: 96 + 32 = 128 > 100, so return
    #
    # Result: 96 samples (correct - stops before exceeding 100)

    print("Requested: 100 samples")
    print(f"Actually seen: {len(seen_samples)} samples")
    print(f"Number of batches: {iterations}")

    # Should get 96 samples (stops before the next batch would exceed 100)
    assert len(seen_samples) == 96, f"Expected 96 samples, but got {len(seen_samples)}"


def test_batch_limiter_with_pytorch_dataloader():
    """
    Test BatchLimiter with a real PyTorch DataLoader to ensure the bug
    manifests with actual production code.
    """
    # Create a dataset with 105 samples
    dataset = TensorDataset(torch.arange(105))

    # Create a PyTorch DataLoader
    dataloader = DataLoader(dataset, batch_size=32, drop_last=False)

    # Request 100 samples
    limiter = BatchLimiter(dataloader, n_samples=100)

    # Collect all samples
    seen_samples = []
    for batch in limiter:
        # batch is a tuple from TensorDataset
        seen_samples.extend(batch[0].tolist())

    print("Requested: 100 samples")
    print(f"Actually seen: {len(seen_samples)} samples")

    # This assertion FAILS
    assert len(seen_samples) <= 100, (
        f"Expected at most 100 samples, but got {len(seen_samples)}"
    )


def test_batch_limiter_exact_multiple():
    """
    Test when n_samples is an exact multiple of batch_size.
    This case should work correctly even with the bug.
    """
    data = list(range(128))
    dataloader = SimpleMockDataLoader(data, batch_size=32, drop_last=False)

    # Request exactly 96 samples (3 * 32)
    limiter = BatchLimiter(dataloader, n_samples=96)

    seen_samples = []
    for batch in limiter:
        seen_samples.extend(batch)

    # This should work because:
    # - Batch 1: 32 samples, counter = 32
    # - Batch 2: 32 samples, counter = 64
    # - Batch 3: 32 samples, counter = 96
    # - Check: 96 <= 96 is False (should be >), so it continues
    # - Batch 4: 32 samples, counter = 128 > 96, return
    #
    # Total: 128 samples (still wrong!)

    print("Requested: 96 samples")
    print(f"Actually seen: {len(seen_samples)} samples")

    # Even this case FAILS
    assert len(seen_samples) == 96, (
        f"Expected exactly 96 samples, but got {len(seen_samples)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
