import math
from typing import Any, Iterator, Protocol, runtime_checkable

import beartype


@beartype.beartype
class Scheduler:
    def step(self) -> float:
        err_msg = f"{self.__class__.__name__} must implement step()."
        raise NotImplementedError(err_msg)

    def __repr__(self) -> str:
        err_msg = f"{self.__class__.__name__} must implement __repr__()."
        raise NotImplementedError(err_msg)


@beartype.beartype
class Warmup(Scheduler):
    """
    Linearly increases from `init` to `final` over `n_warmup_steps` steps.
    """

    def __init__(self, init: float, final: float, n_steps: int):
        self.final = final
        self.init = init
        self.n_steps = n_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        if self._step < self.n_steps:
            return self.init + (self.final - self.init) * (self._step / self.n_steps)

        return self.final

    def __repr__(self) -> str:
        return f"Warmup(init={self.init}, final={self.final}, n_steps={self.n_steps})"


@beartype.beartype
class WarmupCosine(Scheduler):
    """
    Linearly increases from `init` to `peak` over `n_warmup` steps, then decrease down to final using cosine decay over n_steps - n_warmup.
    """

    def __init__(
        self, init: float, n_warmup: int, peak: float, n_steps: int, final: float
    ):
        self.init = init
        self.peak = peak
        self.final = final
        self.n_warmup = n_warmup
        self.n_steps = n_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        if self._step < self.n_warmup:
            return self.init + (self.peak - self.init) * (self._step / self.n_warmup)
        elif self._step < self.n_steps:
            # Cosine decay from self.peak to self.final over (n_steps - n_warmup)
            progress = (self._step - self.n_warmup) / (self.n_steps - self.n_warmup)
            cosine_factor = (1 + math.cos(math.pi * progress)) / 2
            return self.final + (self.peak - self.final) * cosine_factor

        return self.final

    def __repr__(self) -> str:
        return f"WarmupCosine(init={self.init}, peak={self.peak}, final={self.final}, n_warmup={self.n_warmup}, n_steps={self.n_steps})"


@runtime_checkable
class DataLoaderLike(Protocol):
    drop_last: bool
    batch_size: int  # This is also needed since BatchLimiter uses it

    def __iter__(self) -> Iterator[Any]: ...


@beartype.beartype
class BatchLimiter:
    """
    Limits the number of batches to only return `n_samples` total samples.
    """

    def __init__(self, dataloader: DataLoaderLike, n_samples: int):
        self.dataloader = dataloader
        self.n_samples = n_samples
        self.batch_size = dataloader.batch_size

    def __len__(self) -> int:
        return self.n_samples // self.batch_size

    def __getattr__(self, name: str) -> Any:
        """Pass through attribute access to the wrapped dataloader."""
        # __getattr__ is only called when the attribute wasn't found on self
        # So we delegate to the wrapped dataloader
        try:
            return getattr(self.dataloader, name)
        except AttributeError:
            # Re-raise with more context about where the attribute was not found
            raise AttributeError(
                f"'{self.__class__.__name__}' object and its wrapped dataloader have no attribute '{name}'"
            )

    def __iter__(self):
        self.n_seen = 0
        while self.n_seen < self.n_samples:
            for batch in self.dataloader:
                # Count the actual batch size, not the configured batch_size.
                # The actual batch may be smaller (e.g., last batch when drop_last=False).
                actual_batch_size = self._get_batch_size(batch)

                # Check BEFORE yielding to avoid going over the limit
                if self.n_seen + actual_batch_size > self.n_samples:
                    return

                yield batch
                self.n_seen += actual_batch_size

    def _get_batch_size(self, batch: Any) -> int:
        """Determine the actual size of a batch from various data structures."""
        # Handle dict-like batches (common in custom dataloaders)
        if isinstance(batch, dict):
            for key in ["act", "image", "input", "data"]:
                if key in batch:
                    return len(batch[key])
            # Fallback: try first value
            first_val = next(iter(batch.values()))
            if hasattr(first_val, "__len__"):
                return len(first_val)

        # Handle tuple/list batches from PyTorch DataLoader
        # PyTorch returns batches as list/tuple of tensors
        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            first_item = batch[0]
            # Check if first item is a tensor/array with length
            if hasattr(first_item, "__len__"):
                return len(first_item)
            # If it's a scalar, the list itself is the batch
            return len(batch)

        # Handle direct tensor batches
        if hasattr(batch, "__len__"):
            return len(batch)

        # Fallback to configured batch_size if we can't determine
        return self.batch_size


def _plot_example_schedules():
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()

    n_steps = 1000
    xs = np.arange(n_steps)

    schedule = WarmupCosine(0.1, 100, 0.9, 1000, 0.0)
    ys = [schedule.step() for _ in xs]

    ax.plot(xs, ys, label=str(schedule))

    fig.tight_layout()
    fig.savefig("schedules.png")


if __name__ == "__main__":
    _plot_example_schedules()
