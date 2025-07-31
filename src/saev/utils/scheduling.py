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
        while True:
            for batch in self.dataloader:
                yield batch

                # Sometimes we underestimate because the final batch in the dataloader might not be a full batch.
                self.n_seen += self.batch_size
                if self.n_seen > self.n_samples:
                    return

            # We try to mitigate the above issue by ignoring the last batch if we don't have drop_last.
            if not self.dataloader.drop_last:
                self.n_seen -= self.batch_size


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
