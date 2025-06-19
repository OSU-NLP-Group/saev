"""
Useful helpers for `saev`.
"""

import collections.abc
import logging
import os
import time

import beartype


@beartype.beartype
def get_cache_dir() -> str:
    """
    Get cache directory from environment variables, defaulting to the current working directory (.)

    Returns:
        A path to a cache directory (might not exist yet).
    """
    cache_dir = ""
    for var in ("SAEV_CACHE", "HF_HOME", "HF_HUB_CACHE"):
        cache_dir = cache_dir or os.environ.get(var, "")
    return cache_dir or "."


@beartype.beartype
class progress:
    def __init__(self, it, *, every: int = 10, desc: str = "progress", total: int = 0):
        """
        Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.

        Args:
            it: Iterable to wrap.
            every: How many iterations between logging progress.
            desc: What to name the logger.
            total: If non-zero, how long the iterable is.
        """
        self.it = it
        self.every = every
        self.logger = logging.getLogger(desc)
        self.total = total

    def __iter__(self):
        start = time.time()

        try:
            total = len(self)
        except TypeError:
            total = None

        for i, obj in enumerate(self.it):
            yield obj

            if (i + 1) % self.every == 0:
                now = time.time()
                duration_s = now - start
                per_min = (i + 1) / (duration_s / 60)

                if total is not None:
                    pred_min = (total - (i + 1)) / per_min
                    self.logger.info(
                        "%d/%d (%.1f%%) | %.1f it/m (expected finish in %.1fm)",
                        i + 1,
                        total,
                        (i + 1) / total * 100,
                        per_min,
                        pred_min,
                    )
                else:
                    self.logger.info("%d/? | %.1f it/m", i + 1, per_min)

    def __len__(self) -> int:
        if self.total > 0:
            return self.total

        # Will throw exception.
        return len(self.it)


###################
# FLATTENED DICTS #
###################


@beartype.beartype
def flattened(
    dct: dict[str, object], *, sep: str = "."
) -> dict[str, str | int | float | bool | None]:
    """
    Flatten a potentially nested dict to a single-level dict with `.`-separated keys.
    """
    new = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            for nested_key, nested_value in flattened(value).items():
                new[key + "." + nested_key] = nested_value
            continue

        new[key] = value

    return new


@beartype.beartype
def get(dct: dict[str, object], key: str, *, sep: str = ".") -> object:
    key = key.split(sep)
    key = list(reversed(key))

    while len(key) > 1:
        popped = key.pop()
        dct = dct[popped]

    return dct[key.pop()]


@beartype.beartype
class batched_idx:
    """
    Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.

    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.

    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.
    """

    def __init__(self, total_size: int, batch_size: int):
        """
        Args:
            total_size: total number of examples
            batch_size: maximum distance between the generated indices
        """
        self.total_size = total_size
        self.batch_size = batch_size

    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]:
        """Yield (start, end) index pairs for batching."""
        for start in range(0, self.total_size, self.batch_size):
            stop = min(start + self.batch_size, self.total_size)
            yield start, stop

    def __len__(self) -> int:
        """Return the number of batches."""
        return (self.total_size + self.batch_size - 1) // self.batch_size
