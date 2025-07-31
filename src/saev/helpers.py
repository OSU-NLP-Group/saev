# src/saev/helpers.py
import collections.abc
import dataclasses
import itertools
import logging
import os
import pathlib
import subprocess
import time
import typing

import beartype


@beartype.beartype
class RemovedFeatureError(RuntimeError):
    """Feature existed before but is no longer supported."""

    pass


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


#################
# SWEEP HELPERS #
#################


T = typing.TypeVar("T")


@beartype.beartype
def expand(config: dict[str, object]) -> collections.abc.Iterator[dict[str, object]]:
    """Expand a nested dict that may contain lists into many dicts."""

    yield from _expand_discrete(dict(config))


@beartype.beartype
def _expand_discrete(
    config: dict[str, object],
) -> collections.abc.Iterator[dict[str, object]]:
    if not config:
        yield {}
        return

    key, value = config.popitem()

    if isinstance(value, list):
        for c in _expand_discrete(config):
            for v in value:
                yield {**c, key: v}
    elif isinstance(value, dict):
        for c, v in itertools.product(
            _expand_discrete(config), _expand_discrete(value)
        ):
            yield {**c, key: v}
    else:
        for c in _expand_discrete(config):
            yield {**c, key: value}


@beartype.beartype
def grid(cfg: T, sweep_dct: dict[str, object]) -> tuple[list[T], list[str]]:
    """Generate configs from ``cfg`` according to ``sweep_dct``."""

    cfgs: list[T] = []
    errs: list[str] = []

    for d, dct in enumerate(expand(sweep_dct)):
        updates = {}
        for key, value in dct.items():
            attr = getattr(cfg, key)
            if dataclasses.is_dataclass(attr) and isinstance(value, dict):
                sub_updates = dict(value)
                if hasattr(attr, "seed"):
                    sub_updates["seed"] = (
                        sub_updates.get("seed", attr.seed) + getattr(cfg, "seed", 0) + d
                    )
                updates[key] = dataclasses.replace(attr, **sub_updates)
            else:
                updates[key] = value

        if hasattr(cfg, "seed"):
            updates.setdefault("seed", getattr(cfg, "seed") + d)

        try:
            cfgs.append(dataclasses.replace(cfg, **updates))
        except Exception as err:  # pragma: no cover - passthrough for caller
            errs.append(str(err))

    return cfgs, errs


@beartype.beartype
def current_git_commit() -> str | None:
    """
    Best-effort short SHA of the repo containing *this* file.

    Returns `None` when
    * `git` executable is missing,
    * we’re not inside a git repo (e.g. installed wheel),
    * or any git call errors out.
    """
    try:
        # Walk up until we either hit a .git dir or the FS root
        here = pathlib.Path(__file__).resolve()
        for parent in (here, *here.parents):
            if (parent / ".git").exists():
                break
        else:  # no .git found
            return None

        result = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
