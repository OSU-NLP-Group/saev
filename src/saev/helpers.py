# src/saev/helpers.py
import dataclasses
import logging
import math
import os
import pathlib
import re
import subprocess
import time
import typing as tp
from collections.abc import Hashable, Iterable, Iterator

import beartype
import numpy as np
import orjson


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
def fssafe(s: str) -> str:
    """
    Convert a string to be filesystem-safe by replacing special characters.

    This is particularly useful for checkpoint names that contain characters like
    'hf-hub:timm/ViT-L-16-SigLIP2-256' which need to be converted to something like
    'hf-hub_timm_ViT-L-16-SigLIP2-256'.

    Args:
        s: String to make filesystem-safe.

    Returns:
        Filesystem-safe version of the string.
    """
    # Replace common problematic characters with underscores
    replacements = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
        " ": "_",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    # Remove any remaining non-alphanumeric characters except - _ .
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


@beartype.beartype
class progress:
    """
    Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.

    Args:
        it: Iterable to wrap.
        every: How many iterations between logging progress.
        desc: What to name the logger.
        total: If non-zero, how long the iterable is.
    """

    def __init__(
        self, it: Iterable, *, every: int = 10, desc: str = "progress", total: int = 0
    ):
        self.it = it
        self.every = max(every, 1)
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
        self.total_size = total_size
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """Yield (start, end) index pairs for batching."""
        for start in range(0, self.total_size, self.batch_size):
            stop = min(start + self.batch_size, self.total_size)
            yield start, stop

    def __len__(self) -> int:
        """Return the number of batches."""
        return (self.total_size + self.batch_size - 1) // self.batch_size


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


@beartype.beartype
def get_slurm_max_array_size() -> int:
    """
    Get the MaxArraySize configuration from the current Slurm cluster.

    Returns:
        int: The maximum array size allowed on the cluster. Returns 1000 as fallback if unable to determine.
    """
    logger = logging.getLogger("helpers.slurm")
    try:
        # Run scontrol command to get config information
        result = subprocess.run(
            ["scontrol", "show", "config"], capture_output=True, text=True, check=True
        )

        # Search for MaxArraySize in the output
        match = re.search(r"MaxArraySize\s*=\s*(\d+)", result.stdout)
        if match:
            max_array_size = int(match.group(1))
            logger.info("Detected MaxArraySize = %d", max_array_size)
            return max_array_size
        else:
            logger.warning(
                "Could not find MaxArraySize in scontrol output, using default of 1000"
            )
            return 1000

    except subprocess.SubprocessError as e:
        logger.error("Error running scontrol: %s", e)
        return 1000  # Safe default
    except ValueError as e:
        logger.error("Error parsing MaxArraySize: %s", e)
        return 1000  # Safe default
    except FileNotFoundError:
        logger.warning(
            "scontrol command not found. Assuming not in Slurm environment. Returning default MaxArraySize=1000."
        )
        return 1000


@beartype.beartype
def get_slurm_max_submit_jobs() -> int:
    """
    Get the MaxSubmitJobs limit from the current user's QOS.

    Returns:
        int: The maximum number of jobs that can be submitted at once. Returns 1000 as fallback.
    """
    logger = logging.getLogger("helpers.slurm")
    try:
        # First, try to get the QOS from a recent job
        result = subprocess.run(
            ["scontrol", "show", "job", "-o"],
            capture_output=True,
            text=True,
            check=False,
        )

        qos_name = None
        if result.returncode == 0 and result.stdout:
            # Extract QOS from job info
            match = re.search(r"QOS=(\S+)", result.stdout)
            if match:
                qos_name = match.group(1)

        if not qos_name:
            # If no jobs, try to get default QOS from association
            # This is less reliable but better than nothing
            logger.warning("No active jobs to determine QOS, using default of 1000")
            return 1000

        # Get the MaxSubmitJobs for this QOS
        result = subprocess.run(
            ["sacctmgr", "show", "qos", qos_name, "format=maxsubmitjobs", "-n", "-P"],
            capture_output=True,
            text=True,
            check=True,
        )

        max_submit = result.stdout.strip()
        if max_submit and max_submit.isdigit():
            limit = int(max_submit)
            logger.info("Detected MaxSubmitJobs = %d for QOS %s", limit, qos_name)
            return limit
        else:
            logger.warning("Could not parse MaxSubmitJobs, using default of 1000")
            return 1000

    except subprocess.SubprocessError as e:
        logger.error("Error getting MaxSubmitJobs: %s", e)
        return 1000
    except (ValueError, FileNotFoundError) as e:
        logger.error("Error: %s", e)
        return 1000


@beartype.beartype
def get_slurm_job_count() -> int:
    """
    Get the current number of jobs in the queue for the current user.

    Uses squeue's -r flag to properly count job array elements individually.
    For example, a job array 12345_[0-99] will be counted as 100 jobs.
    """
    try:
        # Use -r to display each array element on its own line
        result = subprocess.run(
            ["squeue", "--me", "-h", "-r"], capture_output=True, text=True, check=True
        )

        # Count non-empty lines
        lines = result.stdout.strip().split("\n")
        return len([line for line in lines if line.strip()])

    except (subprocess.SubprocessError, FileNotFoundError):
        # If we can't check, assume no jobs
        return 0


@beartype.beartype
def make_hashable(x: object) -> Hashable:
    # Primitives that are already hashable and immutable
    if x is None or isinstance(x, (bool, int, str, bytes)):
        return x

    # Floats: optionally coalesce all NaNs if you want NaN == NaN
    if isinstance(x, float):
        if math.isnan(x):
            return ("float_nan",)
        return x  # note: NaN != NaN by default

    # Byte-ish things
    if isinstance(x, (bytearray, memoryview)):
        return bytes(x)

    # Sequences
    if isinstance(x, tuple):
        return ("tuple", tuple(make_hashable(e) for e in x))
    if isinstance(x, list):
        return ("list", tuple(make_hashable(e) for e in x))

    # Sets (order-insensitive)
    if isinstance(x, set):
        return ("set", frozenset(make_hashable(e) for e in x))
    if isinstance(x, frozenset):
        return ("frozenset", frozenset(make_hashable(e) for e in x))

    # Mappings (order-insensitive)
    if isinstance(x, dict):
        return (
            "dict",
            frozenset((make_hashable(k), make_hashable(v)) for k, v in x.items()),
        )

    # Dataclasses (by fields)
    if dataclasses.is_dataclass(x):
        return (
            "dataclass",
            x.__class__,
            tuple(
                (f.name, make_hashable(getattr(x, f.name)))
                for f in dataclasses.fields(x)
            ),
        )

    # Generic Python objects: fall back to class + __dict__ (customize if needed)
    if hasattr(x, "__dict__"):
        return ("object", x.__class__, make_hashable(vars(x)))

    # Objects with __slots__
    if hasattr(x, "__slots__"):
        items = []
        for name in x.__slots__:
            if hasattr(x, name):
                items.append((name, make_hashable(getattr(x, name))))
        return ("object_slots", x.__class__, frozenset(items))

    raise TypeError(f"Unsupported type {type(x).__name__}; add a converter if needed.")


def _dumps_default(obj: object):
    if isinstance(obj, pathlib.Path):
        return str(obj)
    raise TypeError


@beartype.beartype
def dump(obj: object, fd: tp.BinaryIO, *, option: int | None = None):
    fd.write(dumps(obj, option=option))


@beartype.beartype
def dumps(obj: object, *, option: int | None = None) -> bytes:
    return orjson.dumps(obj, option=option, default=_dumps_default)


@beartype.beartype
class NumpyTopK(tp.NamedTuple):
    values: np.ndarray
    indices: np.ndarray


@beartype.beartype
def np_topk(arr: np.ndarray, k: int, axis: int | None = None) -> NumpyTopK:
    """A numpy implementation of torch.topk.

    Returns the k largest elements along the given axis. If axis is None, the array is flattened first.

    Args:
        arr: Input array.
        k: Number of top elements to return.
        axis: Axis along which to find top k elements. If None, flattens array first.

    Returns:
        Array of k largest values along the specified axis, sorted in descending order.
    """
    if axis is None:
        arr = arr.flatten()
        axis = 0

    # Handle negative axis
    if axis < 0:
        axis = arr.ndim + axis

    # For each position along other axes, sort and take top k
    # Use argsort which is stable and will preserve order for equal values
    sort_indices = np.argsort(-arr, axis=axis, kind="stable")

    # Take the first k sorted indices
    topk_indices = np.take(sort_indices, np.arange(k), axis=axis)

    # Gather the top k values
    topk_values = np.take_along_axis(arr, topk_indices, axis=axis)

    return NumpyTopK(values=topk_values, indices=topk_indices)
