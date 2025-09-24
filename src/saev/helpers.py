# src/saev/helpers.py
import collections.abc
import dataclasses
import itertools
import logging
import os
import pathlib
import re
import subprocess
import time
import types
import typing as tp

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
        self,
        it: collections.abc.Iterable,
        *,
        every: int = 10,
        desc: str = "progress",
        total: int = 0,
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


T = tp.TypeVar("T")


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


def _recursive_dataclass_update(obj, updates: dict[str, object], base_cfg, d: int):
    """Recursively update nested dataclasses."""
    if not dataclasses.is_dataclass(obj):
        # If obj is not a dataclass, we can't update it recursively
        return updates

    result = {}
    for key, value in updates.items():
        if not hasattr(obj, key):
            # Key doesn't exist on this object, pass it through
            result[key] = value
            continue

        attr = getattr(obj, key)

        if dataclasses.is_dataclass(attr) and isinstance(value, dict):
            # Recursively update the nested dataclass
            nested_updates = _recursive_dataclass_update(attr, value, base_cfg, d)

            # Handle seed updates for nested objects
            if hasattr(attr, "seed") and "seed" not in nested_updates:
                base_seed = getattr(base_cfg, "seed", 0) if base_cfg else 0
                nested_updates["seed"] = getattr(attr, "seed", 0) + base_seed + d

            # Create a new instance of the nested dataclass with updates
            result[key] = dataclasses.replace(attr, **nested_updates)
        else:
            # Direct assignment for non-dataclass fields or non-dict values
            result[key] = value

    return result


@beartype.beartype
def grid(cfg: T, sweep_dct: dict[str, object]) -> tuple[list[T], list[str]]:
    """Generate configs from ``cfg`` according to ``sweep_dct``."""

    cfgs: list[T] = []
    errs: list[str] = []

    for d, dct in enumerate(expand(sweep_dct)):
        updates = _recursive_dataclass_update(cfg, dct, cfg, d)

        if hasattr(cfg, "seed") and "seed" not in updates:
            updates["seed"] = getattr(cfg, "seed", 0) + d

        try:
            cfgs.append(dataclasses.replace(cfg, **updates))
        except Exception as err:
            errs.append(str(err))

    return cfgs, errs


@beartype.beartype
def dict_to_dataclass(data: dict, cls: type[T]) -> T:
    """Recursively convert a dictionary to a dataclass instance."""
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}

    for field_name, field_type in field_types.items():
        if field_name not in data:
            continue

        value = data[field_name]

        # Handle Optional types
        origin = tp.get_origin(field_type)
        args = tp.get_args(field_type)

        # Handle tuple[str, ...]
        if origin is tuple and args:
            kwargs[field_name] = tuple(value) if isinstance(value, list) else value
        # Handle list[DataclassType]
        elif origin is list and args and dataclasses.is_dataclass(args[0]):
            kwargs[field_name] = [dict_to_dataclass(item, args[0]) for item in value]
        # Handle regular dataclass fields
        elif dataclasses.is_dataclass(field_type):
            kwargs[field_name] = dict_to_dataclass(value, field_type)
        # Handle pathlib.Path
        elif field_type is pathlib.Path:
            # Required Path field - always convert
            kwargs[field_name] = pathlib.Path(value) if value is not None else value
        elif origin is tp.Union and pathlib.Path in args:
            # Optional Path field (typing.Union style)
            kwargs[field_name] = pathlib.Path(value) if value is not None else value
        elif origin is types.UnionType and pathlib.Path in args:
            # Optional Path field (Python 3.10+ union style with |)
            kwargs[field_name] = pathlib.Path(value) if value is not None else value
        else:
            kwargs[field_name] = value

    return cls(**kwargs)


@beartype.beartype
def get_non_default_values(obj: T, default_obj: T) -> dict:
    """Recursively find fields that differ from defaults."""
    # Check that obj and default_obj are instances of a dataclass.
    assert dataclasses.is_dataclass(obj) and not isinstance(obj, type)
    assert dataclasses.is_dataclass(default_obj) and not isinstance(default_obj, type)

    obj_dict = dataclasses.asdict(obj)
    default_dict = dataclasses.asdict(default_obj)

    diff = {}
    for key, value in obj_dict.items():
        default_value = default_dict.get(key)
        if value != default_value:
            diff[key] = value

    return diff


@beartype.beartype
def merge_configs(base: T, overrides: dict) -> T:
    """Recursively merge override values into a base config."""
    if not overrides:
        return base

    # Check that base is an instance of a dataclass.
    assert dataclasses.is_dataclass(base) and not isinstance(base, type)

    base_dict = dataclasses.asdict(base)

    for key, value in overrides.items():
        if key in base_dict:
            # For nested dataclasses, merge recursively
            if isinstance(value, dict) and dataclasses.is_dataclass(getattr(base, key)):
                base_dict[key] = dataclasses.asdict(
                    merge_configs(getattr(base, key), value)
                )
            else:
                base_dict[key] = value

    return dict_to_dataclass(base_dict, type(base))


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
