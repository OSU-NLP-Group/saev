# src/saev/disk.py
"""
Helpers for sticking with the layout described in [disk-layout.md](../developers/disk-layout.md).
"""

import json
import pathlib

import beartype


@beartype.beartype
def is_run_root(path: pathlib.Path) -> bool:
    """
    Check if `path` is a valid run root directory.

    A valid run root ends with `saev/runs` and exists as a directory.

    Args:
        path: Path to check.

    Returns:
        True if path is a directory ending in saev/runs.
    """
    return path.is_dir() and path.parts[-2:] == ("saev", "runs")


@beartype.beartype
def is_shards_root(path: pathlib.Path) -> bool:
    """
    Check if `path` is a valid shards root directory.

    A valid shards root ends with `saev/shards` and exists as a directory.

    Args:
        path: Path to check.

    Returns:
        True if path is a directory ending in saev/shards.
    """
    return path.is_dir() and path.parts[-2:] == ("saev", "shards")


@beartype.beartype
def is_shards_dir(path: pathlib.Path) -> bool:
    """
    Check if `path` is a specific shards directory.

    A valid shards directory ends with `saev/shards/<hash>` for any hash value, exists as a directory, and contains the required files (metadata.json, shards.json, labels.bin).

    Args:
        path: Path to check.

    Returns:
        True if path is a directory ending in saev/shards/<hash> with required files.
    """
    if not path.is_dir():
        return False

    if len(path.parts) < 3 or path.parts[-3:-1] != ("saev", "shards"):
        return False

    return True


@beartype.beartype
class Run:
    """
    Represents an SAE training run and some associated data.

    Args:
        root: Root directory, should be $SAEV_NFS/saev/runs/<run_id>. Assumes the run already exists and validates the structure. Use `Run.new()` to create a new run.
    """

    def __init__(self, root: pathlib.Path):
        self.root = root

        if not self.root.exists():
            raise FileNotFoundError(
                f"Run directory does not exist: {self.root}. Use Run.new() to create a new run."
            )
        if not (self.root / "checkpoint").exists():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {self.root / 'checkpoint'}. Use Run.new() to create a new run."
            )
        if not (self.root / "links").exists():
            raise FileNotFoundError(
                f"Links directory does not exist: {self.root / 'links'}. Use Run.new() to create a new run."
            )
        if not (self.root / "inference").exists():
            raise FileNotFoundError(
                f"Inference directory does not exist: {self.root / 'inference'}. Use Run.new() to create a new run."
            )

    @classmethod
    def new(
        cls,
        run_id: str,
        shards_dir: pathlib.Path,
        dataset: pathlib.Path,
        *,
        run_root: pathlib.Path,
    ) -> "Run":
        """
        Create a new run with directory structure and symlinks.

        Args:
            run_id: The run ID (typically from wandb).
            shards_dir: Absolute path to the shards directory (typically $SAEV_SCRATCH/saev/shards/<shard_hash>).
            dataset: Absolute path to the dataset directory.
            run_root: Root directory for runs (typically $SAEV_NFS/saev/runs).

        Returns:
            A new Run instance with all directories and symlinks created.
        """
        root = run_root / run_id
        root.mkdir(parents=True)
        (root / "checkpoint").mkdir()
        (root / "links").mkdir()
        (root / "inference").mkdir()

        (root / "links" / "shards").symlink_to(shards_dir)
        (root / "links" / "dataset").symlink_to(dataset)

        return cls(root)

    @property
    def run_id(self) -> str:
        """The run ID, created by wandb."""
        return self.root.name

    @property
    def config(self) -> dict[str, object]:
        """The training run config. Not a train.Config object because we don't want to import from train.py."""
        config_fpath = self.root / "checkpoint" / "config.json"
        with open(config_fpath, encoding="utf-8") as fd:
            return json.load(fd)

    @property
    def ckpt(self) -> pathlib.Path:
        """Path to the sae.pt checkpoint."""
        return self.root / "checkpoint" / "sae.pt"

    @property
    def shards(self) -> pathlib.Path:
        """Path to shard root with metadata.json and acts*.bin files."""
        return (self.root / "links" / "shards").resolve()

    @property
    def dataset(self) -> pathlib.Path:
        """Path to dataset root."""
        return (self.root / "links" / "dataset").resolve()

    @property
    def inference(self) -> pathlib.Path:
        """Path to the inference/ directory."""
        return self.root / "inference"
