# src/saev/disk.py
"""
Helpers for sticking with the layout described in [disk-layout.md](../developers/disk-layout.md).
"""

import json
import pathlib

import beartype


@beartype.beartype
def is_runs_root(path: pathlib.Path) -> bool:
    """
    Check if `path` is a valid runs root directory.

    A valid runs root ends with `saev/runs` and exists as a directory.

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
        run_dir: Run directory, should be $SAEV_NFS/saev/runs/<run_id>. Assumes the run already exists and validates the structure. Use `Run.new()` to create a new run.
    """

    def __init__(self, run_dir: pathlib.Path):
        self.run_dir = run_dir

        if len(self.run_dir.parts) < 3 or self.run_dir.parts[-3:-1] != ("saev", "runs"):
            raise ValueError("Run directory is invalid.")

        if not self.run_dir.exists():
            raise FileNotFoundError(
                f"Run directory does not exist: {self.run_dir}. Use Run.new() to create a new run."
            )
        if not (self.run_dir / "checkpoint").exists():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {self.run_dir / 'checkpoint'}. Use Run.new() to create a new run."
            )
        if not (self.run_dir / "links").exists():
            raise FileNotFoundError(
                f"Links directory does not exist: {self.run_dir / 'links'}. Use Run.new() to create a new run."
            )
        if not (self.run_dir / "inference").exists():
            raise FileNotFoundError(
                f"Inference directory does not exist: {self.run_dir / 'inference'}. Use Run.new() to create a new run."
            )

    @classmethod
    def new(
        cls,
        run_id: str,
        *,
        train_shards_dir: pathlib.Path,
        val_shards_dir: pathlib.Path,
        runs_root: pathlib.Path,
    ) -> "Run":
        """
        Create a new run with directory structure and symlinks.

        Args:
            run_id: The run ID (typically from wandb).
            train_shards_dir: Absolute path to the train shards directory (typically $SAEV_SCRATCH/saev/shards/<shard_hash>).
            val_shards_dir: Absolute path to the val shards directory (typically $SAEV_SCRATCH/saev/shards/<shard_hash>).
            runs_root: Root directory for runs (typically $SAEV_NFS/saev/runs).

        Returns:
            A new Run instance with all directories and symlinks created.
        """
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "checkpoint").mkdir()
        (run_dir / "links").mkdir()
        (run_dir / "inference").mkdir()

        (run_dir / "links" / "train-shards").symlink_to(train_shards_dir)
        (run_dir / "links" / "val-shards").symlink_to(val_shards_dir)

        return cls(run_dir)

    @property
    def run_id(self) -> str:
        """The run ID, created by wandb."""
        return self.run_dir.name

    @property
    def config(self) -> dict[str, object]:
        """The training run config. Not a train.Config object because we don't want to import from train.py."""
        config_fpath = self.run_dir / "checkpoint" / "config.json"
        with open(config_fpath, encoding="utf-8") as fd:
            return json.load(fd)

    @property
    def ckpt(self) -> pathlib.Path:
        """Path to the sae.pt checkpoint."""
        return self.run_dir / "checkpoint" / "sae.pt"

    @property
    def val_shards(self) -> pathlib.Path:
        """Path to shard root with metadata.json and acts*.bin files."""
        return (self.run_dir / "links" / "val-shards").resolve()

    @property
    def train_shards(self) -> pathlib.Path:
        """Path to shard root with metadata.json and acts*.bin files."""
        return (self.run_dir / "links" / "train-shards").resolve()

    @property
    def inference(self) -> pathlib.Path:
        """Path to the inference/ directory."""
        return self.run_dir / "inference"
