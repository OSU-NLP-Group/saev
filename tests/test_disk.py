import json

import pytest

from saev.disk import Run


@pytest.fixture
def tmp_runs_root(tmp_path):
    """Create a temporary run root directory."""
    runs_root = tmp_path / "saev" / "runs"
    return runs_root


@pytest.fixture
def populated_runs_root(tmp_path):
    """Create a populated run directory with all expected structure."""
    runs_root = tmp_path / "saev" / "runs" / "test_run_456"
    runs_root.mkdir(parents=True)

    # Create checkpoint directory with sae.pt and config.json
    checkpoint = runs_root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "sae.pt").touch()
    (checkpoint / "config.json").write_text(
        json.dumps({"lr": 0.001, "epochs": 10}), encoding="utf-8"
    )

    # Create links directory
    links = runs_root / "links"
    links.mkdir()

    # Create train shard directory structure
    train_scratch = tmp_path / "scratch" / "shards" / "abc123"
    train_scratch.mkdir(parents=True)
    (train_scratch / "metadata.json").touch()
    (train_scratch / "acts000000.bin").touch()
    (train_scratch / "labels.bin").touch()

    # Create val shard directory structure
    val_scratch = tmp_path / "scratch" / "shards" / "def456"
    val_scratch.mkdir(parents=True)
    (val_scratch / "metadata.json").touch()
    (val_scratch / "acts000000.bin").touch()
    (val_scratch / "labels.bin").touch()

    # Create symlinks
    (links / "train-shards").symlink_to(train_scratch)
    (links / "val-shards").symlink_to(val_scratch)

    # Create inference directory
    inference = runs_root / "inference"
    inference.mkdir()

    return runs_root


def test_new_run_creates_directories(tmp_path):
    """Test that creating a new run creates all expected directories."""
    runs_root = tmp_path / "saev" / "runs"
    train_shards_dpath = tmp_path / "scratch" / "shards" / "abc123"
    train_shards_dpath.mkdir(parents=True)
    val_shards_dpath = tmp_path / "scratch" / "shards" / "def456"
    val_shards_dpath.mkdir(parents=True)

    run = Run.new(
        run_id="test_run_123",
        train_shards_dir=train_shards_dpath,
        val_shards_dir=val_shards_dpath,
        runs_root=runs_root,
    )

    assert run.run_dir.exists()
    assert (run.run_dir / "checkpoint").exists()
    assert (run.run_dir / "links").exists()
    assert (run.run_dir / "inference").exists()
    assert (run.run_dir / "links" / "train-shards").is_symlink()
    assert (run.run_dir / "links" / "val-shards").is_symlink()
    assert run.train_shards == train_shards_dpath
    assert run.val_shards == val_shards_dpath


def test_existing_run_validates_structure(populated_runs_root):
    """Test that an existing run validates its directory structure."""
    run = Run(populated_runs_root)

    assert run.ckpt.exists()
    assert run.train_shards.exists()
    assert run.val_shards.exists()
    assert run.inference.exists()


def test_run_id_property(populated_runs_root):
    """Test that run_id returns the directory name."""
    run = Run(populated_runs_root)

    assert run.run_id == "test_run_456"


def test_config_property_loads_json(populated_runs_root):
    """Test that config property loads the checkpoint config.json."""
    run = Run(populated_runs_root)

    config = run.config
    assert config["lr"] == 0.001
    assert config["epochs"] == 10


def test_ckpt_property_returns_checkpoint_path(populated_runs_root):
    """Test that ckpt property returns path to sae.pt."""
    run = Run(populated_runs_root)

    assert run.ckpt == populated_runs_root / "checkpoint" / "sae.pt"
    assert run.ckpt.exists()


def test_shards_property_resolves_symlink(populated_runs_root):
    """Test that shards property resolves the symlink correctly."""
    run = Run(populated_runs_root)

    assert run.train_shards.is_absolute()
    assert (run.train_shards / "metadata.json").exists()
    assert (run.train_shards / "acts000000.bin").exists()
    assert (run.train_shards / "labels.bin").exists()

    assert run.val_shards.is_absolute()
    assert (run.val_shards / "metadata.json").exists()
    assert (run.val_shards / "acts000000.bin").exists()
    assert (run.val_shards / "labels.bin").exists()


def test_inference_property_returns_inference_dir(populated_runs_root):
    """Test that inference property returns the inference directory."""
    run = Run(populated_runs_root)

    assert run.inference == populated_runs_root / "inference"
    assert run.inference.exists()


def test_existing_run_missing_checkpoint_raises(tmp_path):
    """Test that loading an existing run without checkpoint raises an error."""
    runs_root = tmp_path / "saev" / "runs" / "broken_run"
    runs_root.mkdir(parents=True)
    (runs_root / "links").mkdir()
    (runs_root / "inference").mkdir()

    with pytest.raises(FileNotFoundError, match="Use Run.new()"):
        Run(runs_root)


def test_existing_run_missing_links_raises(tmp_path):
    """Test that loading an existing run without links directory raises an error."""
    runs_root = tmp_path / "saev" / "runs" / "broken_run"
    runs_root.mkdir(parents=True)
    checkpoint = runs_root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "sae.pt").touch()
    (runs_root / "inference").mkdir()

    with pytest.raises(FileNotFoundError, match="Use Run.new()"):
        Run(runs_root)


def test_run_missing_root_raises(tmp_path):
    """Test that Run() raises helpful error for missing directory."""
    runs_root = tmp_path / "saev" / "runs" / "nonexistent"

    with pytest.raises(FileNotFoundError, match="Use Run.new()"):
        Run(runs_root)


def test_config_property_missing_config_raises(tmp_path):
    """Test that accessing config without config.json raises an error."""
    runs_root = tmp_path / "saev" / "runs" / "no_config_run"
    runs_root.mkdir(parents=True)
    checkpoint = runs_root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "sae.pt").touch()

    links = runs_root / "links"
    links.mkdir()

    train_scratch = tmp_path / "scratch" / "shards" / "xyz789"
    train_scratch.mkdir(parents=True)
    (links / "train-shards").symlink_to(train_scratch)

    val_scratch = tmp_path / "scratch" / "shards" / "uvw012"
    val_scratch.mkdir(parents=True)
    (links / "val-shards").symlink_to(val_scratch)

    (runs_root / "inference").mkdir()

    run = Run(runs_root)

    with pytest.raises(FileNotFoundError):
        _ = run.config
