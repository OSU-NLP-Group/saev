import json

import pytest

from saev.disk import Run


@pytest.fixture
def tmp_run_root(tmp_path):
    """Create a temporary run root directory."""
    run_root = tmp_path / "runs" / "test_run_123"
    return run_root


@pytest.fixture
def populated_run_root(tmp_path):
    """Create a populated run directory with all expected structure."""
    run_root = tmp_path / "runs" / "test_run_456"
    run_root.mkdir(parents=True)

    # Create checkpoint directory with sae.pt and config.json
    checkpoint = run_root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "sae.pt").touch()
    (checkpoint / "config.json").write_text(
        json.dumps({"lr": 0.001, "epochs": 10}), encoding="utf-8"
    )

    # Create links directory
    links = run_root / "links"
    links.mkdir()

    # Create shard directory structure
    scratch = tmp_path / "scratch" / "shards" / "abc123"
    scratch.mkdir(parents=True)
    (scratch / "metadata.json").touch()
    (scratch / "acts000000.bin").touch()
    (scratch / "labels.bin").touch()

    # Create dataset directory
    dataset = tmp_path / "datasets" / "butterflies"
    dataset.mkdir(parents=True)

    # Create symlinks
    (links / "shards").symlink_to(scratch)
    (links / "dataset").symlink_to(dataset)

    # Create inference directory
    inference = run_root / "inference"
    inference.mkdir()

    return run_root


def test_new_run_creates_directories(tmp_path):
    """Test that creating a new run creates all expected directories."""
    run_root = tmp_path / "runs"
    shards_dpath = tmp_path / "scratch" / "shards" / "abc123"
    shards_dpath.mkdir(parents=True)
    dataset_dpath = tmp_path / "datasets" / "butterflies"
    dataset_dpath.mkdir(parents=True)

    run = Run.new("test_run_123", shards_dpath, dataset_dpath, run_root=run_root)

    assert run.root.exists()
    assert (run.root / "checkpoint").exists()
    assert (run.root / "links").exists()
    assert (run.root / "inference").exists()
    assert (run.root / "links" / "shards").is_symlink()
    assert (run.root / "links" / "dataset").is_symlink()
    assert run.shards == shards_dpath
    assert run.dataset == dataset_dpath


def test_existing_run_validates_structure(populated_run_root):
    """Test that an existing run validates its directory structure."""
    run = Run(populated_run_root)

    assert run.ckpt.exists()
    assert run.shards.exists()
    assert run.dataset.exists()
    assert run.inference.exists()


def test_run_id_property(populated_run_root):
    """Test that run_id returns the directory name."""
    run = Run(populated_run_root)

    assert run.run_id == "test_run_456"


def test_config_property_loads_json(populated_run_root):
    """Test that config property loads the checkpoint config.json."""
    run = Run(populated_run_root)

    config = run.config
    assert config["lr"] == 0.001
    assert config["epochs"] == 10


def test_ckpt_property_returns_checkpoint_path(populated_run_root):
    """Test that ckpt property returns path to sae.pt."""
    run = Run(populated_run_root)

    assert run.ckpt == populated_run_root / "checkpoint" / "sae.pt"
    assert run.ckpt.exists()


def test_shards_property_resolves_symlink(populated_run_root):
    """Test that shards property resolves the symlink correctly."""
    run = Run(populated_run_root)

    assert run.shards.is_absolute()
    assert (run.shards / "metadata.json").exists()
    assert (run.shards / "acts000000.bin").exists()
    assert (run.shards / "labels.bin").exists()


def test_dataset_property_resolves_symlink(populated_run_root):
    """Test that dataset property resolves the dataset symlink correctly."""
    run = Run(populated_run_root)

    assert run.dataset.is_absolute()
    assert run.dataset.exists()
    assert run.dataset.name == "butterflies"


def test_inference_property_returns_inference_dir(populated_run_root):
    """Test that inference property returns the inference directory."""
    run = Run(populated_run_root)

    assert run.inference == populated_run_root / "inference"
    assert run.inference.exists()


def test_existing_run_missing_checkpoint_raises(tmp_path):
    """Test that loading an existing run without checkpoint raises an error."""
    run_root = tmp_path / "runs" / "broken_run"
    run_root.mkdir(parents=True)
    (run_root / "links").mkdir()
    (run_root / "inference").mkdir()

    with pytest.raises(FileNotFoundError, match="Use Run.new()"):
        Run(run_root)


def test_existing_run_missing_links_raises(tmp_path):
    """Test that loading an existing run without links directory raises an error."""
    run_root = tmp_path / "runs" / "broken_run"
    run_root.mkdir(parents=True)
    checkpoint = run_root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "sae.pt").touch()
    (run_root / "inference").mkdir()

    with pytest.raises(FileNotFoundError, match="Use Run.new()"):
        Run(run_root)


def test_run_missing_root_raises(tmp_path):
    """Test that Run() raises helpful error for missing directory."""
    run_root = tmp_path / "runs" / "nonexistent"

    with pytest.raises(FileNotFoundError, match="Use Run.new()"):
        Run(run_root)


def test_config_property_missing_config_raises(tmp_path):
    """Test that accessing config without config.json raises an error."""
    run_root = tmp_path / "runs" / "no_config_run"
    run_root.mkdir(parents=True)
    checkpoint = run_root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "sae.pt").touch()

    links = run_root / "links"
    links.mkdir()

    scratch = tmp_path / "scratch" / "shards" / "xyz789"
    scratch.mkdir(parents=True)
    (links / "shards").symlink_to(scratch)

    dataset = tmp_path / "datasets" / "test"
    dataset.mkdir(parents=True)
    (links / "dataset").symlink_to(dataset)

    (run_root / "inference").mkdir()

    run = Run(run_root)

    with pytest.raises(FileNotFoundError):
        _ = run.config
