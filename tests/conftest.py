import contextlib
import pathlib
import tempfile

import beartype
import pytest

import saev.data


def pytest_addoption(parser):
    parser.addoption(
        "--shards",
        action="extend",
        nargs="+",
        default=[],
        help="Root directory with activation shards",
    )
    parser.addoption(
        "--ckpt-path",
        action="store",
        default=None,
        help="Path to a checkpoint file for testing load functionality",
    )


def pytest_generate_tests(metafunc):
    if "shards_dir" in metafunc.fixturenames:
        paths = [pathlib.Path(p) for p in metafunc.config.getoption("--shards")]

        ids = [f"shards={p.name[:8]}" for p in paths]
        metafunc.parametrize("shards_dir", paths, ids=ids, indirect=True)


@pytest.fixture(scope="session")
def shards_dir(pytestconfig, request) -> pathlib.Path:
    path = getattr(request, "param")
    assert isinstance(path, pathlib.Path)
    if not path.exists():
        pytest.skip(f"--shards path does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def shards_dir_with_token_labels(shards_dir):
    """Fixture for shards that have a labels.bin file."""
    labels_path = shards_dir / "labels.bin"
    if not labels_path.exists():
        pytest.skip("--shards has no labels.bin")

    return shards_dir


@beartype.beartype
class namespace:
    @staticmethod
    @contextlib.contextmanager
    def tmp_shards_root():
        """Create a temporary shard root directory."""
        # We cannot use the tmp_path fixture because of Hypothesis.
        # See https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck.function_scoped_fixture
        with tempfile.TemporaryDirectory() as tmp_path:
            shards_root = pathlib.Path(tmp_path) / "saev" / "shards"
            shards_root.mkdir(parents=True)
            yield shards_root

    @staticmethod
    @contextlib.contextmanager
    def tmp_runs_root():
        """Create a temporary runs directory."""
        # We cannot use the tmp_path fixture because of Hypothesis.
        # See https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck.function_scoped_fixture
        with tempfile.TemporaryDirectory() as tmp_path:
            runs_root = pathlib.Path(tmp_path) / "saev" / "runs"
            runs_root.mkdir(parents=True)
            yield runs_root

    @staticmethod
    def write_shards(shards_root: pathlib.Path, **kwargs) -> pathlib.Path:
        from saev.data import shards

        default_kwargs = dict(
            data=saev.data.datasets.FakeImg(n_examples=2),
            family="clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            d_model=128,
            layers=[0],
            content_tokens_per_example=16,
            cls_token=True,  # This model has a [CLS] token
            max_tokens_per_shard=1000,
            batch_size=4,
            n_workers=0,
            device="cpu",
            shards_root=shards_root,
        )
        default_kwargs.update(kwargs)
        return shards.worker_fn(**default_kwargs)

    @staticmethod
    def write_token_labels(shards_dir: pathlib.Path):
        """Create distinct labels for each patch position across all images"""
        import numpy as np

        md = saev.data.Metadata.load(shards_dir)

        labels = np.memmap(
            shards_dir / "labels.bin",
            mode="w+",
            dtype=np.uint8,
            shape=(md.n_examples, md.content_tokens_per_example),
        )
        for example_idx in range(md.n_examples):
            for token_idx in range(md.content_tokens_per_example):
                # Create a unique label based on image and patch index
                labels[example_idx, token_idx] = (example_idx * 10 + token_idx) % 150

        labels.flush()


pytest.helpers = namespace
