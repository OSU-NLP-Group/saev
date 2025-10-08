import contextlib
import pathlib
import tempfile

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--shards",
        action="store",
        default=None,
        help="Root directory with activation shards",
    )
    parser.addoption(
        "--ckpt-path",
        action="store",
        default=None,
        help="Path to a checkpoint file for testing load functionality",
    )


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


pytest.helpers = namespace
