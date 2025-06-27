import numpy as np
import pytest
from lib import cub200


@pytest.fixture(scope="session")
def cub_root(pytestconfig):
    p = pytestconfig.getoption("--cub-root")
    if p is None:
        pytest.skip("--cub-root not supplied")
    return p


def test_trait_prevalence(cub_root):
    test_y_true_NT = cub200.load_attrs(cub_root, is_train=False).numpy()
    n_test, n_traits = test_y_true_NT.shape
    pos_T = test_y_true_NT.mean(axis=0)
    assert np.median(pos_T) < 0.3
