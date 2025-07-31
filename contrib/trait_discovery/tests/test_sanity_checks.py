import numpy as np
import pytest
import sklearn.metrics
from lib import cub200


@pytest.fixture(scope="session")
def cub_root(pytestconfig):
    p = pytestconfig.getoption("--cub-root")
    if p is None:
        pytest.skip("--cub-root not supplied")
    return p


def test_trait_prevalence_test(cub_root):
    """Expect trait prevalance to be between 0 and 1, and the median trait should be present in less than 30% of all test images."""
    test_y_true_NT = cub200.load_attrs(cub_root, is_train=False).numpy()
    pos_T = test_y_true_NT.mean(axis=0)
    assert np.median(pos_T) < 0.3


def test_trait_prevalence_train(cub_root):
    """Expect trait prevalance to be between 0 and 1, and the median trait should be present in less than 30% of all trainimages."""
    test_y_true_NT = cub200.load_attrs(cub_root, is_train=True).numpy()
    pos_T = test_y_true_NT.mean(axis=0)
    assert np.median(pos_T) < 0.3


def test_constant_map(cub_root):
    """Making trivial predictions (equal scores) should lead to very bad mAP."""
    test_y_true_NT = cub200.load_attrs(cub_root, is_train=False).numpy()
    n_test, n_traits = test_y_true_NT.shape
    pred_N = np.zeros(n_test)
    const_ap_T = np.array([
        sklearn.metrics.average_precision_score(test_y_true_NT[:, i], pred_N)
        for i in range(n_traits)
    ])

    assert const_ap_T.mean() < 0.2
