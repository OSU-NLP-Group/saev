"""
Uses [hypothesis]() and [hypothesis-torch](https://hypothesis-torch.readthedocs.io/en/stable/compatability/) to generate test cases to compare our normalized MSE implementation to a reference MSE implementation.
"""

import hypothesis
import hypothesis.strategies as st
import hypothesis_torch
import pytest
import torch

from saev.nn import objectives


def test_mse_same():
    x = torch.ones((45, 12), dtype=torch.float)
    x_hat = torch.ones((45, 12), dtype=torch.float)
    expected = torch.zeros((45, 12), dtype=torch.float)
    actual = objectives.mean_squared_err(x_hat, x)
    torch.testing.assert_close(actual, expected)


def test_mse_zero_x_hat():
    x = torch.ones((3, 2), dtype=torch.float)
    x_hat = torch.zeros((3, 2), dtype=torch.float)
    expected = torch.ones((3, 2), dtype=torch.float)
    actual = objectives.mean_squared_err(x_hat, x, norm=False)
    torch.testing.assert_close(actual, expected)


def test_mse_nonzero():
    x = torch.full((3, 2), 3, dtype=torch.float)
    x_hat = torch.ones((3, 2), dtype=torch.float)
    expected = objectives.ref_mean_squared_err(x_hat, x)
    actual = objectives.mean_squared_err(x_hat, x)
    torch.testing.assert_close(actual, expected)


def test_safe_mse_large_x():
    x = torch.full((3, 2), 3e28, dtype=torch.float)
    x_hat = torch.ones((3, 2), dtype=torch.float)

    ref = objectives.ref_mean_squared_err(x_hat, x, norm=True)
    assert ref.isnan().any()

    safe = objectives.mean_squared_err(x_hat, x, norm=True)
    assert not safe.isnan().any()


def test_factories():
    assert isinstance(
        objectives.get_objective(objectives.Vanilla()), objectives.VanillaObjective
    )


# basic element generator
finite32 = st.floats(
    min_value=-1e9,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

tensor123 = hypothesis_torch.tensor_strategy(
    dtype=torch.float32,
    shape=(1, 2, 3),
    elements=finite32,
    layout=torch.strided,
    device=torch.device("cpu"),
)


@st.composite
def tensor_pair(draw):
    x_hat = draw(tensor123)
    x = draw(tensor123)
    # ensure denominator in your safe-mse is not zero
    hypothesis.assume(torch.linalg.norm(x, ord=2, dim=-1).max() > 1e-8)
    return x_hat, x


@pytest.mark.slow
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow], deadline=None
)
@hypothesis.given(pair=tensor_pair())
def test_safe_mse_hypothesis(pair):
    x_hat, x = pair  # both finite, same device/layout
    expected = objectives.ref_mean_squared_err(x_hat, x)
    actual = objectives.mean_squared_err(x_hat, x)
    torch.testing.assert_close(actual, expected)
