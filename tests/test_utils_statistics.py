import math

import numpy as np
import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from saev.utils.statistics import calc_batch_entropy


def _tensor(values: list[int]) -> Tensor:
    return torch.tensor(values, dtype=torch.int64)


def test_calc_batch_entropy_uniform_distribution() -> None:
    example_idx = _tensor([0, 0, 1, 1])
    token_idx = _tensor([0, 1, 0, 1])

    results = calc_batch_entropy(
        example_idx,
        token_idx,
        n_examples=2,
        content_tokens_per_example=2,
    )

    assert math.isclose(results["loader/example_entropy"], math.log(2.0), rel_tol=1e-8)
    assert math.isclose(results["loader/token_entropy"], math.log(2.0), rel_tol=1e-8)
    assert math.isclose(results["loader/example_entropy_normalized"], 1.0, rel_tol=1e-8)
    assert math.isclose(results["loader/token_entropy_normalized"], 1.0, rel_tol=1e-8)
    assert math.isclose(results["loader/example_coverage"], 1.0, rel_tol=1e-8)
    assert math.isclose(results["loader/token_coverage"], 1.0, rel_tol=1e-8)


def test_calc_batch_entropy_handles_skew_and_support() -> None:
    example_idx = _tensor([0, 0, 1, 1, 1, 1])
    token_idx = _tensor([0, 0, 0, 0, 0, 0])

    results = calc_batch_entropy(
        example_idx,
        token_idx,
        n_examples=3,
        content_tokens_per_example=5,
    )

    expected_example_entropy = -(1 / 3) * math.log(1 / 3) - (2 / 3) * math.log(2 / 3)
    assert math.isclose(
        results["loader/example_entropy"], expected_example_entropy, rel_tol=1e-8
    )
    assert results["loader/token_entropy"] == pytest.approx(0.0, rel=1e-8, abs=1e-12)
    assert math.isclose(
        results["loader/example_entropy_normalized"],
        expected_example_entropy / math.log(3.0),
        rel_tol=1e-8,
    )
    assert results["loader/token_entropy_normalized"] == pytest.approx(
        0.0, rel=1e-8, abs=1e-12
    )
    assert results["loader/example_coverage"] == pytest.approx(
        2 / 3, rel=1e-8, abs=1e-12
    )
    assert results["loader/token_coverage"] == pytest.approx(1 / 5, rel=1e-8, abs=1e-12)


def test_calc_batch_entropy_accepts_numpy_inputs() -> None:
    example_idx = np.array([4, 5, 4, 5, 6], dtype=np.int32)
    token_idx = np.array([0, 0, 1, 1, 2], dtype=np.int64)

    results = calc_batch_entropy(
        example_idx,
        token_idx,
        n_examples=5,
        content_tokens_per_example=8,
    )

    assert results["loader/example_coverage"] == pytest.approx(
        3 / 5, rel=1e-8, abs=1e-12
    )
    assert results["loader/token_coverage"] == pytest.approx(3 / 8, rel=1e-8, abs=1e-12)


def test_calc_batch_entropy_validates_inputs() -> None:
    with pytest.raises(ValueError):
        calc_batch_entropy(
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            n_examples=1,
            content_tokens_per_example=1,
        )

    with pytest.raises(TypeCheckError):
        calc_batch_entropy(
            _tensor([0]),
            torch.empty(0, dtype=torch.int64),
            n_examples=1,
            content_tokens_per_example=1,
        )

    with pytest.raises(ValueError):
        calc_batch_entropy(
            _tensor([0]),
            _tensor([0]),
            n_examples=0,
            content_tokens_per_example=1,
        )

    with pytest.raises(ValueError):
        calc_batch_entropy(
            _tensor([0]),
            _tensor([0]),
            n_examples=1,
            content_tokens_per_example=0,
        )
