import beartype
import numpy as np
import pytest
import torch
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jaxtyping import Bool, Float, jaxtyped
from lib import metrics
from sklearn.metrics import average_precision_score
from torch import Tensor


@jaxtyped(typechecker=beartype.beartype)
def ap_ref(y: Bool[Tensor, "..."], s: Float[Tensor, "..."]) -> float:
    """sklearn baseline, y,s 1-D numpy"""
    print(y.shape, s.shape)
    return average_precision_score(y.cpu().numpy(), s.cpu().numpy())


@beartype.beartype
def torch_array(shape: tuple[int, ...], *, floats=True):
    if floats:
        return arrays(np.float32, shape, elements=st.floats(-5, 5))
    return arrays(np.bool_, shape)


def test_perfect_rank():
    y = torch.tensor([[1], [1], [0], [0]], dtype=torch.bool)  # N=4,T=1
    s = torch.tensor([[0.9], [0.8], [0.1], [0.0]])  # N=4,C=1
    ap = metrics.calc_avg_prec(s, y).squeeze()
    assert torch.allclose(ap, torch.tensor(1.0))


def test_perfect_rank_multi():
    """
    N = 4 images
    C = 2 prototypes
    T = 3 traits
    Prototype-0 gives perfect (or near-perfect) ordering for traits 0 and 1.
    Prototype-1 gives the reverse ordering, so it is better for trait-2.
    """

    # Trait matrix  (N=4, T=3)
    # img0 img1 img2 img3
    #  1     1    0    0   ← trait-0
    #  1     0    1    0   ← trait-1
    #  0     1    0    1   ← trait-2
    y = torch.tensor(
        [
            [1, 1, 0],  # img 0
            [1, 0, 1],  # img 1
            [0, 1, 0],  # img 2
            [0, 0, 1],  # img 3
        ],
        dtype=torch.bool,
    )

    # Score matrix  (N=4, C=2)
    # Prototype-0:  descending scores 0.9 > 0.8 > 0.7 > 0.6  (good order)
    # Prototype-1:  ascending scores  0.1 < 0.2 < 0.3 < 0.4  (reverse order)
    s = torch.tensor([
        [0.9, 0.1],  # img 0
        [0.8, 0.2],  # img 1
        [0.7, 0.3],  # img 2
        [0.6, 0.4],  # img 3
    ])

    ap = metrics.calc_avg_prec(s, y)  # shape (C=2, T=3)

    expected = torch.tensor(
        [
            [1.0, 5 / 6, 0.5],  # prototype-0
            [5 / 12, 0.5, 5 / 6],  # prototype-1
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(ap, expected, atol=1e-6)


def test_alternating_rank():
    y = torch.tensor([[1], [0], [1], [0], [1], [0]], dtype=torch.bool)
    s = torch.tensor([[0.9], [0.8], [0.7], [0.6], [0.5], [0.4]])
    ap = metrics.calc_avg_prec(s, y).squeeze()
    hand = torch.tensor((1 / 1 + 2 / 3 + 3 / 5) / 3)  # 0.636...
    assert torch.allclose(ap, hand, atol=1e-6)


def test_all_negatives():
    y = torch.zeros(5, 1, dtype=torch.bool)
    s = torch.rand(5, 1)
    ap = metrics.calc_avg_prec(s, y).squeeze()
    assert ap == 0.0


def test_all_positives():
    y = torch.ones(5, 1, dtype=torch.bool)
    s = torch.rand(5, 1)
    ap = metrics.calc_avg_prec(s, y).squeeze()
    assert torch.allclose(ap, torch.tensor(1.0))


@pytest.mark.parametrize("N,C,T", [(50, 7, 3), (100, 2, 5)])
def test_against_sklearn(N, C, T):
    torch.manual_seed(0)
    y = torch.randint(0, 2, (N, T)).bool()
    s = torch.randn(N, C)
    ap = metrics.calc_avg_prec(s, y)  # (C,T)
    for c in range(C):
        for t in range(T):
            ref = ap_ref(y[:, t], s[:, c])
            assert abs(ap[c, t].item() - ref) < 1e-6


def test_identical_scores():
    """All scores identical - should handle ties gracefully"""
    y = torch.tensor([[1], [0], [1], [0]], dtype=torch.bool)
    s = torch.tensor([[0.5], [0.5], [0.5], [0.5]])  # all same score
    ap = metrics.calc_avg_prec(s, y).squeeze()
    # With ties, order is arbitrary but AP should be reasonable
    assert 0.0 <= ap <= 1.0


def test_single_image():
    """Edge case: N=1"""
    y = torch.tensor([[1]], dtype=torch.bool)
    s = torch.tensor([[0.7]])
    ap = metrics.calc_avg_prec(s, y).squeeze()
    assert torch.allclose(ap, torch.tensor(1.0))

    y = torch.tensor([[0]], dtype=torch.bool)
    s = torch.tensor([[0.7]])
    ap = metrics.calc_avg_prec(s, y).squeeze()
    assert ap == 0.0


def test_extreme_scores():
    """Very large/small scores shouldn't break anything"""
    y = torch.tensor([[1], [0], [1]], dtype=torch.bool)
    s = torch.tensor([[1e6], [-1e6], [0.0]])
    ap = metrics.calc_avg_prec(s, y).squeeze()
    assert torch.allclose(ap, torch.tensor(1.0))


def test_batch_independence():
    """Different prototypes/traits shouldn't interfere"""
    torch.manual_seed(42)
    y = torch.randint(0, 2, (20, 4)).bool()
    s1 = torch.randn(20, 1)
    s2 = torch.randn(20, 1)

    # Calculate separately
    ap1 = metrics.calc_avg_prec(s1, y)
    ap2 = metrics.calc_avg_prec(s2, y)

    # Calculate together
    s_combined = torch.cat([s1, s2], dim=1)
    ap_combined = metrics.calc_avg_prec(s_combined, y)

    assert torch.allclose(ap_combined[0], ap1[0])
    assert torch.allclose(ap_combined[1], ap2[0])


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_device_consistency(device):
    """Results should be identical across devices"""
    torch.manual_seed(123)
    y = torch.randint(0, 2, (30, 2)).bool()
    s = torch.randn(30, 3)

    ap_cpu = metrics.calc_avg_prec(s, y)
    ap_device = metrics.calc_avg_prec(s.to(device), y.to(device))

    assert torch.allclose(ap_cpu, ap_device.cpu(), atol=1e-6)
