import torch

from . import visuals


def test_gather_batched_small():
    values = torch.arange(0, 64, dtype=torch.float).view(4, 2, 8)
    i = torch.tensor([[0], [0], [1], [1]])
    actual = visuals.gather_batched(values, i)

    expected = torch.tensor([
        [[0, 1, 2, 3, 4, 5, 6, 7]],
        [[16, 17, 18, 19, 20, 21, 22, 23]],
        [[40, 41, 42, 43, 44, 45, 46, 47]],
        [[56, 57, 58, 59, 60, 61, 62, 63]],
    ]).float()

    torch.testing.assert_close(actual, expected)
