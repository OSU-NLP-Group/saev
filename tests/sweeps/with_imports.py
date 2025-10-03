"""Sweep file that uses imports and helper functions."""

import itertools


def _make_grid(lrs, sparsities):
    """Helper function to create grid."""
    return [{"lr": lr, "sparsity": sp} for lr, sp in itertools.product(lrs, sparsities)]


def make_cfgs() -> list[dict]:
    return _make_grid([1e-4, 3e-4], [4e-4, 8e-4])
