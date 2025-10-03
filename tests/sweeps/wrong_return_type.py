"""Sweep file where make_cfgs() returns wrong type."""


def make_cfgs():
    return {"lr": 1e-4}  # Should return list, not dict
