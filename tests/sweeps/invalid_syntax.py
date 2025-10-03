# fmt: off 
"""Sweep file with invalid Python syntax."""

def make_cfgs() -> list[dict]
    return [{"lr": 1e-4}]  # Missing colon after function def
