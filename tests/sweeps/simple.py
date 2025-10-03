"""Simple sweep for testing."""


def make_cfgs() -> list[dict]:
    return [
        {"lr": 1e-4, "objective": {"sparsity_coeff": 4e-4}},
        {"lr": 3e-4, "objective": {"sparsity_coeff": 8e-4}},
        {"lr": 1e-3, "objective": {"sparsity_coeff": 1.6e-3}},
    ]
