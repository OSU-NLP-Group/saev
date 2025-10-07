"""Test for axis vs dim parameter inconsistency in objectives.py

This test demonstrates that the code uses non-standard 'axis' parameter
instead of PyTorch's canonical 'dim' parameter in some places.
"""

import pytest
import torch

from saev.nn import objectives


def test_type_checker_should_pass_on_objectives():
    """
    This test verifies that the objectives module uses PyTorch's canonical
    parameter names. The type checker (ty/pyright) should not flag any errors
    when checking the objectives module.

    Currently, the code uses 'axis' instead of 'dim' in:
    - VanillaObjective.forward() line 107: (f_x > 0).float().sum(axis=1).mean(axis=0)
    - MatryoshkaObjective.forward() line 174: (f_x > 0).float().sum(axis=1).mean(axis=0)
    - MatryoshkaObjective.forward() line 175: f_x.abs().sum(axis=1).mean(axis=0)

    While PyTorch accepts 'axis' at runtime for backwards compatibility with NumPy,
    the canonical parameter name is 'dim' and using 'axis' causes type checker errors.
    """
    import subprocess

    result = subprocess.run(
        ["uvx", "ty", "check", "src/saev/nn/objectives.py"],
        capture_output=True,
        text=True,
    )

    # The type checker should not report any errors
    # Currently this test will FAIL because of the axis/dim bug
    assert "error[no-matching-overload]" not in result.stderr, (
        f"Type checker found errors in objectives.py:\n{result.stderr}"
    )


def test_vanilla_objective_parameter_consistency():
    """
    Test that VanillaObjective uses consistent parameter names internally.

    Currently line 107 uses 'axis' while line 108 uses 'dim', which is inconsistent.
    """
    import inspect

    source = inspect.getsource(objectives.VanillaObjective.forward)

    # Check that the source doesn't use 'axis' parameter
    # This test will FAIL until the bug is fixed
    assert "axis=" not in source, (
        "VanillaObjective.forward() uses non-canonical 'axis' parameter. "
        "Should use 'dim' instead for consistency with PyTorch conventions."
    )


def test_matryoshka_objective_parameter_consistency():
    """
    Test that MatryoshkaObjective uses consistent parameter names internally.

    Currently lines 174-175 use 'axis' instead of 'dim'.
    """
    import inspect

    source = inspect.getsource(objectives.MatryoshkaObjective.forward)

    # Check that the source doesn't use 'axis' parameter
    # This test will FAIL until the bug is fixed
    assert "axis=" not in source, (
        "MatryoshkaObjective.forward() uses non-canonical 'axis' parameter. "
        "Should use 'dim' instead for consistency with PyTorch conventions."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
