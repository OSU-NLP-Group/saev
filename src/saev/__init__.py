"""
saev is a Python package for training sparse autoencoders (SAEs) on vision transformers (ViTs) in PyTorch.
"""

import importlib.metadata
import pathlib
import tomllib  # std-lib in Python ≥3.11


def _version_from_pyproject() -> str:
    """
    Parse `[project].version` out of pyproject.toml that sits two directories above this file:
        saev/__init__.py
        saev/
        pyproject.toml

    Returns "0.0.0+unknown" on any error.
    """
    try:
        pp = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pp.open("rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except Exception:  # key missing, file missing, bad TOML, ...
        return "0.0.0+unknown"


try:
    __version__ = importlib.metadata.version("saev")  # installed wheel / editable
except importlib.metadata.PackageNotFoundError:
    __version__ = _version_from_pyproject()  # running from source tree
