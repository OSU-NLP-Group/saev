"""Test for decode shape bug in test_nn_objectives.py"""

import pytest
import torch

import saev.nn


def test_decode_returns_batch_first():
    """Test that decode returns (batch, n_prefixes, d_model) not (n_prefixes, batch, d_model).

    This is a regression test for a bug in test_decode_prefixes_device_handling where
    the shape assertion was incorrect: it expected (3, 2, 16) but should be (2, 3, 16).
    """
    sae_cfg = saev.nn.SparseAutoencoderConfig(d_model=16, exp_factor=2)
    sae = saev.nn.SparseAutoencoder(sae_cfg)

    batch_size = 2
    d_sae = 32
    f_x = torch.randn(batch_size, d_sae)
    prefixes = torch.tensor([8, 16, 32])

    x_hats = sae.decode(f_x, prefixes=prefixes)

    # Should be (batch, n_prefixes, d_model)
    expected_shape = (batch_size, len(prefixes), sae_cfg.d_model)

    # This will fail if the bug exists
    assert x_hats.shape == expected_shape, f"Expected shape {expected_shape}, got {x_hats.shape}"


def test_wrong_shape_assertion_fails():
    """Demonstrate that the incorrect shape assertion from test_decode_prefixes_device_handling is wrong.

    The original test has:
        f_x = torch.randn(2, 32)
        prefixes = torch.tensor([8, 16, 32])
        x_hats = sae.decode(f_x, prefixes=prefixes)
        assert x_hats.shape == (3, 2, 16)  # BUG: should be (2, 3, 16)
    """
    sae_cfg = saev.nn.SparseAutoencoderConfig(d_model=16, exp_factor=2)
    sae = saev.nn.SparseAutoencoder(sae_cfg)

    f_x = torch.randn(2, 32)
    prefixes = torch.tensor([8, 16, 32])

    x_hats = sae.decode(f_x, prefixes=prefixes)

    # The buggy assertion from the original test
    wrong_shape = (3, 2, 16)

    # This should fail because the shape is actually (2, 3, 16)
    with pytest.raises(AssertionError):
        assert x_hats.shape == wrong_shape

    # The correct assertion
    assert x_hats.shape == (2, 3, 16)
