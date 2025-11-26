import typing as tp

import hypothesis
import hypothesis.strategies as st
import hypothesis_torch
import pytest
import torch

import saev.nn
from saev.nn import activations, objectives


def test_mse_same():
    x = torch.ones((45, 12), dtype=torch.float)
    x_hat = torch.ones((45, 12), dtype=torch.float)
    expected = torch.zeros((45, 12), dtype=torch.float)
    actual = objectives.mean_squared_err(x_hat, x)
    torch.testing.assert_close(actual, expected)


def test_mse_zero_x_hat():
    x = torch.ones((3, 2), dtype=torch.float)
    x_hat = torch.zeros((3, 2), dtype=torch.float)
    expected = torch.ones((3, 2), dtype=torch.float)
    actual = objectives.mean_squared_err(x_hat, x, norm=False)
    torch.testing.assert_close(actual, expected)


def test_mse_nonzero():
    x = torch.full((3, 2), 3, dtype=torch.float)
    x_hat = torch.ones((3, 2), dtype=torch.float)
    expected = objectives.ref_mean_squared_err(x_hat, x)
    actual = objectives.mean_squared_err(x_hat, x)
    torch.testing.assert_close(actual, expected)


def test_safe_mse_large_x():
    x = torch.full((3, 2), 3e28, dtype=torch.float)
    x_hat = torch.ones((3, 2), dtype=torch.float)

    ref = objectives.ref_mean_squared_err(x_hat, x, norm=True)
    assert ref.isnan().any()

    safe = objectives.mean_squared_err(x_hat, x, norm=True)
    assert not safe.isnan().any()


def test_mse_norm_keyword_works():
    x = torch.randn(3, 4)
    y = x.clone()
    m = objectives.mean_squared_err(y, x, norm=True)
    assert torch.isfinite(m).all()


def test_factories():
    assert isinstance(
        objectives.get_objective(objectives.Matryoshka()),
        objectives.MatryoshkaObjective,
    )


# basic element generator
finite32 = st.floats(
    min_value=-1e9,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)

tensor123 = hypothesis_torch.tensor_strategy(
    dtype=torch.float32,
    shape=(1, 2, 3),
    elements=finite32,
    layout=torch.strided,
    device=torch.device("cpu"),
)


@st.composite
def tensor_pair(draw):
    x_hat = draw(tensor123)
    x = draw(tensor123)
    # ensure denominator in your safe-mse is not zero
    hypothesis.assume(torch.linalg.norm(x, ord=2, dim=-1).max() > 1e-8)
    return x_hat, x


@pytest.mark.slow
@pytest.mark.xfail(reason="Hypothesis can always find something bad")
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow], deadline=None
)
@hypothesis.given(pair=tensor_pair())
def test_safe_mse_hypothesis(pair):
    x_hat, x = pair  # both finite, same device/layout
    expected = objectives.ref_mean_squared_err(x_hat, x)
    actual = objectives.mean_squared_err(x_hat, x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_matryoshka_objective_new_api():
    """Test MatryoshkaObjective with new API accepting (sae, x)."""
    from saev import nn

    # Create config and models
    sae_cfg = nn.SparseAutoencoderConfig(d_model=64, d_sae=128)
    obj_cfg = objectives.Matryoshka(n_prefixes=5)

    sae = nn.SparseAutoencoder(sae_cfg)  # Note: using regular SAE now
    objective = objectives.get_objective(obj_cfg)

    # Create test data
    x = torch.randn(8, 64)  # batch=8, d_model=64

    # Test new API
    loss = objective(sae, x)

    # Verify loss has expected attributes
    assert hasattr(loss, "mse")
    assert hasattr(loss, "sparsity")
    assert hasattr(loss, "l0")
    assert hasattr(loss, "l1")
    assert hasattr(loss, "loss")

    # Verify shapes
    assert loss.mse.shape == torch.Size([])
    assert loss.sparsity.shape == torch.Size([])
    assert loss.l0.shape == torch.Size([])
    assert loss.l1.shape == torch.Size([])

    # Verify loss is finite
    assert torch.isfinite(loss.loss)

    # Test backward pass
    loss.loss.backward()

    # Verify gradients exist
    assert sae.W_enc.grad is not None
    assert sae.W_dec.grad is not None
    assert sae.b_enc.grad is not None
    assert sae.b_dec.grad is not None


def test_matryoshka_prefix_sampling():
    """Test prefix sampling in MatryoshkaObjective."""

    # Test prefix sampling
    d_sae = 256
    prefixes = objectives.sample_prefixes(d_sae, n_prefixes=10)

    # Check properties
    assert len(prefixes) == 10
    assert prefixes[-1] == d_sae  # Last prefix should be full dimension
    assert all(
        prefixes[i] <= prefixes[i + 1] for i in range(len(prefixes) - 1)
    )  # Sorted
    assert all(1 <= p <= d_sae for p in prefixes)  # Valid range


def test_matryoshka_prefix_masking():
    """Test that Matryoshka prefix masking works correctly."""

    # Create config and models
    sae_cfg = saev.nn.SparseAutoencoderConfig(d_model=64, d_sae=256)
    sae = saev.nn.SparseAutoencoder(sae_cfg)

    # Create test data
    x = torch.randn(4, 64)  # batch=4, d_model=64
    f_x = sae.encode(x)

    # Test that masking works correctly
    prefix_sizes = [8, 16, 32, 64, 128, 256]  # d_sae = 64*4 = 256

    for prefix_size in prefix_sizes:
        mask = torch.zeros_like(f_x)
        mask[:, :prefix_size] = 1.0
        masked_f_x = f_x * mask

        # Check that only first prefix_size elements are non-zero
        assert (masked_f_x[:, :prefix_size] == f_x[:, :prefix_size]).all()
        assert (masked_f_x[:, prefix_size:] == 0).all()

        # Check that decode doesn't crash
        x_hat = sae.decode(masked_f_x)
        assert x_hat[:, 0, :].shape == x.shape
        assert torch.isfinite(x_hat).all()


def test_objective_uses_standard_sae():
    """Test that objectives work with standard SparseAutoencoder (no special subclass needed)."""

    sae_cfg = saev.nn.SparseAutoencoderConfig(d_model=32, d_sae=64)
    sae = saev.nn.SparseAutoencoder(sae_cfg)  # Standard SAE

    # Test with both objective types
    matryoshka_obj = objectives.get_objective(objectives.Matryoshka(n_prefixes=3))

    x = torch.randn(4, 32)

    # Should work with standard SAE
    matryoshka_loss = matryoshka_obj(sae, x)
    assert isinstance(matryoshka_loss, objectives.MatryoshkaLoss)


def test_matryoshka_different_prefix_counts():
    """Test MatryoshkaObjective with different n_prefixes values."""

    sae_cfg = saev.nn.SparseAutoencoderConfig(d_model=32, d_sae=128)
    sae = saev.nn.SparseAutoencoder(sae_cfg)
    x = torch.randn(4, 32)

    for n_prefixes in [1, 5, 10, 20]:
        obj_cfg = objectives.Matryoshka(n_prefixes=n_prefixes)
        objective = objectives.get_objective(obj_cfg)

        loss = objective(sae, x)
        assert torch.isfinite(loss.loss)

        # Check that prefix sampling returns correct number
        prefixes = objectives.sample_prefixes(sae.cfg.d_sae, n_prefixes)
        assert len(prefixes) == max(1, n_prefixes)


def test_metrics_methods():
    """Test that Loss.metrics() returns expected dictionary."""

    sae_cfg = saev.nn.SparseAutoencoderConfig(d_model=32, d_sae=64)
    sae = saev.nn.SparseAutoencoder(sae_cfg)
    x = torch.randn(4, 32)

    # Test MatryoshkaLoss metrics
    matryoshka_obj = objectives.get_objective(objectives.Matryoshka())
    matryoshka_loss = matryoshka_obj(sae, x)
    metrics = matryoshka_loss.metrics()

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "mse" in metrics
    assert "l0" in metrics
    assert "l1" in metrics
    assert "sparsity" in metrics


@pytest.mark.parametrize("cfg", [cls() for cls in tp.get_args(activations.Config)])
def test_objectives_with_different_activations(cfg):
    """Test that objectives work with different activation functions."""
    from saev import nn

    sae_cfg = nn.SparseAutoencoderConfig(d_model=32, d_sae=64, activation=cfg)
    sae = nn.SparseAutoencoder(sae_cfg)
    x = torch.randn(4, 32)

    # Test with both objectives
    matryoshka_obj = objectives.get_objective(objectives.Matryoshka(n_prefixes=3))

    matryoshka_loss = matryoshka_obj(sae, x)
    assert torch.isfinite(matryoshka_loss.loss)


def test_matryoshka_edge_cases():
    """Test edge cases for MatryoshkaObjective."""

    # Test with very small d_sae
    sae_cfg = saev.nn.SparseAutoencoderConfig(d_model=8, d_sae=8)  # d_sae = 8
    sae = saev.nn.SparseAutoencoder(sae_cfg)
    x = torch.randn(2, 8)

    # Test with n_prefixes == d_sae
    obj = objectives.get_objective(objectives.Matryoshka(n_prefixes=8))
    loss = obj(sae, x)
    assert torch.isfinite(loss.loss)

    # Test with n_prefixes = 1 (should work like vanilla)
    obj_single = objectives.get_objective(objectives.Matryoshka(n_prefixes=1))
    loss_single = obj_single(sae, x)
    assert torch.isfinite(loss_single.loss)


def test_gradient_flow_through_matryoshka():
    """Test that gradients flow correctly through Matryoshka's multiple decode ops."""

    sae_cfg = saev.nn.SparseAutoencoderConfig(d_model=32, d_sae=64)
    sae = saev.nn.SparseAutoencoder(sae_cfg)
    obj = objectives.get_objective(objectives.Matryoshka(n_prefixes=5))

    x = torch.randn(4, 32, requires_grad=True)

    # Forward and backward
    loss = obj(sae, x)
    loss.loss.backward()

    # Check gradients exist for all parameters
    for name, param in sae.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        assert (param.grad != 0).any(), f"Zero gradient for {name}"


def test_unified_decode_matryoshka():
    """Test that decode with prefixes returns cumulative reconstructions."""

    sae_cfg = saev.nn.SparseAutoencoderConfig(d_model=32, d_sae=128)  # d_sae=128
    sae = saev.nn.SparseAutoencoder(sae_cfg)

    f_x = torch.randn(4, 128)  # batch=4, d_sae=128
    prefixes = torch.tensor([32, 64, 128])

    # Matryoshka decode
    x_hats = sae.decode(f_x, prefixes=prefixes)

    assert x_hats.shape == (4, 3, 32)
    assert torch.isfinite(x_hats).all()


def test_decode_prefixes_device_handling():
    """Test that decode handles device placement correctly."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    sae_cfg = saev.nn.SparseAutoencoderConfig(d_model=16, d_sae=32)
    sae = saev.nn.SparseAutoencoder(sae_cfg).cuda()

    f_x = torch.randn(2, 32).cuda()
    prefixes = torch.tensor([8, 16, 32])  # CPU tensor

    # Should handle CPU prefixes with CUDA model/data
    x_hats = sae.decode(f_x, prefixes=prefixes)

    assert x_hats.device == f_x.device
    assert x_hats.shape == (3, 2, 16)


def test_decode_default_prefix_is_long_and_does_not_crash():
    cfg = saev.nn.SparseAutoencoderConfig(d_model=8, d_sae=8)
    sae = saev.nn.SparseAutoencoder(cfg)
    x = torch.randn(2, 8)
    f = sae.encode(x)
    out = sae.decode(f)  # should not raise
    assert out.shape == (2, 1, 8)
