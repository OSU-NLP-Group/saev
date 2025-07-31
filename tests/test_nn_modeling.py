import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from scipy.stats import kstest, pareto

from saev.nn import modeling


def test_factories():
    assert isinstance(modeling.get_activation(modeling.Relu()), torch.nn.ReLU)
    assert isinstance(modeling.get_activation(modeling.TopK()), modeling.TopKActivation)
    assert isinstance(
        modeling.get_activation(modeling.BatchTopK()), modeling.BatchTopKActivation
    )


def relu_cfgs():
    return st.builds(
        modeling.Relu,
        d_vit=st.sampled_from([32, 64, 128]),
        exp_factor=st.sampled_from([2, 4]),
    )


def topk_cfgs():
    return st.builds(
        modeling.TopK,
        d_vit=st.sampled_from([32, 64, 128]),
        exp_factor=st.sampled_from([2, 4]),
        top_k=st.sampled_from([1, 2, 4, 8]),
    )


def batch_topk_cfgs():
    return st.builds(
        modeling.BatchTopK,
        d_vit=st.sampled_from([32, 64, 128]),
        exp_factor=st.sampled_from([2, 4]),
        top_k=st.sampled_from([1, 2, 4, 8]),
    )


@given(cfg=topk_cfgs(), batch=st.integers(min_value=1, max_value=4))
def test_topk_activation(cfg, batch):
    act = modeling.get_activation(cfg)
    x = torch.randn(batch, cfg.d_vit * cfg.exp_factor)
    y = act(x)
    assert y.shape == (batch, cfg.d_vit * cfg.exp_factor)
    # Check that only k elements are non-zero per sample
    assert (y != 0).sum(dim=1).eq(cfg.top_k).all()


@given(cfg=batch_topk_cfgs(), batch=st.integers(min_value=1, max_value=4))
def test_batch_topk_activation(cfg, batch):
    act = modeling.get_activation(cfg)
    x = torch.randn(batch, cfg.d_vit * cfg.exp_factor)
    y = act(x)
    assert y.shape == (batch, cfg.d_vit * cfg.exp_factor)
    # Check that only k elements are non-zero per batch
    assert (y != 0).sum(dim=1).sum(dim=0).eq(cfg.top_k)


def test_topk_basic_forward():
    """Test basic TopK forward pass with known values."""
    cfg = modeling.TopK(d_vit=4, exp_factor=1, top_k=2)
    act = modeling.TopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0, 2.0], [2.0, 4.0, 1.0, 3.0]])
    y = act(x)

    expected = torch.tensor([[5.0, 0.0, 3.0, 0.0], [0.0, 4.0, 0.0, 3.0]])
    torch.testing.assert_close(y, expected)


def test_topk_ties():
    """Test TopK behavior with tied values."""
    cfg = modeling.TopK(d_vit=4, exp_factor=1, top_k=2)
    act = modeling.TopKActivation(cfg)

    x = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    y = act(x)

    # Should select first k elements in case of ties
    assert (y != 0).sum() == 2
    # Verify the selected values are correct
    assert y[y != 0].unique().item() == 2.0


def test_topk_k_equals_size():
    """Test TopK when k equals tensor size."""
    cfg = modeling.TopK(d_vit=4, exp_factor=1, top_k=4)
    act = modeling.TopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0, 2.0]])
    y = act(x)

    # All values should be preserved
    torch.testing.assert_close(y, x)


def test_topk_negative_values():
    """Test TopK with negative values."""
    cfg = modeling.TopK(d_vit=4, exp_factor=1, top_k=2)
    act = modeling.TopKActivation(cfg)

    x = torch.tensor([[-5.0, -1.0, -3.0, -2.0]])
    y = act(x)

    # Should select -1.0 and -2.0 (largest values)
    expected = torch.tensor([[0.0, -1.0, 0.0, -2.0]])
    torch.testing.assert_close(y, expected)


def test_topk_gradient_flow():
    """Test that gradients flow correctly through TopK."""
    cfg = modeling.TopK(d_vit=4, exp_factor=1, top_k=2)
    act = modeling.TopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0, 2.0], [2.0, 4.0, 1.0, 3.0]], requires_grad=True)
    y = act(x)

    # Create a simple loss (sum of outputs)
    loss = y.sum()
    loss.backward()

    # Expected gradient: 1.0 for selected elements, 0.0 for others
    expected_grad = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    torch.testing.assert_close(x.grad, expected_grad)


def test_topk_gradient_sparsity():
    """Verify gradient sparsity matches forward pass selection."""
    cfg = modeling.TopK(d_vit=8, exp_factor=1, top_k=3)
    act = modeling.TopKActivation(cfg)

    torch.manual_seed(42)
    x = torch.randn(2, 8, requires_grad=True)
    y = act(x)

    # Use a different upstream gradient
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    # Check that gradient sparsity matches forward pass
    forward_mask = (y != 0).float()
    grad_mask = (x.grad != 0).float()
    torch.testing.assert_close(forward_mask, grad_mask)

    # Verify gradient values for selected elements
    selected_grads = x.grad * forward_mask
    expected_grads = grad_output * forward_mask
    torch.testing.assert_close(selected_grads, expected_grads)


def test_topk_zero_gradient_for_unselected():
    """Explicitly verify that non-selected elements have exactly 0.0 gradients."""
    cfg = modeling.TopK(d_vit=6, exp_factor=1, top_k=2)
    act = modeling.TopKActivation(cfg)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], requires_grad=True)
    y = act(x)

    loss = y.sum()
    loss.backward()

    # Elements at indices 0, 1, 2, 3 should have zero gradients
    torch.testing.assert_close(x.grad[0, 0], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[0, 1], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[0, 2], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[0, 3], torch.tensor(0.0))
    # Elements at indices 4, 5 should have non-zero gradients
    torch.testing.assert_close(x.grad[0, 4], torch.tensor(1.0))
    torch.testing.assert_close(x.grad[0, 5], torch.tensor(1.0))


# BatchTopK Edge Case Tests
def test_batchtopk_basic_forward():
    """Test basic BatchTopK forward pass with known values."""
    cfg = modeling.BatchTopK(d_vit=3, exp_factor=1, top_k=3)
    act = modeling.BatchTopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0], [2.0, 4.0, 1.0]])
    y = act(x)

    # Top 3 values globally are 5.0, 4.0, 3.0
    expected = torch.tensor([[5.0, 0.0, 3.0], [0.0, 4.0, 0.0]])
    torch.testing.assert_close(y, expected)


def test_batchtopk_k_exceeds_total_elements():
    """Test BatchTopK when k exceeds total number of elements."""
    cfg = modeling.BatchTopK(d_vit=3, exp_factor=1, top_k=8)
    act = modeling.BatchTopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0], [2.0, 4.0, 1.0]])
    y = act(x)

    # All elements should be preserved when k > total elements
    torch.testing.assert_close(y, x)


def test_batchtopk_single_batch():
    """Test BatchTopK with single element batch."""
    cfg = modeling.BatchTopK(d_vit=4, exp_factor=1, top_k=2)
    act = modeling.BatchTopKActivation(cfg)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = act(x)

    # Should behave like regular TopK for single batch
    expected = torch.tensor([[0.0, 0.0, 3.0, 4.0]])
    torch.testing.assert_close(y, expected)


def test_batchtopk_uneven_distribution():
    """Test BatchTopK with uneven value distribution across batch."""
    cfg = modeling.BatchTopK(d_vit=3, exp_factor=1, top_k=3)
    act = modeling.BatchTopKActivation(cfg)

    # First batch has large values, second has small values
    x = torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    y = act(x)

    # All top-k should come from first batch
    expected = torch.tensor([[10.0, 20.0, 30.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(y, expected)


def test_batchtopk_ties():
    """Test BatchTopK behavior with tied values."""
    cfg = modeling.BatchTopK(d_vit=3, exp_factor=1, top_k=3)
    act = modeling.BatchTopKActivation(cfg)

    x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    y = act(x)

    # Should select k elements (tie-breaking based on flattened indices)
    assert (y != 0).sum() == 3
    assert y[y != 0].unique().item() == 2.0


# BatchTopK Gradient Tests
def test_batchtopk_gradient_flow():
    """Test that gradients flow correctly through BatchTopK."""
    cfg = modeling.BatchTopK(d_vit=3, exp_factor=1, top_k=3)
    act = modeling.BatchTopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0], [2.0, 4.0, 1.0]], requires_grad=True)
    y = act(x)

    # Create a simple loss (sum of outputs)
    loss = y.sum()
    loss.backward()

    # Expected gradient: 1.0 for selected elements (5.0, 4.0, 3.0), 0.0 for others
    expected_grad = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    torch.testing.assert_close(x.grad, expected_grad)


def test_batchtopk_gradient_global_sparsity():
    """Verify gradient sparsity is global across batch."""
    cfg = modeling.BatchTopK(d_vit=4, exp_factor=1, top_k=4)
    act = modeling.BatchTopKActivation(cfg)

    torch.manual_seed(42)
    x = torch.randn(3, 4, requires_grad=True)  # 12 total elements
    y = act(x)

    # Use a different upstream gradient
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    # Check that exactly k gradients are non-zero globally
    assert (x.grad != 0).sum() == 4

    # Check that gradient sparsity matches forward pass
    forward_mask = (y != 0).float()
    grad_mask = (x.grad != 0).float()
    torch.testing.assert_close(forward_mask, grad_mask)

    # Verify gradient values for selected elements
    selected_grads = x.grad * forward_mask
    expected_grads = grad_output * forward_mask
    torch.testing.assert_close(selected_grads, expected_grads)


def test_batchtopk_gradient_distribution():
    """Test gradient distribution with uneven value distribution."""
    cfg = modeling.BatchTopK(d_vit=3, exp_factor=1, top_k=3)
    act = modeling.BatchTopKActivation(cfg)

    # First batch has large values, second has small values
    x = torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], requires_grad=True)
    y = act(x)

    # Custom upstream gradient
    grad_output = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    y.backward(grad_output)

    # All gradients should flow to first batch only
    expected_grad = torch.tensor([[2.0, 3.0, 4.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(x.grad, expected_grad)


def test_batchtopk_zero_gradient_verification():
    """Explicitly verify BatchTopK zero gradients for unselected elements."""
    cfg = modeling.BatchTopK(d_vit=2, exp_factor=1, top_k=2)
    act = modeling.BatchTopKActivation(cfg)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    y = act(x)

    loss = y.sum()
    loss.backward()

    # Only the last row should have gradients (contains 5.0 and 6.0)
    torch.testing.assert_close(x.grad[0, 0], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[0, 1], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[1, 0], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[1, 1], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[2, 0], torch.tensor(1.0))
    torch.testing.assert_close(x.grad[2, 1], torch.tensor(1.0))


# Hypothesis-based Property Tests
@settings(deadline=None)
@given(cfg=topk_cfgs(), batch=st.integers(min_value=1, max_value=8))
def test_topk_gradient_properties(cfg, batch):
    """Property-based test for TopK gradients."""
    act = modeling.TopKActivation(cfg)

    torch.manual_seed(42)
    x = torch.randn(batch, cfg.d_sae, requires_grad=True)
    y = act(x)

    # Create random upstream gradient
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    # Property 1: Gradient sparsity matches forward pass
    forward_mask = y != 0
    grad_mask = x.grad != 0
    assert torch.equal(forward_mask, grad_mask), (
        "Gradient sparsity doesn't match forward pass"
    )

    # Property 2: Exactly k non-zero gradients per sample
    assert (x.grad != 0).sum(dim=1).eq(cfg.top_k).all(), (
        "Wrong number of non-zero gradients per sample"
    )

    # Property 3: Non-selected elements have exactly zero gradient
    assert (x.grad[~forward_mask] == 0).all(), (
        "Non-selected elements have non-zero gradients"
    )

    # Property 4: Selected elements have gradient equal to upstream gradient
    torch.testing.assert_close(x.grad[forward_mask], grad_output[forward_mask])


@settings(deadline=None)
@given(cfg=batch_topk_cfgs(), batch=st.integers(min_value=1, max_value=8))
def test_batchtopk_gradient_properties(cfg, batch):
    """Property-based test for BatchTopK gradients."""
    act = modeling.BatchTopKActivation(cfg)

    torch.manual_seed(42)
    x = torch.randn(batch, cfg.d_sae, requires_grad=True)
    y = act(x)

    # Skip if k > total elements (edge case handled separately)
    total_elements = batch * cfg.d_sae
    if cfg.top_k > total_elements:
        return

    # Create random upstream gradient
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    # Property 1: Gradient sparsity matches forward pass
    forward_mask = y != 0
    grad_mask = x.grad != 0
    assert torch.equal(forward_mask, grad_mask), (
        "Gradient sparsity doesn't match forward pass"
    )

    # Property 2: Exactly k non-zero gradients globally
    assert (x.grad != 0).sum() == cfg.top_k, (
        f"Expected {cfg.top_k} non-zero gradients, got {(x.grad != 0).sum()}"
    )

    # Property 3: Non-selected elements have exactly zero gradient
    assert (x.grad[~forward_mask] == 0).all(), (
        "Non-selected elements have non-zero gradients"
    )

    # Property 4: Selected elements have gradient equal to upstream gradient
    torch.testing.assert_close(x.grad[forward_mask], grad_output[forward_mask])


# Chain Rule and Advanced Gradient Tests
def test_topk_chain_rule():
    """Test gradient flow through TopK in a deeper network."""
    cfg = modeling.TopK(d_vit=4, exp_factor=1, top_k=2)
    act = modeling.TopKActivation(cfg)

    # Build a simple network: Linear -> TopK -> Linear
    linear1 = torch.nn.Linear(4, 4)
    linear2 = torch.nn.Linear(4, 2)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

    # Forward pass
    h1 = linear1(x)
    h2 = act(h1)
    output = linear2(h2)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Verify gradients exist and flow correctly
    assert x.grad is not None
    assert linear1.weight.grad is not None
    assert linear2.weight.grad is not None

    # Verify sparsity pattern propagates through the network
    # Get which elements were selected in TopK
    selected_mask = h2 != 0

    # linear2.weight.grad should only have non-zero gradients for selected features
    # Shape: linear2.weight is (2, 4), we expect gradients only for selected columns
    for i in range(4):
        if not selected_mask[0, i]:
            assert (linear2.weight.grad[:, i] == 0).all()


def test_batchtopk_chain_rule():
    """Test gradient flow through BatchTopK in a deeper network."""
    cfg = modeling.BatchTopK(d_vit=3, exp_factor=1, top_k=3)
    act = modeling.BatchTopKActivation(cfg)

    # Build a simple network
    linear1 = torch.nn.Linear(3, 3)
    linear2 = torch.nn.Linear(3, 1)

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

    # Forward pass
    h1 = linear1(x)
    h2 = act(h1)
    output = linear2(h2)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Verify gradients exist
    assert x.grad is not None
    assert linear1.weight.grad is not None
    assert linear2.weight.grad is not None

    # Verify global sparsity pattern
    assert (h2 != 0).sum() == 3


def test_topk_non_differentiable_selection():
    """Verify that TopK gradient is not differentiable w.r.t. the selection boundary."""
    cfg = modeling.TopK(d_vit=4, exp_factor=1, top_k=2)
    act = modeling.TopKActivation(cfg)

    # Test that gradients exist for first order
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    y = act(x)
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]

    # Verify the gradient pattern (should be sparse)
    expected_grad = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    torch.testing.assert_close(grad, expected_grad)

    # TopK is piecewise constant in its gradient, so second derivatives
    # don't exist in the traditional sense (the selection is non-differentiable)


def test_gradient_magnitude_preservation():
    """Test that gradient magnitudes are preserved for selected elements."""
    cfg = modeling.TopK(d_vit=4, exp_factor=1, top_k=2)
    act = modeling.TopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0, 2.0]], requires_grad=True)
    y = act(x)

    # Use custom gradient with known magnitudes
    custom_grad = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    y.backward(custom_grad)

    # Check that selected elements preserve gradient magnitude
    # Elements at indices 0 and 2 should be selected (values 5.0 and 3.0)
    torch.testing.assert_close(
        x.grad[0, 0], torch.tensor(2.0)
    )  # Gradient for value 5.0
    torch.testing.assert_close(
        x.grad[0, 2], torch.tensor(4.0)
    )  # Gradient for value 3.0
    torch.testing.assert_close(x.grad[0, 1], torch.tensor(0.0))  # Not selected
    torch.testing.assert_close(x.grad[0, 3], torch.tensor(0.0))  # Not selected


@settings(deadline=None)
@given(cfg=relu_cfgs(), batch=st.integers(min_value=1, max_value=4))
def test_sae_shapes(cfg, batch):
    sae = modeling.SparseAutoencoder(cfg)
    x = torch.randn(batch, cfg.d_vit)
    x_hat, f = sae(x)
    assert x_hat.shape == (batch, cfg.d_vit)
    assert f.shape == (batch, cfg.d_sae)


hf_ckpts = [
    "osunlp/SAE_BioCLIP_24K_ViT-B-16_iNat21",
    "osunlp/SAE_CLIP_24K_ViT-B-16_IN1K",
    "osunlp/SAE_DINOv2_24K_ViT-B-14_IN1K",
]


@pytest.mark.parametrize("repo_id", hf_ckpts)
@pytest.mark.slow
def test_load_bioclip_checkpoint(repo_id, tmp_path):
    pytest.importorskip("huggingface_hub")

    import huggingface_hub

    ckpt_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename="sae.pt", cache_dir=tmp_path
    )

    model = modeling.load(ckpt_path)

    # Smoke-test shapes & numerics
    x = torch.randn(2, model.cfg.d_vit)
    x_hat, f_x = model(x)
    assert x_hat.shape == x.shape
    assert f_x.shape[1] == model.cfg.d_sae
    # reconstruction shouldn’t be exactly identical, but should have finite values
    assert torch.isfinite(x_hat).all()


roundtrip_cases = [
    modeling.Relu(d_vit=512, exp_factor=8, seed=0),
    modeling.Relu(d_vit=768, exp_factor=16, seed=1),
    modeling.Relu(d_vit=1024, exp_factor=32, seed=2),
]


@pytest.mark.parametrize("sae_cfg", roundtrip_cases)
def test_dump_load_roundtrip(tmp_path, sae_cfg):
    """Write → load → verify state-dict & cfg equality."""
    sae = modeling.SparseAutoencoder(sae_cfg)
    _ = sae(torch.randn(2, sae_cfg.d_vit))  # touch all params once

    ckpt = tmp_path / "sae.pt"
    modeling.dump(str(ckpt), sae)
    sae_loaded = modeling.load(str(ckpt))

    # configs identical
    assert sae_cfg == sae_loaded.cfg

    # tensors identical
    for k, v in sae.state_dict().items():
        torch.testing.assert_close(v, sae_loaded.state_dict()[k])


@given(cfg=relu_cfgs(), n=st.integers(20, 100), pareto_power=st.floats(0.1, 0.8))
@settings(deadline=None)
def test_sample_prefix(cfg, n, pareto_power):
    """Check to make sure that the prefix sampling follows a pareto dist."""
    sae = modeling.MatryoshkaSparseAutoencoder(cfg)
    assert isinstance(sae, modeling.MatryoshkaSparseAutoencoder)

    prefixes = sae.sample_prefixes(
        sae.cfg.d_sae, n, pareto_power=pareto_power, replacement=True
    )

    # Test against pareto distribution with the same params
    # cdf is scaled according to the max value of the cdf (at d_sae)
    # we also need to discretize the distribution since we are sampling integers.
    # Testing this using the equation is a bit tricky, so we sample integers instead.
    def pareto_cdf(x):
        return pareto.cdf(x, b=pareto_power) / pareto.cdf(sae.cfg.d_sae, b=pareto_power)

    pareto_pdf = np.diff(
        np.concatenate(([0], [pareto_cdf(i) for i in range(1, sae.cfg.d_sae + 1)]))
    )

    pareto_prefixes = np.random.choice(
        np.arange(1, sae.cfg.d_sae), size=n, replace=True, p=pareto_pdf[1:]
    )

    statistic, p_value = kstest(prefixes, pareto_prefixes)

    assert p_value > 0.01
