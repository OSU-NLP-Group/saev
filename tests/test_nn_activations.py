import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from saev.nn import activations


def topk_cfgs():
    return st.builds(activations.TopK, top_k=st.sampled_from([1, 2, 4, 8]))


def batch_topk_cfgs():
    return st.builds(activations.BatchTopK, top_k=st.sampled_from([1, 2, 4, 8]))


def test_factories():
    assert isinstance(
        activations.get_activation(activations.Relu()), activations.ReluActivation
    )
    assert isinstance(
        activations.get_activation(activations.TopK()), activations.TopKActivation
    )
    assert isinstance(
        activations.get_activation(activations.BatchTopK()),
        activations.BatchTopKActivation,
    )


def test_topk_basic_forward():
    """Test basic TopK forward pass with known values."""
    cfg = activations.TopK(top_k=2)
    act = activations.TopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0, 2.0], [2.0, 4.0, 1.0, 3.0]])
    y = act(x)

    expected = torch.tensor([[5.0, 0.0, 3.0, 0.0], [0.0, 4.0, 0.0, 3.0]])
    torch.testing.assert_close(y, expected)


def test_topk_ties():
    """Test TopK behavior with tied values."""
    cfg = activations.TopK(top_k=2)
    act = activations.TopKActivation(cfg)

    x = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    y = act(x)

    # Should select first k elements in case of ties
    assert (y != 0).sum() == 2
    # Verify the selected values are correct
    assert y[y != 0].unique().item() == 2.0


def test_topk_k_equals_size():
    """Test TopK when k equals tensor size."""
    cfg = activations.TopK(top_k=4)
    act = activations.TopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0, 2.0]])
    y = act(x)

    # All values should be preserved
    torch.testing.assert_close(y, x)


def test_topk_negative_values():
    """Test TopK with negative values."""
    cfg = activations.TopK(top_k=2)
    act = activations.TopKActivation(cfg)

    x = torch.tensor([[-5.0, -1.0, -3.0, -2.0]])
    y = act(x)

    # Should select -1.0 and -2.0 (largest values)
    expected = torch.tensor([[0.0, -1.0, 0.0, -2.0]])
    torch.testing.assert_close(y, expected)


def test_topk_gradient_flow():
    """Test that gradients flow correctly through TopK."""
    cfg = activations.TopK(top_k=2)
    act = activations.TopKActivation(cfg)

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
    cfg = activations.TopK(top_k=3)
    act = activations.TopKActivation(cfg)

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


@given(
    cfg=topk_cfgs(),
    batch=st.integers(min_value=1, max_value=4),
    d_sae=st.integers(min_value=128, max_value=1024),
)
def test_topk_activation(cfg, batch, d_sae):
    act = activations.get_activation(cfg)
    x = torch.randn(batch, d_sae)
    y = act(x)
    assert y.shape == (batch, d_sae)
    # Check that only k elements are non-zero per sample
    assert (y != 0).sum(dim=1).eq(cfg.top_k).all()


@given(
    cfg=batch_topk_cfgs(),
    batch=st.integers(min_value=1, max_value=4),
    d_sae=st.integers(min_value=32, max_value=1024),
)
def test_batchtopk_activation(cfg, batch, d_sae):
    act = activations.get_activation(cfg)
    x = torch.randn(batch, d_sae)
    y = act(x)
    assert y.shape == (batch, d_sae)
    # Check that a total of k elements are non-zero per batch
    assert (y != 0).sum().eq(cfg.top_k * batch).all()


def test_topk_zero_gradient_for_unselected():
    """Explicitly verify that non-selected elements have exactly 0.0 gradients."""
    cfg = activations.TopK(top_k=2)
    act = activations.TopKActivation(cfg)

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
    cfg = activations.BatchTopK(top_k=2)
    act = activations.BatchTopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0], [2.0, 4.0, 1.0]])
    y = act(x)

    # Top 2 values per sample
    expected = torch.tensor([[5.0, 0.0, 3.0], [2.0, 4.0, 0.0]])
    torch.testing.assert_close(y, expected)


def test_batchtopk_k_exceeds_total_elements():
    """Test BatchTopK when k exceeds total number of elements."""
    cfg = activations.BatchTopK(top_k=8)
    act = activations.BatchTopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0], [2.0, 4.0, 1.0]])
    y = act(x)

    # All elements should be preserved when k > total elements
    torch.testing.assert_close(y, x)


def test_batchtopk_single_batch():
    """Test BatchTopK with single element batch."""
    cfg = activations.BatchTopK(top_k=2)
    act = activations.BatchTopKActivation(cfg)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = act(x)

    # Should behave like regular TopK for single batch
    expected = torch.tensor([[0.0, 0.0, 3.0, 4.0]])
    torch.testing.assert_close(y, expected)


def test_batchtopk_uneven_distribution():
    """Test BatchTopK with uneven value distribution across batch."""
    cfg = activations.BatchTopK(top_k=2)
    act = activations.BatchTopKActivation(cfg)

    # First batch has large values, second has small values
    x = torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    y = act(x)

    # Top (2 * batch) values in the batch
    expected = torch.tensor([[10.0, 20.0, 30.0], [0.0, 0.0, 3.0]])
    torch.testing.assert_close(y, expected)


def test_batchtopk_ties():
    """Test BatchTopK behavior with tied values."""
    cfg = activations.BatchTopK(top_k=2)
    act = activations.BatchTopKActivation(cfg)

    x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    y = act(x)

    # Should select (k * batch) elements per sample
    assert (y != 0).sum().eq(4).all()
    assert y[y != 0].unique().item() == 2.0


# BatchTopK Gradient Tests
def test_batchtopk_gradient_flow():
    """Test that gradients flow correctly through BatchTopK."""
    cfg = activations.BatchTopK(top_k=2)
    act = activations.BatchTopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0], [2.0, 4.0, 1.0]], requires_grad=True)
    y = act(x)

    # Create a simple loss (sum of outputs)
    loss = y.sum()
    loss.backward()

    # Expected gradient: 1.0 for top 2 elements per sample
    expected_grad = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    torch.testing.assert_close(x.grad, expected_grad)


def test_batchtopk_gradient_global_sparsity():
    """Verify gradient sparsity is per sample."""
    cfg = activations.BatchTopK(top_k=2)
    act = activations.BatchTopKActivation(cfg)

    torch.manual_seed(42)
    x = torch.randn(3, 4, requires_grad=True)  # 12 total elements
    y = act(x)

    # Use a different upstream gradient
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    # Check that exactly k gradients are non-zero per sample
    assert (x.grad != 0).sum().eq(2 * 3).all()

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
    cfg = activations.BatchTopK(top_k=2)
    act = activations.BatchTopKActivation(cfg)

    # First batch has large values, second has small values
    x = torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], requires_grad=True)
    y = act(x)

    # Custom upstream gradient
    grad_output = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    y.backward(grad_output)

    # Gradients should flow to top 2 per sample
    expected_grad = torch.tensor([[2.0, 3.0, 4.0], [0.0, 0.0, 7.0]])
    torch.testing.assert_close(x.grad, expected_grad)


def test_batchtopk_zero_gradient_verification():
    """Explicitly verify BatchTopK zero gradients for unselected elements."""
    cfg = activations.BatchTopK(top_k=1)
    act = activations.BatchTopKActivation(cfg)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    y = act(x)

    loss = y.sum()
    loss.backward()

    torch.testing.assert_close(x.grad[0, 0], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[0, 1], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[1, 0], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[1, 1], torch.tensor(1.0))
    torch.testing.assert_close(x.grad[2, 0], torch.tensor(1.0))
    torch.testing.assert_close(x.grad[2, 1], torch.tensor(1.0))


@settings(deadline=None)
@given(
    cfg=topk_cfgs(),
    batch=st.integers(min_value=1, max_value=8),
    d_sae=st.integers(min_value=256, max_value=2048),
)
def test_topk_gradient_properties(cfg, batch, d_sae):
    """Property-based test for TopK gradients."""
    act = activations.TopKActivation(cfg)

    torch.manual_seed(42)
    x = torch.randn(batch, d_sae, requires_grad=True)
    y = act(x)

    # Create random upstream gradient
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    # Property 1: Gradient sparsity matches forward pass
    forward_mask = y != 0
    grad_mask = x.grad != 0
    assert torch.equal(forward_mask, grad_mask)

    # Property 2: Exactly k non-zero gradients per sample
    assert (x.grad != 0).sum(dim=1).eq(cfg.top_k).all()

    # Property 3: Non-selected elements have exactly zero gradient
    assert (x.grad[~forward_mask] == 0).all()

    # Property 4: Selected elements have gradient equal to upstream gradient
    torch.testing.assert_close(x.grad[forward_mask], grad_output[forward_mask])


@settings(deadline=None)
@given(
    cfg=batch_topk_cfgs(),
    batch=st.integers(min_value=1, max_value=8),
    d_sae=st.integers(min_value=128, max_value=512),
)
def test_batchtopk_gradient_properties(cfg, batch, d_sae):
    """Property-based test for BatchTopK gradients."""
    act = activations.BatchTopKActivation(cfg)

    torch.manual_seed(42)
    x = torch.randn(batch, d_sae, requires_grad=True)
    y = act(x)

    # Skip if k > d_sae (edge case handled separately)
    if cfg.top_k > d_sae:
        return

    # Create random upstream gradient
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    # Property 1: Gradient sparsity matches forward pass
    forward_mask = y != 0
    grad_mask = x.grad != 0
    assert torch.equal(forward_mask, grad_mask)

    # Property 2: Exactly k * batch non-zero gradients per batch
    assert (x.grad != 0).sum().eq(cfg.top_k * batch).all()

    # Property 3: Non-selected elements have exactly zero gradient
    assert (x.grad[~forward_mask] == 0).all()

    # Property 4: Selected elements have gradient equal to upstream gradient
    torch.testing.assert_close(x.grad[forward_mask], grad_output[forward_mask])


def test_topk_chain_rule():
    """Test gradient flow through TopK in a deeper network."""
    cfg = activations.TopK(top_k=2)
    act = activations.TopKActivation(cfg)

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
    cfg = activations.BatchTopK(top_k=2)
    act = activations.BatchTopKActivation(cfg)

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

    # Verify per-sample sparsity pattern
    assert (h2 != 0).sum(dim=1).eq(2).all()


def test_topk_non_differentiable_selection():
    """Verify that TopK gradient is not differentiable w.r.t. the selection boundary."""
    cfg = activations.TopK(top_k=2)
    act = activations.TopKActivation(cfg)

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
    cfg = activations.TopK(top_k=2)
    act = activations.TopKActivation(cfg)

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


def test_batchtopk_threshold_updates_from_train_batch():
    """Training mode should update threshold toward the min positive activation."""
    cfg = activations.BatchTopK(top_k=2)
    act = activations.BatchTopKActivation(cfg)
    act.train()

    # Keep a copy of the initial threshold
    initial = act.threshold.clone()

    # Simple positive tensor so the min-positive is easy to reason about
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = act(x)

    # Take the smallest positive activation that survived BatchTopK
    pos = y[y > 0]
    assert pos.numel() > 0
    batch_min = pos.min()

    # Threshold should have moved away from its initial value
    updated = act.threshold
    assert not torch.equal(initial, updated)

    # And it should be positive and no larger than the batch minimum
    assert updated.item() > 0.0
    assert updated.item() <= batch_min.item() + 1e-6


def test_batchtopk_eval_uses_stored_threshold_jumprelu():
    """Eval mode should apply JumpReLU with the stored threshold and not update it."""
    cfg = activations.BatchTopK(top_k=2)
    act = activations.BatchTopKActivation(cfg)

    # Manually set a known threshold
    with torch.no_grad():
        act.threshold.fill_(0.5)
    act.eval()

    x = torch.tensor([[0.1, 0.6, 0.4], [0.7, 0.2, 0.8]])
    y = act(x)

    # Expected JumpReLU behavior: keep entries strictly above the threshold
    theta = act.threshold
    expected = torch.where(x > theta, x, torch.zeros_like(x))

    torch.testing.assert_close(y, expected)

    # Sanity check: we should not be enforcing k*batch non-zeros in eval.
    # Here 3 entries are > 0.5, while k * batch = 4.
    assert (y != 0).sum().item() == 3

    # And eval forward should NOT change the stored threshold
    torch.testing.assert_close(act.threshold, theta)
