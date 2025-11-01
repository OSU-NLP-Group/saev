import itertools
import typing as tp

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given, settings

from saev.nn import modeling


def test_factories():
    assert isinstance(modeling.get_activation(modeling.Relu()), torch.nn.ReLU)
    assert isinstance(modeling.get_activation(modeling.TopK()), modeling.TopKActivation)
    assert isinstance(
        modeling.get_activation(modeling.BatchTopK()), modeling.BatchTopKActivation
    )


def sae_cfgs():
    return st.builds(
        lambda d_model, expansion: modeling.SparseAutoencoderConfig(
            d_model=d_model,
            d_sae=d_model * expansion,
        ),
        d_model=st.sampled_from([32, 64, 128]),
        expansion=st.sampled_from([2, 4]),
    )


def sae_cfgs_comprehensive():
    """Comprehensive SAE config strategy for testing various combinations."""
    # Define activation strategies
    relu_strategy = st.builds(modeling.Relu)
    topk_strategy = st.builds(
        modeling.TopK, top_k=st.sampled_from([8, 16, 32, 64, 128])
    )
    batch_topk_strategy = st.builds(
        modeling.BatchTopK, top_k=st.sampled_from([8, 16, 32, 64, 128])
    )

    activation_strategy = st.one_of(relu_strategy, topk_strategy, batch_topk_strategy)

    return st.builds(
        lambda d_model,
        expansion,
        seed,
        normalize_w_dec,
        remove_parallel_grads,
        n_reinit_samples,
        activation: modeling.SparseAutoencoderConfig(
            d_model=d_model,
            d_sae=d_model * expansion,
            seed=seed,
            normalize_w_dec=normalize_w_dec,
            remove_parallel_grads=remove_parallel_grads,
            n_reinit_samples=n_reinit_samples,
            activation=activation,
        ),
        d_model=st.sampled_from([256, 384, 512, 768, 1024]),
        expansion=st.sampled_from([2, 4, 8, 16, 32]),
        seed=st.integers(min_value=0, max_value=100),
        normalize_w_dec=st.booleans(),
        remove_parallel_grads=st.booleans(),
        n_reinit_samples=st.sampled_from([1024, 1024 * 16, 1024 * 16 * 32]),
        activation=activation_strategy,
    )


def topk_cfgs():
    return st.builds(modeling.TopK, top_k=st.sampled_from([1, 2, 4, 8]))


def batch_topk_cfgs():
    return st.builds(modeling.BatchTopK, top_k=st.sampled_from([1, 2, 4, 8]))


@given(
    cfg=topk_cfgs(),
    batch=st.integers(min_value=1, max_value=4),
    d_sae=st.integers(min_value=256, max_value=2048),
)
def test_topk_activation(cfg, batch, d_sae):
    act = modeling.get_activation(cfg)
    x = torch.randn(batch, d_sae)
    y = act(x)
    assert y.shape == (batch, d_sae)
    # Check that only k elements are non-zero per sample
    assert (y != 0).sum(dim=1).eq(cfg.top_k).all()


@given(
    cfg=batch_topk_cfgs(),
    batch=st.integers(min_value=1, max_value=4),
    d_sae=st.integers(min_value=256, max_value=2048),
)
def test_batch_topk_activation(cfg, batch, d_sae):
    act = modeling.get_activation(cfg)
    x = torch.randn(batch, d_sae)
    y = act(x)
    assert y.shape == (batch, d_sae)
    # Check that only k elements are non-zero per sample
    assert (y != 0).sum(dim=1).eq(cfg.top_k).all()


def test_topk_basic_forward():
    """Test basic TopK forward pass with known values."""
    cfg = modeling.TopK(top_k=2)
    act = modeling.TopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0, 2.0], [2.0, 4.0, 1.0, 3.0]])
    y = act(x)

    expected = torch.tensor([[5.0, 0.0, 3.0, 0.0], [0.0, 4.0, 0.0, 3.0]])
    torch.testing.assert_close(y, expected)


def test_topk_ties():
    """Test TopK behavior with tied values."""
    cfg = modeling.TopK(top_k=2)
    act = modeling.TopKActivation(cfg)

    x = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    y = act(x)

    # Should select first k elements in case of ties
    assert (y != 0).sum() == 2
    # Verify the selected values are correct
    assert y[y != 0].unique().item() == 2.0


def test_topk_k_equals_size():
    """Test TopK when k equals tensor size."""
    cfg = modeling.TopK(top_k=4)
    act = modeling.TopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0, 2.0]])
    y = act(x)

    # All values should be preserved
    torch.testing.assert_close(y, x)


def test_topk_negative_values():
    """Test TopK with negative values."""
    cfg = modeling.TopK(top_k=2)
    act = modeling.TopKActivation(cfg)

    x = torch.tensor([[-5.0, -1.0, -3.0, -2.0]])
    y = act(x)

    # Should select -1.0 and -2.0 (largest values)
    expected = torch.tensor([[0.0, -1.0, 0.0, -2.0]])
    torch.testing.assert_close(y, expected)


def test_topk_gradient_flow():
    """Test that gradients flow correctly through TopK."""
    cfg = modeling.TopK(top_k=2)
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
    cfg = modeling.TopK(top_k=3)
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
    cfg = modeling.TopK(top_k=2)
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
    cfg = modeling.BatchTopK(top_k=2)
    act = modeling.BatchTopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0], [2.0, 4.0, 1.0]])
    y = act(x)

    # Top 2 values per sample
    expected = torch.tensor([[5.0, 0.0, 3.0], [2.0, 4.0, 0.0]])
    torch.testing.assert_close(y, expected)


def test_batchtopk_k_exceeds_total_elements():
    """Test BatchTopK when k exceeds total number of elements."""
    cfg = modeling.BatchTopK(top_k=8)
    act = modeling.BatchTopKActivation(cfg)

    x = torch.tensor([[5.0, 1.0, 3.0], [2.0, 4.0, 1.0]])
    y = act(x)

    # All elements should be preserved when k > total elements
    torch.testing.assert_close(y, x)


def test_batchtopk_single_batch():
    """Test BatchTopK with single element batch."""
    cfg = modeling.BatchTopK(top_k=2)
    act = modeling.BatchTopKActivation(cfg)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = act(x)

    # Should behave like regular TopK for single batch
    expected = torch.tensor([[0.0, 0.0, 3.0, 4.0]])
    torch.testing.assert_close(y, expected)


def test_batchtopk_uneven_distribution():
    """Test BatchTopK with uneven value distribution across batch."""
    cfg = modeling.BatchTopK(top_k=2)
    act = modeling.BatchTopKActivation(cfg)

    # First batch has large values, second has small values
    x = torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    y = act(x)

    # Top 2 values per sample
    expected = torch.tensor([[0.0, 20.0, 30.0], [0.0, 2.0, 3.0]])
    torch.testing.assert_close(y, expected)


def test_batchtopk_ties():
    """Test BatchTopK behavior with tied values."""
    cfg = modeling.BatchTopK(top_k=2)
    act = modeling.BatchTopKActivation(cfg)

    x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    y = act(x)

    # Should select k elements per sample
    assert (y != 0).sum(dim=1).eq(2).all()
    assert y[y != 0].unique().item() == 2.0


# BatchTopK Gradient Tests
def test_batchtopk_gradient_flow():
    """Test that gradients flow correctly through BatchTopK."""
    cfg = modeling.BatchTopK(top_k=2)
    act = modeling.BatchTopKActivation(cfg)

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
    cfg = modeling.BatchTopK(top_k=2)
    act = modeling.BatchTopKActivation(cfg)

    torch.manual_seed(42)
    x = torch.randn(3, 4, requires_grad=True)  # 12 total elements
    y = act(x)

    # Use a different upstream gradient
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    # Check that exactly k gradients are non-zero per sample
    assert (x.grad != 0).sum(dim=1).eq(2).all()

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
    cfg = modeling.BatchTopK(top_k=2)
    act = modeling.BatchTopKActivation(cfg)

    # First batch has large values, second has small values
    x = torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], requires_grad=True)
    y = act(x)

    # Custom upstream gradient
    grad_output = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    y.backward(grad_output)

    # Gradients should flow to top 2 per sample
    expected_grad = torch.tensor([[0.0, 3.0, 4.0], [0.0, 6.0, 7.0]])
    torch.testing.assert_close(x.grad, expected_grad)


def test_batchtopk_zero_gradient_verification():
    """Explicitly verify BatchTopK zero gradients for unselected elements."""
    cfg = modeling.BatchTopK(top_k=1)
    act = modeling.BatchTopKActivation(cfg)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    y = act(x)

    loss = y.sum()
    loss.backward()

    # Only the highest value per row should have gradients
    torch.testing.assert_close(x.grad[0, 0], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[0, 1], torch.tensor(1.0))
    torch.testing.assert_close(x.grad[1, 0], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[1, 1], torch.tensor(1.0))
    torch.testing.assert_close(x.grad[2, 0], torch.tensor(0.0))
    torch.testing.assert_close(x.grad[2, 1], torch.tensor(1.0))


@settings(deadline=None)
@given(
    cfg=topk_cfgs(),
    batch=st.integers(min_value=1, max_value=8),
    d_sae=st.integers(min_value=256, max_value=2048),
)
def test_topk_gradient_properties(cfg, batch, d_sae):
    """Property-based test for TopK gradients."""
    act = modeling.TopKActivation(cfg)

    torch.manual_seed(42)
    x = torch.randn(batch, d_sae, requires_grad=True)
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
@given(
    cfg=batch_topk_cfgs(),
    batch=st.integers(min_value=1, max_value=8),
    d_sae=st.integers(min_value=256, max_value=2048),
)
def test_batchtopk_gradient_properties(cfg, batch, d_sae):
    """Property-based test for BatchTopK gradients."""
    act = modeling.BatchTopKActivation(cfg)

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
    assert torch.equal(forward_mask, grad_mask), (
        "Gradient sparsity doesn't match forward pass"
    )

    # Property 2: Exactly k non-zero gradients per sample
    assert (x.grad != 0).sum(dim=1).eq(cfg.top_k).all(), (
        f"Expected {cfg.top_k} non-zero gradients per sample"
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
    cfg = modeling.TopK(top_k=2)
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
    cfg = modeling.BatchTopK(top_k=2)
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

    # Verify per-sample sparsity pattern
    assert (h2 != 0).sum(dim=1).eq(2).all()


def test_topk_non_differentiable_selection():
    """Verify that TopK gradient is not differentiable w.r.t. the selection boundary."""
    cfg = modeling.TopK(top_k=2)
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
    cfg = modeling.TopK(top_k=2)
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
@given(cfg=sae_cfgs(), batch=st.integers(min_value=1, max_value=4))
def test_sae_shapes(cfg, batch):
    sae = modeling.SparseAutoencoder(cfg)
    x = torch.randn(batch, cfg.d_model)
    x_hat, f = sae(x)
    assert x_hat.shape == (batch, 1, cfg.d_model)
    assert f.shape == (batch, cfg.d_sae)


hf_ckpts = [
    "osunlp/SAE_BioCLIP_24K_ViT-B-16_iNat21",
    "osunlp/SAE_CLIP_24K_ViT-B-16_IN1K",
    "osunlp/SAE_DINOv2_24K_ViT-B-14_IN1K",
]


@pytest.mark.parametrize("repo_id", hf_ckpts)
@pytest.mark.slow
def test_load_existing_checkpoint(repo_id, tmp_path):
    pytest.importorskip("huggingface_hub")

    import huggingface_hub

    ckpt_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename="sae.pt", cache_dir=tmp_path
    )

    model = modeling.load(ckpt_path)

    # Smoke-test shapes & numerics
    x = torch.randn(2, model.cfg.d_model)
    x_hat, f_x = model(x)
    assert x_hat.shape == x[:, None, :].shape
    assert f_x.shape[1] == model.cfg.d_sae
    # reconstruction shouldn’t be exactly identical, but should have finite values
    assert torch.isfinite(x_hat).all()


def test_dump_load_roundtrip_exhaustive(tmp_path):
    """Test dump/load roundtrip for all combinations of activations and various configs."""
    # Test all activation types with different configurations
    activation_cfgs = [cls() for cls in tp.get_args(modeling.ActivationConfig)]

    # Various SAE configurations - reduced set for faster testing
    sae_cfgs = [
        {"d_model": 256, "d_sae": 1024, "seed": 0},
        {"d_model": 512, "d_sae": 4096, "seed": 1, "normalize_w_dec": False},
        {"d_model": 384, "d_sae": 4608, "seed": 2, "remove_parallel_grads": False},
    ]

    for i, (act_cfg, cfg_args) in enumerate(
        itertools.product(activation_cfgs, sae_cfgs)
    ):
        sae_cfg = modeling.SparseAutoencoderConfig(**cfg_args, activation=act_cfg)
        sae = modeling.SparseAutoencoder(sae_cfg)
        _ = sae(torch.randn(2, sae_cfg.d_model))  # touch all params once

        ckpt = tmp_path / f"sae_{i}.pt"
        modeling.dump(ckpt, sae)
        sae_loaded = modeling.load(ckpt)

        # configs identical
        assert sae_cfg == sae_loaded.cfg

        # tensors identical
        for k, v in sae.state_dict().items():
            torch.testing.assert_close(v, sae_loaded.state_dict()[k])


@pytest.mark.parametrize(
    "sae_cfg",
    [
        modeling.SparseAutoencoderConfig(d_model=512, d_sae=4096, seed=0),
        modeling.SparseAutoencoderConfig(d_model=768, d_sae=12288, seed=1),
        modeling.SparseAutoencoderConfig(d_model=1024, d_sae=32768, seed=2),
    ],
)
def test_dump_load_roundtrip_simple(tmp_path, sae_cfg):
    """Write → load → verify state-dict & cfg equality."""
    sae = modeling.SparseAutoencoder(sae_cfg)
    _ = sae(torch.randn(2, sae_cfg.d_model))  # touch all params once

    ckpt = tmp_path / "sae.pt"
    modeling.dump(ckpt, sae)
    sae_loaded = modeling.load(ckpt)

    # configs identical
    assert sae_cfg == sae_loaded.cfg

    # tensors identical
    for k, v in sae.state_dict().items():
        torch.testing.assert_close(v, sae_loaded.state_dict()[k])


@given(sae_cfg=sae_cfgs_comprehensive())
@settings(deadline=None, max_examples=10)
def test_dump_load_roundtrip_hypothesis(sae_cfg):
    """Property-based test for dump/load roundtrip with random configurations."""
    import tempfile

    # Create SAE and test roundtrip
    sae = modeling.SparseAutoencoder(sae_cfg)
    _ = sae(torch.randn(2, sae_cfg.d_model))  # touch all params once

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt = f"{tmp_dir}/sae.pt"
        modeling.dump(ckpt, sae)
        sae_loaded = modeling.load(ckpt)

        # configs identical
        assert sae_cfg == sae_loaded.cfg

        # tensors identical
        for k, v in sae.state_dict().items():
            torch.testing.assert_close(v, sae_loaded.state_dict()[k])


def test_load_local_checkpoint(request):
    """Test loading a checkpoint from a local path specified via --ckpt-path."""
    ckpt_path = request.config.getoption("--ckpt-path")
    if ckpt_path is None:
        pytest.skip("No checkpoint path provided. Use --ckpt-path to specify one.")

    # Test loading the checkpoint
    model = modeling.load(ckpt_path)

    # Basic smoke tests
    assert isinstance(model, modeling.SparseAutoencoder)
    assert hasattr(model, "cfg")
    assert isinstance(model.cfg, modeling.SparseAutoencoderConfig)

    # Test forward pass
    x = torch.randn(2, model.cfg.d_model)
    x_hat, f_x = model(x)

    # Check shapes
    assert x_hat.shape == x.shape
    assert f_x.shape == (2, model.cfg.d_sae)

    # Check that outputs are finite
    assert torch.isfinite(x_hat).all()
    assert torch.isfinite(f_x).all()

    # Print some info about the loaded checkpoint
    print(f"\nLoaded checkpoint from: {ckpt_path}")
    print(f"Model config: d_model={model.cfg.d_model}, d_sae={model.cfg.d_sae}")
    print(f"Activation: {type(model.cfg.activation).__name__}")
    if hasattr(model.cfg.activation, "top_k"):
        print(f"Top-k value: {model.cfg.activation.top_k}")


def test_remove_parallel_grads_handles_non_normalized_rows():
    cfg = modeling.SparseAutoencoderConfig(
        d_model=4, d_sae=4, normalize_w_dec=False, remove_parallel_grads=True
    )
    # Disable automatic normalization but keep removal on
    sae = modeling.SparseAutoencoder(cfg)

    # Artificial W_dec and grad
    with torch.no_grad():
        sae.W_dec.copy_(torch.randn_like(sae.W_dec))
    sae.W_dec.grad = torch.randn_like(sae.W_dec)

    # After removal, each row grad should be orthogonal to its row of W_dec
    sae.remove_parallel_grads()
    row_dots = (sae.W_dec.grad * sae.W_dec).sum(dim=1)
    assert torch.allclose(row_dots, torch.zeros_like(row_dots), atol=1e-6)
