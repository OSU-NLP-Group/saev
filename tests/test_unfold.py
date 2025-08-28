import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from saev.data import transforms


def test_smoke():
    torch.manual_seed(0)

    B, C, H, W, k, D = 2, 3, 12, 10, 2, 7
    x = torch.randn(B, C, H, W)
    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)
    with torch.no_grad():
        conv.weight.normal_()
        conv.bias.normal_()

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.shape == y_unf.shape

    np.testing.assert_allclose(y_conv.detach(), y_unf.detach(), rtol=1e-6, atol=1e-5)


def test_forward_equivalence_single_channel_2x2_kernel():
    torch.manual_seed(42)
    B, C, H, W, k, D = 1, 1, 4, 4, 2, 3
    x = torch.randn(B, C, H, W)
    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.shape == y_unf.shape == (B, 4, D)
    np.testing.assert_allclose(y_conv.detach(), y_unf.detach(), rtol=1e-6, atol=1e-6)


def test_forward_equivalence_multichannel_3x3_kernel():
    torch.manual_seed(123)
    B, C, H, W, k, D = 2, 5, 9, 9, 3, 8
    x = torch.randn(B, C, H, W)
    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.shape == y_unf.shape == (B, 9, D)
    np.testing.assert_allclose(y_conv.detach(), y_unf.detach(), rtol=1e-6, atol=1e-6)


def test_forward_equivalence_large_input_16x16_patches():
    torch.manual_seed(456)
    B, C, H, W, k, D = 4, 3, 224, 224, 16, 768
    x = torch.randn(B, C, H, W)
    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.shape == y_unf.shape == (B, 196, D)
    np.testing.assert_allclose(y_conv.detach(), y_unf.detach(), rtol=1e-5, atol=1e-5)


def test_forward_equivalence_rectangular_input():
    torch.manual_seed(789)
    B, C, H, W, k, D = 3, 2, 6, 8, 2, 10
    x = torch.randn(B, C, H, W)
    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.shape == y_unf.shape == (B, 12, D)
    np.testing.assert_allclose(y_conv.detach(), y_unf.detach(), rtol=1e-6, atol=1e-6)


def test_forward_equivalence_no_bias():
    torch.manual_seed(111)
    B, C, H, W, k, D = 2, 4, 8, 8, 4, 5
    x = torch.randn(B, C, H, W)
    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=False)

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.shape == y_unf.shape == (B, 4, D)
    np.testing.assert_allclose(y_conv.detach(), y_unf.detach(), rtol=1e-6, atol=1e-6)


def test_gradient_equivalence_simple():
    torch.manual_seed(222)
    B, C, H, W, k, D = 1, 2, 4, 4, 2, 3

    x1 = torch.randn(B, C, H, W, requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)

    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    y_conv = transforms.conv2d_to_tokens(x1, conv)
    y_unf = transforms.unfolded_conv2d(x2, conv)

    loss_conv = y_conv.sum()
    loss_unf = y_unf.sum()

    loss_conv.backward()
    loss_unf.backward()

    np.testing.assert_allclose(x1.grad.detach(), x2.grad.detach(), rtol=1e-6, atol=1e-6)


def test_gradient_equivalence_with_downstream_operations():
    torch.manual_seed(333)
    B, C, H, W, k, D = 2, 3, 6, 6, 3, 8

    x1 = torch.randn(B, C, H, W, requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)

    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    y_conv = transforms.conv2d_to_tokens(x1, conv)
    y_unf = transforms.unfolded_conv2d(x2, conv)

    target = torch.randn_like(y_conv)
    loss_conv = ((y_conv - target) ** 2).mean()
    loss_unf = ((y_unf - target) ** 2).mean()

    loss_conv.backward()
    loss_unf.backward()

    np.testing.assert_allclose(x1.grad.detach(), x2.grad.detach(), rtol=1e-6, atol=1e-6)


def test_gradient_equivalence_weight_gradients():
    torch.manual_seed(444)
    B, C, H, W, k, D = 2, 4, 8, 8, 4, 16

    x = torch.randn(B, C, H, W)

    conv1 = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)
    conv2 = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    with torch.no_grad():
        conv2.weight.copy_(conv1.weight)
        conv2.bias.copy_(conv1.bias)

    y_conv = transforms.conv2d_to_tokens(x, conv1)
    y_unf = transforms.unfolded_conv2d(x, conv2)

    loss_conv = y_conv.sum()
    loss_unf = y_unf.sum()

    loss_conv.backward()
    loss_unf.backward()

    np.testing.assert_allclose(
        conv1.weight.grad.detach(), conv2.weight.grad.detach(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        conv1.bias.grad.detach(), conv2.bias.grad.detach(), rtol=1e-6, atol=1e-6
    )


def test_batch_processing_consistency():
    torch.manual_seed(555)
    C, H, W, k, D = 3, 12, 12, 4, 32

    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    x_single = torch.randn(1, C, H, W)
    y_conv_single = transforms.conv2d_to_tokens(x_single, conv)
    y_unf_single = transforms.unfolded_conv2d(x_single, conv)

    x_batch = torch.cat([x_single] * 5, dim=0)
    y_conv_batch = transforms.conv2d_to_tokens(x_batch, conv)
    y_unf_batch = transforms.unfolded_conv2d(x_batch, conv)

    for i in range(5):
        np.testing.assert_allclose(
            y_conv_single.detach(),
            y_conv_batch[i : i + 1].detach(),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            y_unf_single.detach(), y_unf_batch[i : i + 1].detach(), rtol=1e-6, atol=1e-6
        )


def test_zero_input_produces_bias():
    torch.manual_seed(666)
    B, C, H, W, k, D = 2, 3, 6, 6, 2, 5

    x = torch.zeros(B, C, H, W)
    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    expected_shape = (B, 9, D)
    assert y_conv.shape == y_unf.shape == expected_shape

    expected_output = conv.bias.view(1, 1, D).expand(B, 9, D)
    np.testing.assert_allclose(
        y_conv.detach(), expected_output.detach(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        y_unf.detach(), expected_output.detach(), rtol=1e-6, atol=1e-6
    )


def test_identity_convolution():
    torch.manual_seed(777)
    B, H, W, k = 2, 8, 8, 2
    C = D = k * k

    x = torch.randn(B, C, H, W)
    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=False)

    with torch.no_grad():
        conv.weight.zero_()
        conv.weight.view(D, -1).fill_diagonal_(1.0)

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.shape == y_unf.shape
    np.testing.assert_allclose(y_conv.detach(), y_unf.detach(), rtol=1e-6, atol=1e-6)


def test_extreme_dimensions_tiny():
    torch.manual_seed(888)
    B, C, H, W, k, D = 1, 1, 2, 2, 1, 1
    x = torch.randn(B, C, H, W)
    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.shape == y_unf.shape == (B, 4, D)
    np.testing.assert_allclose(y_conv.detach(), y_unf.detach(), rtol=1e-6, atol=1e-6)


def test_extreme_dimensions_large_channels():
    torch.manual_seed(999)
    B, C, H, W, k, D = 1, 128, 8, 8, 4, 256
    x = torch.randn(B, C, H, W)
    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.shape == y_unf.shape == (B, 4, D)
    np.testing.assert_allclose(y_conv.detach(), y_unf.detach(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kernel_size", [1, 2, 4, 7, 14])
def test_various_kernel_sizes(kernel_size):
    torch.manual_seed(1234)
    B, C = 2, 3
    H = W = kernel_size * 4
    D = 10

    x = torch.randn(B, C, H, W)
    conv = nn.Conv2d(
        C, D, kernel_size=kernel_size, stride=kernel_size, padding=0, bias=True
    )

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    expected_tokens = (H // kernel_size) * (W // kernel_size)
    assert y_conv.shape == y_unf.shape == (B, expected_tokens, D)
    np.testing.assert_allclose(y_conv.detach(), y_unf.detach(), rtol=1e-5, atol=1e-5)


def test_gradient_accumulation():
    torch.manual_seed(4321)
    B, C, H, W, k, D = 1, 2, 4, 4, 2, 3

    x1 = torch.randn(B, C, H, W, requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)

    conv = nn.Conv2d(C, D, kernel_size=k, stride=k, padding=0, bias=True)

    for _ in range(3):
        y_conv = transforms.conv2d_to_tokens(x1, conv)
        y_unf = transforms.unfolded_conv2d(x2, conv)

        loss_conv = y_conv.sum()
        loss_unf = y_unf.sum()

        loss_conv.backward(retain_graph=True)
        loss_unf.backward(retain_graph=True)

    np.testing.assert_allclose(x1.grad.detach(), x2.grad.detach(), rtol=1e-6, atol=1e-6)


def test_double_precision():
    torch.manual_seed(5678)
    B, C, H, W, k, D = 1, 2, 4, 4, 2, 3

    x = torch.randn(B, C, H, W, dtype=torch.float64)
    conv = nn.Conv2d(
        C, D, kernel_size=k, stride=k, padding=0, bias=True, dtype=torch.float64
    )

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.dtype == y_unf.dtype == torch.float64
    np.testing.assert_allclose(y_conv.detach(), y_unf.detach(), rtol=1e-10, atol=1e-10)


@st.composite
def conv_params(draw):
    """Generate valid parameters for a Conv2d layer that satisfies our constraints."""
    kernel_size = draw(st.integers(min_value=1, max_value=8))
    in_channels = draw(st.integers(min_value=1, max_value=16))
    out_channels = draw(st.integers(min_value=1, max_value=32))
    bias = draw(st.booleans())

    batch_size = draw(st.integers(min_value=1, max_value=4))
    n_patches_h = draw(st.integers(min_value=1, max_value=8))
    n_patches_w = draw(st.integers(min_value=1, max_value=8))

    height = n_patches_h * kernel_size
    width = n_patches_w * kernel_size

    dtype = draw(st.sampled_from([torch.float32, torch.float64]))

    return {
        "batch_size": batch_size,
        "in_channels": in_channels,
        "height": height,
        "width": width,
        "kernel_size": kernel_size,
        "out_channels": out_channels,
        "bias": bias,
        "dtype": dtype,
        "n_patches": n_patches_h * n_patches_w,
    }


@given(params=conv_params())
@settings(max_examples=20, deadline=None)
def test_hypothesis_forward_equivalence(params):
    """Property: unfolded_conv2d and conv2d_to_tokens produce identical outputs."""
    torch.manual_seed(42)

    x = torch.randn(
        params["batch_size"],
        params["in_channels"],
        params["height"],
        params["width"],
        dtype=params["dtype"],
    )

    conv = nn.Conv2d(
        params["in_channels"],
        params["out_channels"],
        kernel_size=params["kernel_size"],
        stride=params["kernel_size"],
        padding=0,
        bias=params["bias"],
        dtype=params["dtype"],
    )

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert (
        y_conv.shape
        == y_unf.shape
        == (params["batch_size"], params["n_patches"], params["out_channels"])
    )
    assert y_conv.dtype == y_unf.dtype == params["dtype"]

    if params["dtype"] == torch.float64:
        np.testing.assert_allclose(
            y_conv.detach().numpy(), y_unf.detach().numpy(), rtol=1e-10, atol=1e-10
        )
    else:
        np.testing.assert_allclose(
            y_conv.detach().numpy(), y_unf.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(params=conv_params())
@settings(max_examples=10, deadline=None)
def test_hypothesis_gradient_equivalence(params):
    """Property: gradients computed through both methods are identical."""
    torch.manual_seed(42)

    x1 = torch.randn(
        params["batch_size"],
        params["in_channels"],
        params["height"],
        params["width"],
        dtype=params["dtype"],
        requires_grad=True,
    )
    x2 = x1.clone().detach().requires_grad_(True)

    conv = nn.Conv2d(
        params["in_channels"],
        params["out_channels"],
        kernel_size=params["kernel_size"],
        stride=params["kernel_size"],
        padding=0,
        bias=params["bias"],
        dtype=params["dtype"],
    )

    y_conv = transforms.conv2d_to_tokens(x1, conv)
    y_unf = transforms.unfolded_conv2d(x2, conv)

    loss_conv = y_conv.sum()
    loss_unf = y_unf.sum()

    loss_conv.backward()
    loss_unf.backward()

    assert x1.grad.dtype == x2.grad.dtype == params["dtype"]

    if params["dtype"] == torch.float64:
        np.testing.assert_allclose(
            x1.grad.detach().numpy(), x2.grad.detach().numpy(), rtol=1e-10, atol=1e-10
        )
    else:
        np.testing.assert_allclose(
            x1.grad.detach().numpy(), x2.grad.detach().numpy(), rtol=1e-5, atol=1e-5
        )


@given(params=conv_params())
@settings(max_examples=10, deadline=None)
def test_hypothesis_weight_gradient_equivalence(params):
    """Property: weight and bias gradients are identical through both methods."""
    torch.manual_seed(42)

    x = torch.randn(
        params["batch_size"],
        params["in_channels"],
        params["height"],
        params["width"],
        dtype=params["dtype"],
    )

    conv1 = nn.Conv2d(
        params["in_channels"],
        params["out_channels"],
        kernel_size=params["kernel_size"],
        stride=params["kernel_size"],
        padding=0,
        bias=params["bias"],
        dtype=params["dtype"],
    )

    conv2 = nn.Conv2d(
        params["in_channels"],
        params["out_channels"],
        kernel_size=params["kernel_size"],
        stride=params["kernel_size"],
        padding=0,
        bias=params["bias"],
        dtype=params["dtype"],
    )

    with torch.no_grad():
        conv2.weight.copy_(conv1.weight)
        if params["bias"]:
            conv2.bias.copy_(conv1.bias)

    y_conv = transforms.conv2d_to_tokens(x, conv1)
    y_unf = transforms.unfolded_conv2d(x, conv2)

    loss_conv = y_conv.sum()
    loss_unf = y_unf.sum()

    loss_conv.backward()
    loss_unf.backward()

    assert conv1.weight.grad.dtype == conv2.weight.grad.dtype == params["dtype"]

    if params["dtype"] == torch.float64:
        rtol, atol = 1e-10, 1e-10
    else:
        rtol, atol = 1e-5, 1e-5

    np.testing.assert_allclose(
        conv1.weight.grad.detach().numpy(),
        conv2.weight.grad.detach().numpy(),
        rtol=rtol,
        atol=atol,
    )

    if params["bias"]:
        assert conv1.bias.grad.dtype == conv2.bias.grad.dtype == params["dtype"]
        np.testing.assert_allclose(
            conv1.bias.grad.detach().numpy(),
            conv2.bias.grad.detach().numpy(),
            rtol=rtol,
            atol=atol,
        )


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    in_channels=st.integers(min_value=1, max_value=8),
    kernel_size=st.integers(min_value=1, max_value=4),
    n_patches=st.integers(min_value=1, max_value=4),
    out_channels=st.integers(min_value=1, max_value=16),
    element_strategy=st.one_of(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
        st.just(0.0),
        st.just(1.0),
        st.just(-1.0),
    ),
)
@settings(max_examples=10, deadline=None)
def test_hypothesis_special_values(
    batch_size, in_channels, kernel_size, n_patches, out_channels, element_strategy
):
    """Test with special values (zeros, ones, uniform values)."""
    height = width = kernel_size * n_patches

    x = torch.full(
        (batch_size, in_channels, height, width),
        fill_value=element_strategy,
        dtype=torch.float32,
    )

    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=kernel_size,
        padding=0,
        bias=True,
    )

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.shape == y_unf.shape
    np.testing.assert_allclose(
        y_conv.detach().numpy(), y_unf.detach().numpy(), rtol=1e-5, atol=1e-5
    )


@given(params=conv_params(), loss_type=st.sampled_from(["sum", "mean", "mse"]))
@settings(max_examples=10, deadline=None)
def test_hypothesis_different_loss_functions(params, loss_type):
    """Test gradient equivalence with different loss functions."""
    torch.manual_seed(42)

    x1 = torch.randn(
        params["batch_size"],
        params["in_channels"],
        params["height"],
        params["width"],
        dtype=params["dtype"],
        requires_grad=True,
    )
    x2 = x1.clone().detach().requires_grad_(True)

    conv = nn.Conv2d(
        params["in_channels"],
        params["out_channels"],
        kernel_size=params["kernel_size"],
        stride=params["kernel_size"],
        padding=0,
        bias=params["bias"],
        dtype=params["dtype"],
    )

    y_conv = transforms.conv2d_to_tokens(x1, conv)
    y_unf = transforms.unfolded_conv2d(x2, conv)

    if loss_type == "sum":
        loss_conv = y_conv.sum()
        loss_unf = y_unf.sum()
    elif loss_type == "mean":
        loss_conv = y_conv.mean()
        loss_unf = y_unf.mean()
    else:  # mse
        target = torch.randn_like(y_conv).detach()
        loss_conv = ((y_conv - target) ** 2).mean()
        loss_unf = ((y_unf - target) ** 2).mean()

    loss_conv.backward()
    loss_unf.backward()

    if params["dtype"] == torch.float64:
        rtol, atol = 1e-10, 1e-10
    else:
        rtol, atol = 1e-5, 1e-5

    np.testing.assert_allclose(
        x1.grad.detach().numpy(), x2.grad.detach().numpy(), rtol=rtol, atol=atol
    )


@st.composite
def edge_case_shapes(draw):
    """Generate edge case shapes for stress testing."""
    case = draw(st.sampled_from(["tiny", "tall", "wide", "square_large", "prime"]))

    if case == "tiny":
        kernel_size = 1
        height = width = draw(st.integers(min_value=1, max_value=3))
    elif case == "tall":
        kernel_size = draw(st.integers(min_value=1, max_value=4))
        n_patches_h = draw(st.integers(min_value=4, max_value=16))
        n_patches_w = 1
        height = n_patches_h * kernel_size
        width = n_patches_w * kernel_size
    elif case == "wide":
        kernel_size = draw(st.integers(min_value=1, max_value=4))
        n_patches_h = 1
        n_patches_w = draw(st.integers(min_value=4, max_value=16))
        height = n_patches_h * kernel_size
        width = n_patches_w * kernel_size
    elif case == "square_large":
        kernel_size = draw(st.sampled_from([8, 16, 32]))
        n_patches = draw(st.integers(min_value=1, max_value=4))
        height = width = n_patches * kernel_size
    else:  # prime
        kernel_size = draw(st.sampled_from([3, 5, 7, 11]))
        n_patches = draw(st.integers(min_value=1, max_value=5))
        height = width = n_patches * kernel_size

    return {
        "kernel_size": kernel_size,
        "height": height,
        "width": width,
        "batch_size": draw(st.integers(min_value=1, max_value=2)),
        "in_channels": draw(st.integers(min_value=1, max_value=8)),
        "out_channels": draw(st.integers(min_value=1, max_value=8)),
        "bias": draw(st.booleans()),
    }


@given(params=edge_case_shapes())
@settings(max_examples=10, deadline=None)
def test_hypothesis_edge_case_shapes(params):
    """Test with edge case shapes."""
    torch.manual_seed(42)

    x = torch.randn(
        params["batch_size"], params["in_channels"], params["height"], params["width"]
    )

    conv = nn.Conv2d(
        params["in_channels"],
        params["out_channels"],
        kernel_size=params["kernel_size"],
        stride=params["kernel_size"],
        padding=0,
        bias=params["bias"],
    )

    y_conv = transforms.conv2d_to_tokens(x, conv)
    y_unf = transforms.unfolded_conv2d(x, conv)

    assert y_conv.shape == y_unf.shape
    np.testing.assert_allclose(
        y_conv.detach().numpy(), y_unf.detach().numpy(), rtol=1e-5, atol=1e-5
    )
