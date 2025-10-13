import collections
import os
import tempfile

import beartype
import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Int, jaxtyped

from saev import helpers


@st.composite
def _labels_and_n(draw):
    """
    Helper strategy: (labels array, n) with n <= len(labels) and n >= #classes
    """
    labels_list = draw(
        st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=300)
    )
    labels = np.array(labels_list, dtype=int)
    n_classes = len(np.unique(labels))
    # choose n in [n_classes, len(labels)]
    n = draw(st.integers(min_value=n_classes, max_value=len(labels)))
    return labels, n


@jaxtyped(typechecker=beartype.beartype)
def _measure_balance(
    labels: Int[np.ndarray, " n_labels"], indices: Int[np.ndarray, " n"]
) -> float:
    """
    Calculate a balance metric (coefficient of variation, lower is better) for the selected samples (labels[indices]).

    Returns 0 for perfect balance, higher for more imbalance.
    """
    if len(indices) == 0:
        return 0.0

    # Get the distribution of classes in the selected samples
    selected_labels = labels[indices]
    class_counts = collections.Counter(selected_labels)

    # Get all unique classes in the original dataset
    all_classes = set(labels)

    # Check if it was possible to include at least one of each class but didn't
    if len(indices) >= len(all_classes) and len(class_counts) < len(all_classes):
        return float("inf")

    # Calculate coefficient of variation (standard deviation / mean)
    counts = np.array(list(class_counts.values()))

    # If only one class is present, return a high value to indicate imbalance
    if len(counts) == 1:
        return float("inf")

    mean = np.mean(counts)
    std = np.std(counts, ddof=1)  # Using sample standard deviation

    # Return coefficient of variation (0 for perfect balance)
    return std / mean if mean > 0 else 0.0


@given(
    total=st.integers(min_value=0, max_value=1_000),
    batch=st.integers(min_value=1, max_value=400),
)
def test_batched_idx_covers_range_without_overlap(total, batch):
    """batched_idx must partition [0,total) into consecutive, non-overlapping spans, each of length <= batch."""
    spans = list(helpers.batched_idx(total, batch))

    # edge-case: nothing to iterate
    if total == 0:
        assert spans == []
        return

    # verify each span and overall coverage
    covered = []
    expected_start = 0
    for start, stop in spans:
        # bounds & width checks
        assert 0 <= start < stop <= total
        assert (stop - start) <= batch
        # consecutiveness (no gaps/overlap)
        assert start == expected_start
        expected_start = stop
        covered.extend(range(start, stop))

    # spans collectively cover exactly [0, total)
    assert covered == list(range(total))


def test_fssafe_common_cases():
    """Test fssafe with common checkpoint names."""
    # HuggingFace hub format
    assert (
        helpers.fssafe("hf-hub:timm/ViT-L-16-SigLIP2-256")
        == "hf-hub_timm_ViT-L-16-SigLIP2-256"
    )

    # Path with slashes
    assert helpers.fssafe("/path/to/model.pt") == "_path_to_model.pt"

    # Windows path
    assert (
        helpers.fssafe("C:\\Users\\Model\\checkpoint.bin")
        == "C__Users_Model_checkpoint.bin"
    )

    # Special characters
    assert helpers.fssafe("model*name?test") == "model_name_test"

    # Already safe string
    assert helpers.fssafe("safe_filename_123.txt") == "safe_filename_123.txt"


def test_fssafe_edge_cases():
    """Test fssafe edge cases."""
    # Empty string
    assert helpers.fssafe("") == ""

    # Only special characters
    assert helpers.fssafe('://\\*?"<>|') == "__________"

    # Spaces
    assert helpers.fssafe("file with spaces.txt") == "file_with_spaces.txt"

    # Newlines and tabs
    assert (
        helpers.fssafe("file\nwith\nnewlines\tand\ttabs")
        == "file_with_newlines_and_tabs"
    )


def test_fssafe_creates_valid_files():
    """Test that fssafe output can be used as filenames."""
    test_cases = [
        "hf-hub:timm/ViT-L-16-SigLIP2-256",
        "path/to/model:version2",
        "model<>name|with*special?chars",
        'file"with"quotes',
        "tabs\tand\nnewlines\rtest",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for original in test_cases:
            safe_name = helpers.fssafe(original)
            if safe_name:  # Skip empty names
                file_path = os.path.join(tmpdir, safe_name + ".txt")
                # This should not raise an exception
                with open(file_path, "w") as f:
                    f.write("test")
                assert os.path.exists(file_path)


@given(
    st.text(
        alphabet=st.characters(min_codepoint=1, max_codepoint=1000),
        min_size=1,
        max_size=200,
    )
)
@settings(max_examples=1000, deadline=None)
def test_fssafe_property(input_string):
    """Property test: fssafe should always produce valid filenames."""
    safe_name = helpers.fssafe(input_string)

    # If the result is not empty, we should be able to create a file with it
    if safe_name:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Truncate to avoid filesystem limits
            truncated = safe_name[:200] if len(safe_name) > 200 else safe_name
            file_path = os.path.join(tmpdir, truncated + ".test")

            # This should not raise an exception
            with open(file_path, "w") as f:
                f.write("test")

            # Verify we can read it back
            with open(file_path, "r") as f:
                assert f.read() == "test"


@given(
    arr=st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    k=st.integers(min_value=1, max_value=10),
)
def test_np_topk_1d(arr, k):
    """np_topk should match torch.topk for 1D arrays."""
    np_arr = np.array(arr, dtype=np.float32)
    torch_arr = torch.from_numpy(np_arr)

    # Adjust k if it's larger than array size
    k = min(k, len(arr))

    np_result = helpers.np_topk(np_arr, k, axis=None)
    torch_values, torch_indices = torch.topk(torch_arr, k)

    # Check values match
    np.testing.assert_allclose(
        np_result.values, torch_values.numpy(), rtol=1e-5, atol=1e-7
    )
    # Check that indices point to the correct values
    # (indices may differ for equal values, which is acceptable)
    indexed_values = np_arr[np_result.indices]
    np.testing.assert_allclose(
        indexed_values, torch_values.numpy(), rtol=1e-5, atol=1e-7
    )


@given(
    shape=st.tuples(
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=1, max_value=20),
    ),
    k=st.integers(min_value=1, max_value=10),
    axis=st.integers(min_value=-1, max_value=1),
)
def test_np_topk_2d(shape, k, axis):
    """np_topk should match torch.topk for 2D arrays with different axes."""
    # Create random array
    np_arr = np.random.randn(*shape).astype(np.float32)
    torch_arr = torch.from_numpy(np_arr)

    # Normalize axis to positive
    if axis < 0:
        axis = len(shape) + axis

    # Adjust k if it's larger than the dimension size
    k = min(k, shape[axis])

    np_result = helpers.np_topk(np_arr, k, axis=axis)
    torch_values, torch_indices = torch.topk(torch_arr, k, dim=axis)

    # Check values match
    np.testing.assert_allclose(
        np_result.values, torch_values.numpy(), rtol=1e-5, atol=1e-7
    )
    # Check that indices point to the correct values
    indexed_values = np.take_along_axis(np_arr, np_result.indices, axis=axis)
    np.testing.assert_allclose(
        indexed_values, torch_values.numpy(), rtol=1e-5, atol=1e-7
    )


@given(
    shape=st.tuples(
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
    ),
    k=st.integers(min_value=1, max_value=5),
    axis=st.integers(min_value=-1, max_value=2),
)
def test_np_topk_3d(shape, k, axis):
    """np_topk should match torch.topk for 3D arrays with different axes."""
    # Create random array
    np_arr = np.random.randn(*shape).astype(np.float32)
    torch_arr = torch.from_numpy(np_arr)

    # Normalize axis to positive
    if axis < 0:
        axis = len(shape) + axis

    # Adjust k if it's larger than the dimension size
    k = min(k, shape[axis])

    np_result = helpers.np_topk(np_arr, k, axis=axis)
    torch_values, torch_indices = torch.topk(torch_arr, k, dim=axis)

    # Check values match
    np.testing.assert_allclose(
        np_result.values, torch_values.numpy(), rtol=1e-5, atol=1e-7
    )
    # Check that indices point to the correct values
    indexed_values = np.take_along_axis(np_arr, np_result.indices, axis=axis)
    np.testing.assert_allclose(
        indexed_values, torch_values.numpy(), rtol=1e-5, atol=1e-7
    )


def test_np_topk_edge_cases():
    """Test np_topk edge cases to match torch.topk."""
    # Test with k=1
    arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    result = helpers.np_topk(arr, 1, axis=None)
    torch_values, torch_indices = torch.topk(torch.from_numpy(arr), 1)
    np.testing.assert_allclose(result.values, torch_values.numpy())
    # Check indices point to correct values
    np.testing.assert_allclose(arr[result.indices], torch_values.numpy())

    # Test with k equal to array size (has duplicate values)
    result = helpers.np_topk(arr, len(arr), axis=None)
    torch_values, torch_indices = torch.topk(torch.from_numpy(arr), len(arr))
    np.testing.assert_allclose(result.values, torch_values.numpy())
    # Check indices point to correct values (may differ for ties)
    np.testing.assert_allclose(arr[result.indices], torch_values.numpy())

    # Test with 2D array, axis=0
    arr_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = helpers.np_topk(arr_2d, 1, axis=0)
    torch_values, torch_indices = torch.topk(torch.from_numpy(arr_2d), 1, dim=0)
    np.testing.assert_allclose(result.values, torch_values.numpy())
    np.testing.assert_array_equal(result.indices, torch_indices.numpy())

    # Test with 2D array, axis=1
    result = helpers.np_topk(arr_2d, 2, axis=1)
    torch_values, torch_indices = torch.topk(torch.from_numpy(arr_2d), 2, dim=1)
    np.testing.assert_allclose(result.values, torch_values.numpy())
    np.testing.assert_array_equal(result.indices, torch_indices.numpy())

    # Test with negative values (no duplicates, so indices should match exactly)
    arr_neg = np.array([-5.0, -2.0, -8.0, -1.0, -3.0])
    result = helpers.np_topk(arr_neg, 3, axis=None)
    torch_values, torch_indices = torch.topk(torch.from_numpy(arr_neg), 3)
    np.testing.assert_allclose(result.values, torch_values.numpy())
    np.testing.assert_array_equal(result.indices, torch_indices.numpy())
