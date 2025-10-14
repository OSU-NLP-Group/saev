import collections
import os
import tempfile
import tracemalloc

import beartype
import numpy as np
import scipy.sparse
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


@given(
    shape=st.tuples(
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=1, max_value=20),
    ),
    k=st.integers(min_value=1, max_value=10),
    sparsity=st.floats(min_value=0.1, max_value=0.9),
)
def test_csr_topk_2d(shape, k, sparsity):
    """csr_topk should match np_topk for CSR matrices."""
    # Create random sparse array
    n_nonzero = int(shape[0] * shape[1] * (1 - sparsity))
    n_nonzero = max(1, n_nonzero)  # At least one non-zero element

    # Create dense array and convert to CSR
    dense = np.zeros(shape, dtype=np.float32)
    indices = np.random.choice(shape[0] * shape[1], size=n_nonzero, replace=False)
    rows = indices // shape[1]
    cols = indices % shape[1]
    values = np.random.randn(n_nonzero).astype(np.float32)
    dense[rows, cols] = values

    csr = scipy.sparse.csr_matrix(dense)

    # Adjust k to be at most the number of columns
    k = min(k, shape[1])

    # Test axis=1 (along columns, which is what CSR is good at)
    csr_result = helpers.csr_topk(csr, k=k, axis=1)
    np_result = helpers.np_topk(dense, k, axis=1)

    # Check values match
    np.testing.assert_allclose(
        csr_result.values, np_result.values, rtol=1e-5, atol=1e-7
    )
    # Check that indices point to the correct values (only for non-zero values)
    for i in range(shape[0]):
        # Only check non-zero values (sparse rows may have fewer than k non-zeros)
        nonzero_mask = csr_result.values[i] != 0
        if np.any(nonzero_mask):
            indexed_values = dense[i, csr_result.indices[i][nonzero_mask]]
            np.testing.assert_allclose(
                indexed_values, np_result.values[i][nonzero_mask], rtol=1e-5, atol=1e-7
            )


def test_csr_topk_edge_cases():
    """Test csr_topk edge cases."""
    # Test with fully sparse row
    dense = np.array([[1.0, 2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0], [5.0, 4.0, 3.0, 2.0]])
    csr = scipy.sparse.csr_matrix(dense)

    result = helpers.csr_topk(csr, k=2, axis=1)
    np_result = helpers.np_topk(dense, 2, axis=1)

    np.testing.assert_allclose(result.values, np_result.values, rtol=1e-5, atol=1e-7)
    # Check indices point to correct values
    for i in range(dense.shape[0]):
        indexed_values = dense[i, result.indices[i]]
        np.testing.assert_allclose(
            indexed_values, np_result.values[i], rtol=1e-5, atol=1e-7
        )

    # Test with k=1
    result = helpers.csr_topk(csr, k=1, axis=1)
    np_result = helpers.np_topk(dense, 1, axis=1)
    np.testing.assert_allclose(result.values, np_result.values, rtol=1e-5, atol=1e-7)

    # Test with k equal to number of columns
    result = helpers.csr_topk(csr, k=4, axis=1)
    np_result = helpers.np_topk(dense, 4, axis=1)
    np.testing.assert_allclose(result.values, np_result.values, rtol=1e-5, atol=1e-7)

    # Test with all positive values
    dense_pos = np.array([[3.0, 1.0, 4.0, 2.0], [5.0, 6.0, 1.0, 2.0]])
    csr_pos = scipy.sparse.csr_matrix(dense_pos)
    result = helpers.csr_topk(csr_pos, k=3, axis=1)
    np_result = helpers.np_topk(dense_pos, 3, axis=1)
    np.testing.assert_allclose(result.values, np_result.values, rtol=1e-5, atol=1e-7)

    # Test with negative values
    dense_neg = np.array([[-5.0, -2.0, -8.0, -1.0], [-3.0, -6.0, -4.0, -7.0]])
    csr_neg = scipy.sparse.csr_matrix(dense_neg)
    result = helpers.csr_topk(csr_neg, k=2, axis=1)
    np_result = helpers.np_topk(dense_neg, 2, axis=1)
    np.testing.assert_allclose(result.values, np_result.values, rtol=1e-5, atol=1e-7)


def test_csr_topk_single_row():
    """Test csr_topk with a single row (common case for batch processing)."""
    # Single row with various values
    dense = np.array([[5.0, 2.0, 8.0, 1.0, 3.0, 7.0, 4.0, 6.0]])
    csr = scipy.sparse.csr_matrix(dense)

    for k_val in [1, 3, 5, 8]:
        result = helpers.csr_topk(csr, k=k_val, axis=1)
        np_result = helpers.np_topk(dense, k_val, axis=1)
        np.testing.assert_allclose(
            result.values, np_result.values, rtol=1e-5, atol=1e-7
        )
        # Check indices point to correct values
        indexed_values = dense[0, result.indices[0]]
        np.testing.assert_allclose(
            indexed_values, np_result.values[0], rtol=1e-5, atol=1e-7
        )


def test_csr_topk_with_duplicates():
    """Test csr_topk handles duplicate values correctly."""
    # Array with many duplicates
    dense = np.array([[5.0, 5.0, 5.0, 1.0], [3.0, 3.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]])
    csr = scipy.sparse.csr_matrix(dense)

    result = helpers.csr_topk(csr, k=2, axis=1)
    np_result = helpers.np_topk(dense, 2, axis=1)

    # Values should match exactly
    np.testing.assert_allclose(result.values, np_result.values, rtol=1e-5, atol=1e-7)
    # Indices may differ for ties, but should point to correct values
    for i in range(dense.shape[0]):
        indexed_values = dense[i, result.indices[i]]
        np.testing.assert_allclose(
            indexed_values, np_result.values[i], rtol=1e-5, atol=1e-7
        )


def test_csr_topk_memory_axis1():
    """Test that axis=1 memory usage is reasonable (processes one row at a time)."""
    # Create a moderately sized sparse matrix
    n_rows, n_cols = 1000, 500
    sparsity = 0.9
    n_nonzero = int(n_rows * n_cols * (1 - sparsity))

    dense = np.zeros((n_rows, n_cols), dtype=np.float32)
    indices = np.random.choice(n_rows * n_cols, size=n_nonzero, replace=False)
    rows = indices // n_cols
    cols = indices % n_cols
    values = np.random.randn(n_nonzero).astype(np.float32)
    dense[rows, cols] = values

    csr = scipy.sparse.csr_matrix(dense)

    # Measure memory usage
    tracemalloc.start()
    result = helpers.csr_topk(csr, k=10, axis=1)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Memory should be reasonable - much less than the full dense matrix
    # Dense matrix would be: n_rows * n_cols * 4 bytes = 2 MB
    # Peak should be well under that since we process row-by-row
    dense_size_mb = (n_rows * n_cols * 4) / (1024 * 1024)
    peak_mb = peak / (1024 * 1024)

    # Assert peak memory is less than 50% of dense matrix size
    assert peak_mb < 0.5 * dense_size_mb, (
        f"Peak memory {peak_mb:.2f} MB exceeds 50% of dense size {dense_size_mb:.2f} MB"
    )

    # Verify result shape
    assert result.values.shape == (n_rows, 10)
    assert result.indices.shape == (n_rows, 10)


def test_csr_topk_memory_axis0_batch_scaling():
    """Test that axis=0 memory scales with batch_size, not total rows."""
    n_rows, n_cols = 5000, 100
    k = 5
    sparsity = 0.95
    n_nonzero = int(n_rows * n_cols * (1 - sparsity))

    # Create sparse matrix
    dense = np.zeros((n_rows, n_cols), dtype=np.float32)
    indices = np.random.choice(n_rows * n_cols, size=n_nonzero, replace=False)
    rows = indices // n_cols
    cols = indices % n_cols
    values = np.random.randn(n_nonzero).astype(np.float32)
    dense[rows, cols] = values
    csr = scipy.sparse.csr_matrix(dense)

    # Test with small batch size
    tracemalloc.start()
    result_small = helpers.csr_topk(csr, k=k, axis=0, batch_size=100)
    _, peak_small = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Test with larger batch size
    tracemalloc.start()
    result_large = helpers.csr_topk(csr, k=k, axis=0, batch_size=1000)
    _, peak_large = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Results should be the same
    np.testing.assert_allclose(result_small.values, result_large.values)

    # Larger batch should use more memory
    assert peak_large > peak_small, (
        f"Larger batch_size should use more memory: "
        f"small={peak_small / 1024 / 1024:.2f} MB, "
        f"large={peak_large / 1024 / 1024:.2f} MB"
    )

    # But memory should scale reasonably (roughly linearly with batch size)
    # Allow some overhead, so check if within 20x ratio (10x batch size increase)
    ratio = peak_large / peak_small
    assert ratio < 20, f"Memory ratio {ratio:.2f} too high for 10x batch size increase"


def test_csr_topk_memory_axis0_bounded():
    """Test that axis=0 memory doesn't grow with number of rows beyond batch size."""
    n_cols = 200
    k = 10
    batch_size = 512
    sparsity = 0.9

    # Test with moderate number of rows
    n_rows_small = 2000
    n_nonzero_small = int(n_rows_small * n_cols * (1 - sparsity))
    dense_small = np.zeros((n_rows_small, n_cols), dtype=np.float32)
    indices_small = np.random.choice(
        n_rows_small * n_cols, size=n_nonzero_small, replace=False
    )
    rows_small = indices_small // n_cols
    cols_small = indices_small % n_cols
    values_small = np.random.randn(n_nonzero_small).astype(np.float32)
    dense_small[rows_small, cols_small] = values_small
    csr_small = scipy.sparse.csr_matrix(dense_small)

    tracemalloc.start()
    _ = helpers.csr_topk(csr_small, k=k, axis=0, batch_size=batch_size)
    _, peak_small = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Test with many more rows (5x)
    n_rows_large = 10000
    n_nonzero_large = int(n_rows_large * n_cols * (1 - sparsity))
    dense_large = np.zeros((n_rows_large, n_cols), dtype=np.float32)
    indices_large = np.random.choice(
        n_rows_large * n_cols, size=n_nonzero_large, replace=False
    )
    rows_large = indices_large // n_cols
    cols_large = indices_large % n_cols
    values_large = np.random.randn(n_nonzero_large).astype(np.float32)
    dense_large[rows_large, cols_large] = values_large
    csr_large = scipy.sparse.csr_matrix(dense_large)

    tracemalloc.start()
    _ = helpers.csr_topk(csr_large, k=k, axis=0, batch_size=batch_size)
    _, peak_large = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Memory should not grow significantly with more rows (same batch_size)
    # Allow for some growth due to tracking arrays, but should be bounded
    ratio = peak_large / peak_small
    assert ratio < 2.0, (
        f"Memory should not grow significantly with 5x more rows: "
        f"ratio={ratio:.2f}, small={peak_small / 1024 / 1024:.2f} MB, "
        f"large={peak_large / 1024 / 1024:.2f} MB"
    )


def test_csr_topk_memory_axis0_large_matrix_estimate():
    """Test memory usage with parameters similar to real use case and verify it's reasonable."""
    # Simulate a smaller version of the real use case:
    # Real: (5.2M rows, 16K cols, k=20)
    # Test: (10K rows, 1K cols, k=20) - scaled down 500x on rows, 16x on cols
    n_rows, n_cols = 10000, 1000
    k = 20
    batch_size = 1024
    sparsity = 0.98  # Similar to real case

    n_nonzero = int(n_rows * n_cols * (1 - sparsity))
    dense = np.zeros((n_rows, n_cols), dtype=np.float32)
    indices = np.random.choice(n_rows * n_cols, size=n_nonzero, replace=False)
    rows = indices // n_cols
    cols = indices % n_cols
    values = np.random.randn(n_nonzero).astype(np.float32)
    dense[rows, cols] = values
    csr = scipy.sparse.csr_matrix(dense)

    tracemalloc.start()
    result = helpers.csr_topk(csr, k=k, axis=0, batch_size=batch_size)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate expected memory components:
    # 1. Dense batch: batch_size * n_cols * 4 bytes
    batch_size_mb = (batch_size * n_cols * 4) / (1024 * 1024)
    # 2. Tracking arrays: k * n_cols * (4 + 8 + 4 + 4) bytes
    #    (values, indices, min_values, counts)
    tracking_size_mb = (k * n_cols * 20) / (1024 * 1024)

    peak_mb = peak / (1024 * 1024)
    expected_mb = batch_size_mb + tracking_size_mb

    # Peak should be within 3x of expected (allows for overhead and copies)
    assert peak_mb < 3 * expected_mb, (
        f"Peak memory {peak_mb:.2f} MB exceeds 3x expected {expected_mb:.2f} MB "
        f"(batch={batch_size_mb:.2f} MB + tracking={tracking_size_mb:.2f} MB)"
    )

    # Verify result shape
    assert result.values.shape == (k, n_cols)
    assert result.indices.shape == (k, n_cols)
