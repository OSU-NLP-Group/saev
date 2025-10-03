import collections
import os
import tempfile

import beartype
import numpy as np
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
