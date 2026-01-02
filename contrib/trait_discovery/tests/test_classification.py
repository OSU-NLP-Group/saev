"""Tests for tie-aware Average Precision in classification.py."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.metrics import average_precision_score
from tdiscovery.classification import compute_ap_for_latent


def sklearn_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Reference AP using sklearn (no tie awareness)."""
    if labels.sum() == 0:
        return 0.0
    return average_precision_score(labels, scores)


def make_one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    n = len(labels)
    one_hot = np.zeros((n, n_classes), dtype=np.float32)
    for i, c in enumerate(labels):
        if 0 <= c < n_classes:
            one_hot[i, c] = 1.0
    return one_hot


# --- Basic functionality tests ---


def test_no_ties_matches_sklearn():
    """When there are no ties, should match sklearn exactly."""
    np.random.seed(42)
    n = 100
    scores = np.random.rand(n).astype(np.float32)
    # Ensure no ties by adding small unique offsets
    scores = scores + np.arange(n) * 1e-7
    labels = np.random.randint(0, 2, n)

    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([labels.sum()], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)
    expected = sklearn_ap(scores, labels)

    np.testing.assert_allclose(ap[0], expected, atol=1e-5)


def test_all_ties_uniform_scores():
    """All items have the same score - expected AP should be well-defined."""
    n = 20
    scores = np.ones(n, dtype=np.float32) * 0.5
    labels = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([labels.sum()], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)

    # AP should be between 0 and 1
    assert 0.0 <= ap[0] <= 1.0

    # Verify by Monte Carlo: average AP over many random orderings
    np.random.seed(42)
    n_trials = 1000
    aps = []
    for _ in range(n_trials):
        # Add larger noise to break ties randomly
        noisy_scores = scores.astype(np.float64) + np.random.uniform(-1e-6, 1e-6, n)
        aps.append(sklearn_ap(noisy_scores, labels))
    mean_ap = np.mean(aps)

    np.testing.assert_allclose(ap[0], mean_ap, atol=0.02)


def test_ties_converge_to_expected():
    """With random tie-breaking, average over permutations should match expected AP."""
    np.random.seed(123)
    n = 50
    # Create scores with many ties (use float64 for noise precision)
    scores = np.array([0.9] * 10 + [0.5] * 30 + [0.1] * 10, dtype=np.float64)
    labels = np.random.randint(0, 2, n)

    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([labels.sum()], dtype=np.float32)

    # Compute expected AP using our tie-aware formula
    expected_ap = compute_ap_for_latent(scores.astype(np.float32), one_hot, n_pos)[0]

    # Compute average AP over many random tie-breaking permutations
    n_trials = 1000
    aps = []
    for _ in range(n_trials):
        # Use larger noise that won't get lost in float precision
        noisy_scores = scores + np.random.uniform(-1e-6, 1e-6, n)
        ap = sklearn_ap(noisy_scores, labels)
        aps.append(ap)

    mean_ap = np.mean(aps)

    # Expected AP should be close to the mean of random tie-breaking
    np.testing.assert_allclose(expected_ap, mean_ap, atol=0.02)


def test_all_positives():
    """All labels are positive - AP should be 1.0."""
    n = 10
    scores = np.random.rand(n).astype(np.float32)
    labels = np.ones(n)

    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([n], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)
    np.testing.assert_allclose(ap[0], 1.0)


def test_all_negatives():
    """All labels are negative - AP should be 0.0."""
    n = 10
    scores = np.random.rand(n).astype(np.float32)
    labels = np.zeros(n)

    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([0], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)
    assert ap[0] == 0.0


def test_single_element_positive():
    """Single positive element."""
    scores = np.array([0.5], dtype=np.float32)
    labels = np.array([1])

    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([1], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)
    np.testing.assert_allclose(ap[0], 1.0)


def test_single_element_negative():
    """Single negative element."""
    scores = np.array([0.5], dtype=np.float32)
    labels = np.array([0])

    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([0], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)
    assert ap[0] == 0.0


def test_perfect_ranking():
    """Perfect ranking - all positives ranked above negatives."""
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1], dtype=np.float32)
    labels = np.array([1, 1, 1, 0, 0])

    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([3], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)
    np.testing.assert_allclose(ap[0], 1.0)


def test_worst_ranking():
    """Worst ranking - all negatives ranked above positives."""
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.0], dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 1])

    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([3], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)
    # AP = (1/3 + 2/4 + 3/5) / 3 = (0.333 + 0.5 + 0.6) / 3 = 0.478
    expected = (1 / 3 + 2 / 4 + 3 / 5) / 3
    np.testing.assert_allclose(ap[0], expected, atol=1e-5)


def test_multiple_classes():
    """Test with multiple segmentation classes."""
    n = 20
    n_classes = 3
    scores = np.random.rand(n).astype(np.float32)
    # Ensure no ties
    scores = scores + np.arange(n) * 1e-7

    # Create labels for 3 classes
    labels_int = np.random.randint(0, n_classes, n)
    one_hot = make_one_hot(labels_int, n_classes)
    n_pos = one_hot.sum(axis=0)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)

    # Check against sklearn for each class
    for c in range(n_classes):
        if n_pos[c] > 0:
            expected = sklearn_ap(scores, one_hot[:, c])
            np.testing.assert_allclose(ap[c], expected, atol=1e-5)


def test_sparse_activations():
    """Test with sparse activations (many zeros) - realistic for SAE."""
    n = 100
    # Mostly zeros with a few non-zero values
    scores = np.zeros(n, dtype=np.float32)
    scores[:5] = [0.9, 0.7, 0.5, 0.3, 0.1]
    np.random.shuffle(scores)

    labels = np.random.randint(0, 2, n)
    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([labels.sum()], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)
    assert 0.0 <= ap[0] <= 1.0


# --- Property-based tests ---


@given(
    arrays(
        np.float32,
        (50,),
        elements=st.floats(0, 1, allow_nan=False, allow_infinity=False),
    ),
    arrays(np.float32, (50,), elements=st.sampled_from([0.0, 1.0])),
)
@settings(max_examples=100)
def test_ap_bounds(scores, labels):
    """AP should always be between 0 and 1."""
    n_pos = labels.sum()
    if n_pos == 0:
        return  # Skip if no positives

    one_hot = labels.reshape(-1, 1)
    n_pos_arr = np.array([n_pos], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos_arr)
    assert 0.0 <= ap[0] <= 1.0, f"AP out of bounds: {ap[0]}"


@given(
    arrays(
        np.float32,
        (30,),
        elements=st.floats(0, 1, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=50)
def test_all_positive_labels_gives_ap_one(scores):
    """If all labels are positive, AP should be 1.0 regardless of scores."""
    n = len(scores)
    labels = np.ones(n, dtype=np.float32)
    one_hot = labels.reshape(-1, 1)
    n_pos = np.array([n], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)
    np.testing.assert_allclose(ap[0], 1.0)


@given(
    arrays(
        np.float32,
        (30,),
        elements=st.floats(0, 1, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=50)
def test_all_negative_labels_gives_ap_zero(scores):
    """If all labels are negative, AP should be 0.0."""
    n = len(scores)
    labels = np.zeros(n, dtype=np.float32)
    one_hot = labels.reshape(-1, 1)
    n_pos = np.array([0], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)
    assert ap[0] == 0.0, f"Expected 0.0, got {ap[0]}"


# --- Regression tests for specific edge cases ---


def test_two_items_one_positive():
    """Two items, one positive - test different score orderings."""
    # Positive ranked first
    scores = np.array([0.9, 0.1], dtype=np.float32)
    labels = np.array([1.0, 0.0])
    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([1], dtype=np.float32)
    ap = compute_ap_for_latent(scores, one_hot, n_pos)
    np.testing.assert_allclose(ap[0], 1.0)

    # Positive ranked second
    scores = np.array([0.9, 0.1], dtype=np.float32)
    labels = np.array([0.0, 1.0])
    ap = compute_ap_for_latent(scores, labels.reshape(-1, 1).astype(np.float32), n_pos)
    np.testing.assert_allclose(ap[0], 0.5)

    # Tied scores - expected AP should be average of the two cases = 0.75
    scores = np.array([0.5, 0.5], dtype=np.float32)
    labels = np.array([1.0, 0.0])
    ap = compute_ap_for_latent(scores, labels.reshape(-1, 1).astype(np.float32), n_pos)
    np.testing.assert_allclose(ap[0], 0.75)


def test_three_items_tied():
    """Three items all tied, one positive."""
    scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    labels = np.array([1.0, 0.0, 0.0])
    one_hot = labels.reshape(-1, 1).astype(np.float32)
    n_pos = np.array([1], dtype=np.float32)

    ap = compute_ap_for_latent(scores, one_hot, n_pos)

    # Expected: average of AP when positive is at position 1, 2, or 3
    # Position 1: precision = 1/1 = 1.0, AP = 1.0
    # Position 2: precision = 1/2 = 0.5, AP = 0.5
    # Position 3: precision = 1/3 = 0.333, AP = 0.333
    # Expected AP = (1.0 + 0.5 + 0.333) / 3 = 0.611
    expected = (1.0 + 0.5 + 1 / 3) / 3
    np.testing.assert_allclose(ap[0], expected, atol=1e-5)
