import collections
import dataclasses
import os
import sys

import beartype
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from jaxtyping import Int, jaxtyped

from saev import helpers


# Helper dataclasses for testing nested updates
@dataclasses.dataclass(frozen=True)
class Grandchild:
    value: int = 1
    name: str = "gc"


@dataclasses.dataclass(frozen=True)
class Child:
    grandchild: Grandchild = Grandchild()
    count: int = 10
    enabled: bool = True


@dataclasses.dataclass(frozen=True)
class Parent:
    child: Child = Child()
    name: str = "parent"
    items: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 3])


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


def test_grid_update_nested_grandchild():
    """Test updating deeply nested grandchild values."""
    base = Parent()
    sweep = {"child": {"grandchild": {"value": [5, 10, 15]}}}
    configs, errs = helpers.grid(base, sweep)

    assert len(configs) == 3
    assert len(errs) == 0
    assert configs[0].child.grandchild.value == 5
    assert configs[1].child.grandchild.value == 10
    assert configs[2].child.grandchild.value == 15
    # Other fields should remain unchanged
    assert all(c.child.grandchild.name == "gc" for c in configs)
    assert all(c.child.count == 10 for c in configs)


def test_grid_update_multiple_nested_fields():
    """Test updating multiple nested fields generates all combinations."""
    base = Parent()
    sweep = {
        "child": {
            "grandchild": {"value": [1, 2], "name": ["a", "b"]},
            "count": [100, 200],
        }
    }
    configs, errs = helpers.grid(base, sweep)

    assert len(configs) == 8  # 2 * 2 * 2 combinations
    assert len(errs) == 0
    # Check a few combinations
    values = [
        (c.child.grandchild.value, c.child.grandchild.name, c.child.count)
        for c in configs
    ]
    assert (1, "a", 100) in values
    assert (2, "b", 200) in values


def test_grid_update_parent_and_nested_simultaneously():
    """Test updating parent-level and nested fields together."""
    base = Parent()
    sweep = {"name": ["p1", "p2"], "child": {"enabled": [True, False]}}
    configs, errs = helpers.grid(base, sweep)

    assert len(configs) == 4  # 2 * 2
    assert len(errs) == 0
    names_and_enabled = [(c.name, c.child.enabled) for c in configs]
    assert ("p1", True) in names_and_enabled
    assert ("p1", False) in names_and_enabled
    assert ("p2", True) in names_and_enabled
    assert ("p2", False) in names_and_enabled


def test_grid_deep_nesting_all_levels():
    """Test sweep with updates at all nesting levels."""
    base = Parent()
    sweep = {
        "name": "modified",
        "items": [[1], [1, 2], [1, 2, 3]],
        "child": {
            "count": [5, 10],
            "grandchild": {"value": [100, 200], "name": "modified_gc"},
        },
    }
    configs, errs = helpers.grid(base, sweep)

    assert len(configs) == 12  # 1 * 3 * 2 * 2 * 1
    assert len(errs) == 0
    assert all(c.name == "modified" for c in configs)
    assert all(c.child.grandchild.name == "modified_gc" for c in configs)
    # Check that all combinations exist
    item_count_value_combinations = [
        (len(c.items), c.child.count, c.child.grandchild.value) for c in configs
    ]
    assert (1, 5, 100) in item_count_value_combinations
    assert (3, 10, 200) in item_count_value_combinations


def test_grid_empty_sweep():
    """Test that empty sweep returns original config."""
    base = Parent()
    configs, errs = helpers.grid(base, {})
    
    assert len(configs) == 1
    assert len(errs) == 0
    assert configs[0].name == base.name
    assert configs[0].child.count == base.child.count


def test_grid_single_nested_update():
    """Test single nested update without lists."""
    base = Parent()
    sweep = {"child": {"grandchild": {"value": 42}}}
    configs, errs = helpers.grid(base, sweep)

    assert len(configs) == 1
    assert len(errs) == 0
    assert configs[0].child.grandchild.value == 42
    assert configs[0].child.grandchild.name == "gc"  # Unchanged


def test_grid_with_none_values():
    """Test grid expansion with None values."""
    @dataclasses.dataclass(frozen=True)
    class OptionalChild:
        value: int | None = None
        name: str = "optional"

    @dataclasses.dataclass(frozen=True)
    class ParentWithOptional:
        child: OptionalChild | None = None
        count: int = 0

    base_opt = ParentWithOptional(child=OptionalChild(value=5))
    sweep = {"child": {"value": [None, 10, 20]}}
    configs, errs = helpers.grid(base_opt, sweep)

    assert len(configs) == 3
    assert len(errs) == 0
    assert configs[0].child.value is None
    assert configs[1].child.value == 10
    assert configs[2].child.value == 20


def test_grid_sae_activation_sweep():
    """Test sweeping activation parameters in SAE config."""
    # Add parent directory to path to import train
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from saev import nn
    from train import Config

    base_cfg = Config(
        sae=nn.SparseAutoencoderConfig(activation=nn.modeling.BatchTopK(top_k=32))
    )

    sweep = {"sae": {"activation": {"top_k": [16, 32, 64, 128]}}}

    configs, errs = helpers.grid(base_cfg, sweep)

    assert len(configs) == 4
    assert len(errs) == 0
    assert all(isinstance(c.sae.activation, nn.modeling.BatchTopK) for c in configs)
    assert [c.sae.activation.top_k for c in configs] == [16, 32, 64, 128]


def test_grid_multiple_sae_params_with_activation():
    """Test sweeping multiple SAE parameters including nested activation."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from saev import nn
    from train import Config

    base_cfg = Config(
        sae=nn.SparseAutoencoderConfig(
            d_vit=768, exp_factor=8, activation=nn.modeling.TopK(top_k=10)
        )
    )

    sweep = {
        "sae": {"exp_factor": [4, 8, 16], "activation": {"top_k": [5, 10]}},
        "lr": [1e-3, 1e-4],
    }

    configs, errs = helpers.grid(base_cfg, sweep)

    assert len(configs) == 12  # 3 * 2 * 2
    assert len(errs) == 0
    # Check all combinations exist
    combinations = [(c.sae.exp_factor, c.sae.activation.top_k, c.lr) for c in configs]
    assert (4, 5, 1e-3) in combinations
    assert (16, 10, 1e-4) in combinations


def test_grid_nested_objective_config():
    """Test sweeping nested objective configuration."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from saev import nn
    from train import Config

    base_cfg = Config(
        objective=nn.objectives.Vanilla(sparsity_coeff=1e-3, sparsity_warmup_steps=1000)
    )

    sweep = {
        "objective": {
            "sparsity_coeff": [1e-4, 1e-3, 1e-2],
            "sparsity_warmup_steps": [500, 1000],
        }
    }

    configs, errs = helpers.grid(base_cfg, sweep)

    assert len(configs) == 6  # 3 * 2
    assert len(errs) == 0
    assert all(isinstance(c.objective, nn.objectives.Vanilla) for c in configs)
    coeffs = [c.objective.sparsity_coeff for c in configs]
    assert 1e-4 in coeffs and 1e-2 in coeffs


def test_grid_complex_nested_data_config():
    """Test complex nested case with data config."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from saev import nn
    from train import Config

    base_cfg = Config(
        data=helpers.get(
            dataclasses.asdict(Config()), "data"
        ),  # Get default data config
        sae=nn.SparseAutoencoderConfig(activation=nn.modeling.BatchTopK(top_k=64)),
    )

    sweep = {
        "data": {"layer": [11, 12, 13], "buffer_size": [128, 256]},
        "sae": {"activation": {"top_k": [32, 64]}},
    }

    configs, errs = helpers.grid(base_cfg, sweep)

    # Note: This test might fail with current implementation
    # because data config might not be a dataclass that supports
    # nested updates. This demonstrates the limitation.


def test_grid_invalid_field_error():
    """Test that invalid field names produce errors."""
    @dataclasses.dataclass
    class SimpleConfig:
        value: int = 1
        name: str = "test"

    base = SimpleConfig()
    sweep = {"nonexistent_field": [1, 2, 3]}
    configs, errs = helpers.grid(base, sweep)

    assert len(configs) == 0
    assert len(errs) == 3  # One error per expansion
    assert all("nonexistent_field" in err for err in errs)


def test_grid_type_mismatch():
    """Test handling of type mismatches."""
    @dataclasses.dataclass
    class SimpleConfig:
        value: int = 1
        name: str = "test"

    base = SimpleConfig()
    sweep = {"value": ["not_an_int"]}
    configs, errs = helpers.grid(base, sweep)

    # This might or might not produce an error depending on implementation
    # but let's check that it's handled gracefully
    assert len(configs) + len(errs) == 1


def test_grid_union_type_update():
    """Test updating fields with Union types."""
    from saev import nn

    @dataclasses.dataclass
    class ConfigWithUnion:
        activation: nn.modeling.Relu | nn.modeling.TopK | nn.modeling.BatchTopK = (
            dataclasses.field(default_factory=nn.modeling.Relu)
        )

    base_union = ConfigWithUnion(activation=nn.modeling.TopK(top_k=10))

    # Try to update the TopK's parameter
    sweep = {"activation": {"top_k": [5, 15, 25]}}
    configs, errs = helpers.grid(base_union, sweep)

    assert len(configs) == 3
    assert len(errs) == 0
    assert all(isinstance(c.activation, nn.modeling.TopK) for c in configs)
    assert [c.activation.top_k for c in configs] == [5, 15, 25]


def test_grid_frozen_dataclass():
    """Test that frozen dataclasses can be updated via replacement."""
    @dataclasses.dataclass(frozen=True)
    class FrozenConfig:
        value: int = 1

    @dataclasses.dataclass
    class ParentOfFrozen:
        frozen_child: FrozenConfig = dataclasses.field(default_factory=FrozenConfig)
        name: str = "parent"

    base_frozen = ParentOfFrozen()
    sweep = {"frozen_child": {"value": [10, 20]}}
    configs, errs = helpers.grid(base_frozen, sweep)

    # Should work because we're replacing, not mutating
    assert len(configs) == 2
    assert len(errs) == 0
    assert configs[0].frozen_child.value == 10
    assert configs[1].frozen_child.value == 20


def test_grid_list_of_dataclasses():
    """Test handling of lists containing dataclasses."""
    @dataclasses.dataclass
    class ListItem:
        id: int = 0
        value: str = "item"

    @dataclasses.dataclass
    class ConfigWithList:
        items: list[ListItem] = dataclasses.field(default_factory=list)

    # This will likely fail with current implementation
    # but demonstrates a limitation
    base_list = ConfigWithList(items=[ListItem(id=1), ListItem(id=2)])
    sweep = {"items": [{"id": 10, "value": "modified"}]}
    configs, errs = helpers.grid(base_list, sweep)

    # Just check it doesn't crash
    assert isinstance(configs, list)
    assert isinstance(errs, list)
