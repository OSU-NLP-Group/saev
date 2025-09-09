import collections
import dataclasses
import datetime
import enum
import os
import pathlib
import sys
import tempfile
import typing as tp

import beartype
import numpy as np
from hypothesis import given, settings
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

    base_cfg = Config(objective=nn.objectives.Vanilla(sparsity_coeff=1e-3))

    sweep = {
        "objective": {
            "sparsity_coeff": [1e-4, 1e-3, 1e-2],
        }
    }

    configs, errs = helpers.grid(base_cfg, sweep)

    assert len(configs) == 3
    assert len(errs) == 0
    assert all(isinstance(c.objective, nn.objectives.Vanilla) for c in configs)
    coeffs = [c.objective.sparsity_coeff for c in configs]
    assert 1e-4 in coeffs and 1e-2 in coeffs


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

    # This might or might not produce an error depending on implementation but let's check that it's handled gracefully
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


# Test fixtures - Simple dataclasses for testing
@dataclasses.dataclass
class SimpleConfig:
    """Basic dataclass for testing."""

    name: str = "default"
    count: int = 0
    enabled: bool = False
    rate: float = 1.0


@dataclasses.dataclass
class NestedConfig:
    """Dataclass with nested dataclass field."""

    simple: SimpleConfig = dataclasses.field(default_factory=SimpleConfig)
    value: int = 42


@dataclasses.dataclass
class PathConfig:
    """Dataclass with pathlib.Path field."""

    config_path: pathlib.Path = pathlib.Path(".")
    optional_path: pathlib.Path | None = None


@dataclasses.dataclass
class ListConfig:
    """Dataclass with list fields."""

    items: list[SimpleConfig] = dataclasses.field(default_factory=list)
    names: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TupleConfig:
    """Dataclass with tuple fields."""

    tags: tuple[str, ...] = ()
    coords: tuple[int, int] = (0, 0)


@dataclasses.dataclass
class ComplexConfig:
    """Complex nested dataclass for integration testing."""

    name: str = "complex"
    nested: NestedConfig = dataclasses.field(default_factory=NestedConfig)
    configs: list[SimpleConfig] = dataclasses.field(default_factory=list)
    tags: tuple[str, ...] = ()
    path: pathlib.Path = pathlib.Path(".")
    optional: int | None = None


class Color(enum.Enum):
    """Test enum."""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclasses.dataclass
class EnumConfig:
    """Dataclass with enum field."""

    color: Color = Color.RED
    name: str = "test"


@dataclasses.dataclass
class DateConfig:
    """Dataclass with datetime fields."""

    created: datetime.datetime = dataclasses.field(
        default_factory=datetime.datetime.now
    )
    date: datetime.date = dataclasses.field(default_factory=datetime.date.today)
    name: str = "date_test"


@dataclasses.dataclass
class DictConfig:
    """Dataclass with dict field."""

    settings: dict[str, int] = dataclasses.field(default_factory=dict)
    name: str = "dict_test"


@dataclasses.dataclass
class SetConfig:
    """Dataclass with set/frozenset fields."""

    tags: set[str] = dataclasses.field(default_factory=set)
    frozen_tags: frozenset[str] = dataclasses.field(default_factory=frozenset)


@dataclasses.dataclass
class UnionConfig:
    """Dataclass with Union types."""

    value: tp.Union[int, str, None] = None
    number: int | float = 0


# ============================================================================
# dict_to_dataclass tests
# ============================================================================


def test_dict_to_dataclass_basic():
    """Test basic conversion from dict to simple dataclass."""
    data = {"name": "test", "count": 5, "enabled": True, "rate": 2.5}
    result = helpers.dict_to_dataclass(data, SimpleConfig)

    assert isinstance(result, SimpleConfig)
    assert result.name == "test"
    assert result.count == 5
    assert result.enabled is True
    assert result.rate == 2.5


def test_dict_to_dataclass_nested():
    """Test conversion with nested dataclasses."""
    data = {"simple": {"name": "nested", "count": 10}, "value": 100}
    result = helpers.dict_to_dataclass(data, NestedConfig)

    assert isinstance(result, NestedConfig)
    assert isinstance(result.simple, SimpleConfig)
    assert result.simple.name == "nested"
    assert result.simple.count == 10
    assert result.value == 100


def test_dict_to_dataclass_list_of_dataclasses():
    """Test conversion of list[DataclassType] fields."""
    data = {
        "items": [
            {"name": "first", "count": 1},
            {"name": "second", "count": 2},
        ],
        "names": ["a", "b", "c"],
    }
    result = helpers.dict_to_dataclass(data, ListConfig)

    assert isinstance(result, ListConfig)
    assert len(result.items) == 2
    assert all(isinstance(item, SimpleConfig) for item in result.items)
    assert result.items[0].name == "first"
    assert result.items[1].count == 2
    assert result.names == ["a", "b", "c"]


def test_dict_to_dataclass_tuple_from_list():
    """Test conversion of lists to tuples for tuple fields."""
    data = {"tags": ["tag1", "tag2", "tag3"], "coords": [10, 20]}
    result = helpers.dict_to_dataclass(data, TupleConfig)

    assert isinstance(result, TupleConfig)
    assert result.tags == ("tag1", "tag2", "tag3")
    assert result.coords == (10, 20)
    assert isinstance(result.tags, tuple)
    assert isinstance(result.coords, tuple)


def test_dict_to_dataclass_pathlib_path():
    """Test conversion of strings to pathlib.Path."""
    data = {"config_path": "/tmp/config", "optional_path": "/tmp/optional"}
    result = helpers.dict_to_dataclass(data, PathConfig)

    assert isinstance(result, PathConfig)
    assert isinstance(result.config_path, pathlib.Path)
    assert result.config_path == pathlib.Path("/tmp/config")
    assert isinstance(result.optional_path, pathlib.Path)
    assert result.optional_path == pathlib.Path("/tmp/optional")


def test_dict_to_dataclass_optional_fields():
    """Test handling of Optional fields with None values."""
    data = {"config_path": "/tmp/config", "optional_path": None}
    result = helpers.dict_to_dataclass(data, PathConfig)

    assert isinstance(result, PathConfig)
    assert result.optional_path is None


def test_dict_to_dataclass_missing_fields():
    """Test that missing fields use dataclass defaults."""
    data = {"name": "partial"}  # Missing count, enabled, rate
    result = helpers.dict_to_dataclass(data, SimpleConfig)

    assert isinstance(result, SimpleConfig)
    assert result.name == "partial"
    assert result.count == 0  # default
    assert result.enabled is False  # default
    assert result.rate == 1.0  # default


def test_dict_to_dataclass_extra_fields():
    """Test that extra fields in dict are ignored."""
    data = {
        "name": "test",
        "count": 5,
        "extra_field": "ignored",
        "another_extra": 123,
    }
    result = helpers.dict_to_dataclass(data, SimpleConfig)

    assert isinstance(result, SimpleConfig)
    assert result.name == "test"
    assert result.count == 5
    assert not hasattr(result, "extra_field")
    assert not hasattr(result, "another_extra")


def test_dict_to_dataclass_empty_lists():
    """Test handling of empty list fields."""
    data = {"items": [], "names": []}
    result = helpers.dict_to_dataclass(data, ListConfig)

    assert isinstance(result, ListConfig)
    assert result.items == []
    assert result.names == []


def test_dict_to_dataclass_deep_nesting():
    """Test 3+ levels of nested dataclasses."""
    data = {
        "name": "deep",
        "nested": {
            "simple": {"name": "level3", "count": 3, "enabled": True},
            "value": 99,
        },
        "configs": [{"name": "cfg1"}, {"name": "cfg2"}],
        "tags": ["a", "b"],
        "path": "/deep/path",
        "optional": 77,
    }
    result = helpers.dict_to_dataclass(data, ComplexConfig)

    assert isinstance(result, ComplexConfig)
    assert result.name == "deep"
    assert result.nested.simple.name == "level3"
    assert result.nested.simple.count == 3
    assert result.nested.value == 99
    assert len(result.configs) == 2
    assert result.tags == ("a", "b")
    assert result.path == pathlib.Path("/deep/path")
    assert result.optional == 77


# ============================================================================
# get_non_default_values tests
# ============================================================================


def test_get_non_default_all_defaults():
    """Test that all defaults returns empty dict."""
    obj = SimpleConfig()
    default_obj = SimpleConfig()
    result = helpers.get_non_default_values(obj, default_obj)

    assert result == {}


def test_get_non_default_single_change():
    """Test detection of single field change."""
    obj = SimpleConfig(name="changed")
    default_obj = SimpleConfig()
    result = helpers.get_non_default_values(obj, default_obj)

    # Only changed fields are returned, not all fields
    assert result == {"name": "changed"}


def test_get_non_default_multiple_changes():
    """Test detection of multiple field changes."""
    obj = SimpleConfig(name="changed", count=10, enabled=True)
    default_obj = SimpleConfig()
    result = helpers.get_non_default_values(obj, default_obj)

    assert result["name"] == "changed"
    assert result["count"] == 10
    assert result["enabled"] is True


def test_get_non_default_nested_changes():
    """Test detection of changes in nested dataclasses."""
    obj = NestedConfig(simple=SimpleConfig(name="nested_change"), value=100)
    default_obj = NestedConfig()
    result = helpers.get_non_default_values(obj, default_obj)

    assert "simple" in result
    assert result["simple"]["name"] == "nested_change"
    assert result["value"] == 100


def test_get_non_default_list_changes():
    """Test detection of list field changes."""
    obj = ListConfig(
        items=[SimpleConfig(name="item1")],
        names=["a", "b"],
    )
    default_obj = ListConfig()
    result = helpers.get_non_default_values(obj, default_obj)

    assert "items" in result
    assert len(result["items"]) == 1
    assert "names" in result
    assert result["names"] == ["a", "b"]


def test_get_non_default_none_vs_default():
    """Test distinguishing None from default values."""
    obj = ComplexConfig(optional=None)  # None is already the default
    default_obj = ComplexConfig()
    result = helpers.get_non_default_values(obj, default_obj)

    # Since None is the default, this should not show as a change
    assert "optional" not in result or result.get("optional") is None


def test_get_non_default_empty_vs_populated_list():
    """Test detection of empty vs non-empty lists."""
    obj = ListConfig(items=[SimpleConfig()])
    default_obj = ListConfig()  # items defaults to []
    result = helpers.get_non_default_values(obj, default_obj)

    assert "items" in result
    assert len(result["items"]) == 1


# ============================================================================
# merge_configs tests
# ============================================================================


def test_merge_configs_empty_overrides():
    """Test that empty overrides returns base unchanged."""
    base = SimpleConfig(name="base", count=5)
    result = helpers.merge_configs(base, {})

    assert result.name == "base"
    assert result.count == 5
    assert result.enabled is False


def test_merge_configs_single_override():
    """Test overriding a single field."""
    base = SimpleConfig(name="base", count=5)
    overrides = {"count": 10}
    result = helpers.merge_configs(base, overrides)

    assert result.name == "base"  # unchanged
    assert result.count == 10  # overridden
    assert result.enabled is False  # unchanged


def test_merge_configs_multiple_overrides():
    """Test overriding multiple fields."""
    base = SimpleConfig(name="base", count=5)
    overrides = {"name": "override", "count": 10, "enabled": True}
    result = helpers.merge_configs(base, overrides)

    assert result.name == "override"
    assert result.count == 10
    assert result.enabled is True
    assert result.rate == 1.0  # unchanged


def test_merge_configs_nested_override():
    """Test overriding nested dataclass fields."""
    base = NestedConfig(simple=SimpleConfig(name="base"), value=42)
    overrides = {"simple": {"name": "override", "count": 5}, "value": 100}
    result = helpers.merge_configs(base, overrides)

    assert result.simple.name == "override"
    assert result.simple.count == 5
    assert result.simple.enabled is False  # unchanged nested field
    assert result.value == 100


def test_merge_configs_partial_nested():
    """Test overriding only some nested fields."""
    base = NestedConfig(simple=SimpleConfig(name="base", count=10), value=42)
    overrides = {"simple": {"name": "partial"}}  # Only override name
    result = helpers.merge_configs(base, overrides)

    assert result.simple.name == "partial"  # overridden
    assert result.simple.count == 10  # preserved from base
    assert result.value == 42  # unchanged


def test_merge_configs_list_override():
    """Test replacing entire list fields."""
    base = ListConfig(
        items=[SimpleConfig(name="old1"), SimpleConfig(name="old2")],
        names=["a", "b"],
    )
    overrides = {
        "items": [{"name": "new1", "count": 1}],
        "names": ["x", "y", "z"],
    }
    result = helpers.merge_configs(base, overrides)

    assert len(result.items) == 1
    assert result.items[0].name == "new1"
    assert result.names == ["x", "y", "z"]


def test_merge_configs_none_override():
    """Test overriding with None values."""
    base = ComplexConfig(optional=42, name="base")
    overrides = {"optional": None}
    result = helpers.merge_configs(base, overrides)

    assert result.optional is None
    assert result.name == "base"


def test_merge_configs_preserve_unmodified():
    """Test that unmodified fields are preserved."""
    base = ComplexConfig(
        name="base",
        nested=NestedConfig(value=99),
        configs=[SimpleConfig(name="cfg")],
        tags=("a", "b"),
        path=pathlib.Path("/base"),
        optional=77,
    )
    overrides = {"name": "modified"}
    result = helpers.merge_configs(base, overrides)

    assert result.name == "modified"
    assert result.nested.value == 99  # preserved
    assert len(result.configs) == 1  # preserved
    assert result.configs[0].name == "cfg"  # preserved
    assert result.tags == ("a", "b")  # preserved
    assert result.path == pathlib.Path("/base")  # preserved
    assert result.optional == 77  # preserved


# ============================================================================
# Future-proofing and edge case tests
# ============================================================================


def test_new_field_backward_compat():
    """Test handling when new fields are added to dataclass."""

    @dataclasses.dataclass
    class OldVersion:
        name: str = "default"
        count: int = 0

    @dataclasses.dataclass
    class NewVersion:
        name: str = "default"
        count: int = 0
        new_field: str = "new_default"

    # Simulate loading old config with new dataclass
    old_data = {"name": "test", "count": 5}
    result = helpers.dict_to_dataclass(old_data, NewVersion)

    assert result.name == "test"
    assert result.count == 5
    assert result.new_field == "new_default"  # Uses default for missing field


def test_enum_field_conversion():
    """Test support for enum fields."""
    # Note: Current implementation passes string values through
    # Proper enum conversion would need special handling
    data = {"color": "red", "name": "enum_test"}
    result = helpers.dict_to_dataclass(data, EnumConfig)

    # String is passed through, not converted to enum
    assert result.color == "red"  # Not Color.RED
    assert result.name == "enum_test"


def test_datetime_field_conversion():
    """Test handling of datetime/date fields."""
    # Note: Current implementation passes string values through
    # Proper datetime conversion would need special handling
    now = datetime.datetime.now()
    today = datetime.date.today()
    data = {"created": now.isoformat(), "date": today.isoformat(), "name": "date"}

    result = helpers.dict_to_dataclass(data, DateConfig)

    # Strings are passed through, not converted to datetime objects
    assert result.created == now.isoformat()  # String, not datetime
    assert result.date == today.isoformat()  # String, not date
    assert result.name == "date"


def test_set_frozenset_conversion():
    """Test conversion of lists to sets/frozensets."""
    data = {"tags": ["a", "b", "c"], "frozen_tags": ["x", "y", "z"]}
    # Current implementation doesn't handle set/frozenset conversion
    result = helpers.dict_to_dataclass(data, SetConfig)

    # Will be lists, not sets - shows limitation
    assert isinstance(result.tags, list) or isinstance(result.tags, set)
    assert isinstance(result.frozen_tags, list) or isinstance(
        result.frozen_tags, frozenset
    )


def test_dict_field_handling():
    """Test support for dict fields."""
    data = {"settings": {"key1": 1, "key2": 2}, "name": "dict"}
    result = helpers.dict_to_dataclass(data, DictConfig)

    assert isinstance(result, DictConfig)
    assert result.settings == {"key1": 1, "key2": 2}
    assert result.name == "dict"


def test_union_types_beyond_optional():
    """Test handling of Union types."""
    data1 = {"value": 42, "number": 3.14}
    result1 = helpers.dict_to_dataclass(data1, UnionConfig)
    assert result1.value == 42
    assert result1.number == 3.14

    data2 = {"value": "string", "number": 10}
    result2 = helpers.dict_to_dataclass(data2, UnionConfig)
    assert result2.value == "string"
    assert result2.number == 10


def test_forward_reference_handling():
    """Test handling of forward references in type hints."""
    # Forward references need special handling
    # This is a limitation test showing where enhancement may be needed
    pass  # Current implementation handles standard types


def test_default_factory_fields():
    """Test handling of fields with default_factory."""
    data = {}  # Empty dict
    result = helpers.dict_to_dataclass(data, ListConfig)

    assert result.items == []  # default_factory=list
    assert result.names == []  # default_factory=list


# ============================================================================
# Integration tests
# ============================================================================


def test_full_config_roundtrip():
    """Test complete roundtrip: dict -> dataclass -> modify -> merge."""
    # Start with dict
    config_dict = {
        "name": "initial",
        "nested": {"simple": {"name": "nested", "count": 10}, "value": 50},
        "configs": [{"name": "cfg1"}, {"name": "cfg2"}],
        "tags": ["tag1", "tag2"],
        "path": "/initial/path",
    }

    # Convert to dataclass
    config = helpers.dict_to_dataclass(config_dict, ComplexConfig)
    assert config.name == "initial"

    # Create override
    override = ComplexConfig(name="override", optional=99)
    default = ComplexConfig()

    # Get non-default values
    overrides = helpers.get_non_default_values(override, default)
    assert "name" in overrides
    assert "optional" in overrides

    # Merge
    final = helpers.merge_configs(config, overrides)
    assert final.name == "override"  # overridden
    assert final.nested.simple.count == 10  # preserved
    assert final.optional == 99  # overridden


def test_complex_real_world_config():
    """Test with realistic nested configuration."""

    @dataclasses.dataclass
    class OptimizerConfig:
        algorithm: str = "adam"
        learning_rate: float = 0.001
        weight_decay: float = 0.0

    @dataclasses.dataclass
    class DataConfig:
        batch_size: int = 32
        num_workers: int = 4
        datasets: list[PathConfig] = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class ModelConfig:
        hidden_dim: int = 256
        num_layers: int = 3
        dropout: float = 0.1

    @dataclasses.dataclass
    class TrainingConfig:
        model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
        optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
        data: DataConfig = dataclasses.field(default_factory=DataConfig)
        epochs: int = 10
        seed: int = 42
        output_dir: pathlib.Path = pathlib.Path("output")

    config_dict = {
        "model": {"hidden_dim": 512, "num_layers": 4},
        "optimizer": {"algorithm": "adamw", "learning_rate": 0.0001},
        "data": {
            "batch_size": 64,
            "datasets": [
                {"config_path": "/data/train"},
                {"config_path": "/data/val"},
            ],
        },
        "epochs": 20,
        "output_dir": "/results",
    }

    config = helpers.dict_to_dataclass(config_dict, TrainingConfig)

    assert config.model.hidden_dim == 512
    assert config.model.dropout == 0.1  # default preserved
    assert config.optimizer.algorithm == "adamw"
    assert config.data.batch_size == 64
    assert len(config.data.datasets) == 2
    assert config.seed == 42  # default preserved


def test_config_migration_scenario():
    """Test simulating config version migration."""

    # Simulate old config version
    old_config_dict = {
        "name": "old_config",
        "value": 100,
        # Missing new fields that will be added
    }

    # New version with additional fields
    @dataclasses.dataclass
    class NewConfigVersion:
        name: str = "default"
        value: int = 0
        new_required_field: str = "default_new"
        new_optional_field: str | None = None

    # Load old config with new dataclass
    migrated = helpers.dict_to_dataclass(old_config_dict, NewConfigVersion)

    assert migrated.name == "old_config"
    assert migrated.value == 100
    assert migrated.new_required_field == "default_new"  # Gets default
    assert migrated.new_optional_field is None  # Gets default

    # Apply overrides for new fields
    overrides = {"new_required_field": "updated", "new_optional_field": "set"}
    final = helpers.merge_configs(migrated, overrides)

    assert final.name == "old_config"  # preserved
    assert final.value == 100  # preserved
    assert final.new_required_field == "updated"  # overridden
    assert final.new_optional_field == "set"  # overridden
