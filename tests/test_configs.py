import dataclasses
import pathlib
import typing as tp

import beartype

from saev.configs import dict_to_dataclass, load_cfgs, load_sweep
from saev.framework import train
from saev.nn import objectives


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SimpleConfig:
    value: int = 1


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class NestedConfig:
    inner_value: int = 5
    other_inner_value: int = 20


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ConfigWithNested:
    nested: NestedConfig = dataclasses.field(default_factory=NestedConfig)
    outer_value: int = 10


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ItemConfig:
    item_id: int = 0
    name: str = "default"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ConfigWithList:
    items: list[ItemConfig] = dataclasses.field(default_factory=list)
    count: int = 0


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ConfigWithTuple:
    tags: tuple[str, ...] = ()
    value: int = 1


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ConfigWithPath:
    path: pathlib.Path = pathlib.Path("/default")
    name: str = "test"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ConfigWithOptionalPath:
    path: pathlib.Path | None = None
    name: str = "test"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ConfigWithSeed:
    seed: int = 0
    value: int = 1


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class NestedConfigWithSeed:
    inner_seed: int = 0
    inner_value: int = 5


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ConfigWithNestedSeed:
    seed: int = 0
    nested: NestedConfigWithSeed = dataclasses.field(
        default_factory=NestedConfigWithSeed
    )
    value: int = 10


def test_load_cfgs_returns_tuple():
    override = SimpleConfig(value=2)
    default = SimpleConfig()
    dct = {}

    result = load_cfgs(override, default=default, sweep_dcts=[dct])

    assert isinstance(result, tuple)


def test_load_cfgs_returns_correct_types():
    override = SimpleConfig(value=2)
    default = SimpleConfig()
    dct = {}

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=[dct])

    assert isinstance(cfgs, list)
    assert isinstance(errs, list)
    assert all(isinstance(cfg, SimpleConfig) for cfg in cfgs)
    assert all(isinstance(err, str) for err in errs)


def test_load_cfgs_override_overrides_default():
    override = SimpleConfig(value=10)
    default = SimpleConfig(value=1)

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=[])

    assert len(cfgs) == 1
    assert cfgs[0].value == 10


def test_load_cfgs_expands_dict_sweep():
    override = SimpleConfig()
    default = SimpleConfig()
    sweep_dcts = [{"value": 1}, {"value": 2}, {"value": 3}]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(cfgs) == 3
    assert cfgs[0].value == 1
    assert cfgs[1].value == 2
    assert cfgs[2].value == 3


def test_load_cfgs_nested_dataclass():
    override = ConfigWithNested()
    default = ConfigWithNested()
    sweep_dcts = [
        {"nested": {"inner_value": 1}},
        {"nested": {"inner_value": 2}},
        {"nested": {"inner_value": 3}},
    ]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(cfgs) == 3
    assert cfgs[0].nested.inner_value == 1
    assert cfgs[1].nested.inner_value == 2
    assert cfgs[2].nested.inner_value == 3
    assert all(cfg.outer_value == 10 for cfg in cfgs)


def test_load_cfgs_nested_and_outer_sweep():
    override = ConfigWithNested()
    default = ConfigWithNested()
    sweep_dcts = [
        {"nested": {"inner_value": 1}, "outer_value": 20},
        {"nested": {"inner_value": 1}, "outer_value": 30},
        {"nested": {"inner_value": 2}, "outer_value": 20},
        {"nested": {"inner_value": 2}, "outer_value": 30},
    ]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(cfgs) == 4
    assert cfgs[0].nested.inner_value == 1
    assert cfgs[0].outer_value == 20
    assert cfgs[1].nested.inner_value == 1
    assert cfgs[1].outer_value == 30
    assert cfgs[2].nested.inner_value == 2
    assert cfgs[2].outer_value == 20
    assert cfgs[3].nested.inner_value == 2
    assert cfgs[3].outer_value == 30


def test_load_cfgs_tuple_field():
    override = ConfigWithTuple()
    default = ConfigWithTuple()
    sweep_dcts = [{"tags": ("a", "b")}, {"tags": ("c", "d", "e")}]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(cfgs) == 2
    assert cfgs[0].tags == ("a", "b")
    assert cfgs[1].tags == ("c", "d", "e")


def test_load_cfgs_path_field():
    override = ConfigWithPath()
    default = ConfigWithPath()
    sweep_dcts = [{"path": "/tmp/a"}, {"path": "/tmp/b"}]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(cfgs) == 2
    assert cfgs[0].path == pathlib.Path("/tmp/a")
    assert cfgs[1].path == pathlib.Path("/tmp/b")
    assert isinstance(cfgs[0].path, pathlib.Path)
    assert isinstance(cfgs[1].path, pathlib.Path)


def test_load_cfgs_optional_path_field():
    override = ConfigWithOptionalPath()
    default = ConfigWithOptionalPath()
    sweep_dcts = [{"path": "/tmp/a"}, {"path": None}]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(cfgs) == 2
    assert cfgs[0].path == pathlib.Path("/tmp/a")
    assert cfgs[1].path is None


def test_load_cfgs_list_of_dataclasses():
    override = ConfigWithList()
    default = ConfigWithList()
    sweep_dcts = [
        {"items": [{"item_id": 1, "name": "first"}]},
        {"items": [{"item_id": 2, "name": "second"}, {"item_id": 3, "name": "third"}]},
    ]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(cfgs) == 2
    assert len(cfgs[0].items) == 1
    assert cfgs[0].items[0].item_id == 1
    assert cfgs[0].items[0].name == "first"
    assert len(cfgs[1].items) == 2
    assert cfgs[1].items[0].item_id == 2
    assert cfgs[1].items[1].item_id == 3


def test_load_cfgs_override_beats_dict():
    override = SimpleConfig(value=100)
    default = SimpleConfig(value=1)
    sweep_dcts = [{"value": 1}, {"value": 2}, {"value": 3}]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert all(cfg.value == 100 for cfg in cfgs)


def test_load_cfgs_override_nested_beats_dict():
    override = ConfigWithNested(nested=NestedConfig(inner_value=999))
    default = ConfigWithNested()
    sweep_dcts = [
        {"nested": {"inner_value": 1}},
        {"nested": {"inner_value": 2}},
        {"nested": {"inner_value": 3}},
    ]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert all(cfg.nested.inner_value == 999 for cfg in cfgs)


def test_load_cfgs_seed_increments():
    override = ConfigWithSeed()
    default = ConfigWithSeed()
    sweep_dcts = [{"value": 10}, {"value": 20}, {"value": 30}]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(cfgs) == 3
    assert cfgs[0].seed == 0
    assert cfgs[1].seed == 1
    assert cfgs[2].seed == 2


def test_load_cfgs_nonexistent_field_creates_error():
    override = SimpleConfig()
    default = SimpleConfig()
    sweep_dcts = [
        {"nonexistent_field": 1},
        {"nonexistent_field": 2},
        {"nonexistent_field": 3},
    ]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(errs) > 0


def test_load_cfgs_mixed_types_same_field():
    override = SimpleConfig()
    default = SimpleConfig()
    sweep_dcts = [{"value": 1}, {"value": "not_an_int"}, {"value": 3}]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(errs) >= 1


def test_load_cfgs_deeply_nested():
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Level3:
        val: int = 0

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Level2:
        level3: Level3 = dataclasses.field(default_factory=Level3)

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Level1:
        level2: Level2 = dataclasses.field(default_factory=Level2)

    override = Level1()
    default = Level1()
    sweep_dcts = [
        {"level2": {"level3": {"val": 1}}},
        {"level2": {"level3": {"val": 2}}},
        {"level2": {"level3": {"val": 3}}},
    ]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(cfgs) == 3
    assert cfgs[0].level2.level3.val == 1
    assert cfgs[1].level2.level3.val == 2
    assert cfgs[2].level2.level3.val == 3


def test_load_cfgs_nested_override_filters_empty_nested_dict():
    """Test line 59: when filtered nested dict is empty, it's not included."""
    override = ConfigWithNested(nested=NestedConfig(inner_value=100))
    default = ConfigWithNested()
    sweep_dcts = [
        {"nested": {"inner_value": 1}, "outer_value": 10},
        {"nested": {"inner_value": 2}, "outer_value": 10},
        {"nested": {"inner_value": 3}, "outer_value": 10},
        {"nested": {"inner_value": 1}, "outer_value": 20},
        {"nested": {"inner_value": 2}, "outer_value": 20},
        {"nested": {"inner_value": 3}, "outer_value": 20},
    ]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    # Should only sweep outer_value since nested.inner_value is overridden
    assert len(cfgs) == 6
    assert all(cfg.nested.inner_value == 100 for cfg in cfgs)
    assert cfgs[0].outer_value == 10
    assert cfgs[1].outer_value == 10
    assert cfgs[2].outer_value == 10
    assert cfgs[3].outer_value == 20
    assert cfgs[4].outer_value == 20
    assert cfgs[3].outer_value == 20


def test_load_cfgs_nested_override_only_nested_field():
    """When"""
    override = ConfigWithNested(nested=NestedConfig(inner_value=100))
    default = ConfigWithNested()
    sweep_dcts = [
        {"nested": {"other_inner_value": 1}, "outer_value": 10},
        {"nested": {"other_inner_value": 2}, "outer_value": 10},
        {"nested": {"other_inner_value": 3}, "outer_value": 10},
        {"nested": {"other_inner_value": 1}, "outer_value": 20},
        {"nested": {"other_inner_value": 2}, "outer_value": 20},
        {"nested": {"other_inner_value": 3}, "outer_value": 20},
    ]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    # Should only sweep outer_value since nested.inner_value is overridden
    assert len(cfgs) == 6
    assert all(cfg.nested.inner_value == 100 for cfg in cfgs)
    assert cfgs[0].outer_value == 10
    assert cfgs[1].outer_value == 10
    assert cfgs[2].outer_value == 10
    assert cfgs[3].outer_value == 20
    assert cfgs[4].outer_value == 20
    assert cfgs[5].outer_value == 20
    assert cfgs[0].nested.other_inner_value == 1
    assert cfgs[1].nested.other_inner_value == 2
    assert cfgs[2].nested.other_inner_value == 3
    assert cfgs[3].nested.other_inner_value == 1
    assert cfgs[4].nested.other_inner_value == 2
    assert cfgs[5].nested.other_inner_value == 3


def test_load_cfgs_dataclass_field_in_dict():
    """Test line 128: dict_to_dataclass called from _convert_value."""

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Inner:
        x: int = 0

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Outer:
        inner: Inner = dataclasses.field(default_factory=Inner)

    override = Outer()
    default = Outer()
    sweep_dcts = [{"inner": {"x": 1}}, {"inner": {"x": 2}}]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(cfgs) == 2
    assert cfgs[0].inner.x == 1
    assert cfgs[1].inner.x == 2


def test_load_cfgs_typing_union_path():
    """Test line 133: typing.Union style optional path (Python 3.9 style)."""

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class ConfigWithTypingUnionPath:
        path: tp.Union[pathlib.Path, None] = None

    override = ConfigWithTypingUnionPath()
    default = ConfigWithTypingUnionPath()
    sweep_dcts = [{"path": "/tmp/x"}]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert len(cfgs) == 1
    assert cfgs[0].path == pathlib.Path("/tmp/x")


def test_load_cfgs_nested_seed_increment():
    """Test lines 168-169: nested seed auto-increment when sweeping nested field."""

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class NestedWithSeed:
        seed: int = 0
        value: int = 5

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class OuterWithNestedSeed:
        seed: int = 0
        nested: NestedWithSeed = dataclasses.field(default_factory=NestedWithSeed)

    override = OuterWithNestedSeed()
    default = OuterWithNestedSeed()
    sweep_dcts = [
        {"nested": {"value": 1}},
        {"nested": {"value": 2}},
        {"nested": {"value": 3}},
    ]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    # Nested seed should auto-increment
    assert len(cfgs) == 3
    assert cfgs[0].nested.seed == 0
    assert cfgs[1].nested.seed == 1
    assert cfgs[2].nested.seed == 2


def test_dict_to_dataclass_with_missing_fields():
    """Test line 191: dict_to_dataclass when field is not in data."""

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Config:
        a: int = 1
        b: int = 2

    result = dict_to_dataclass({"a": 10}, Config)

    assert result.a == 10
    assert result.b == 2


def test_dict_to_dataclass_non_dataclass():
    """Test line 184: dict_to_dataclass when cls is not a dataclass."""

    result = dict_to_dataclass({"a": 1}, dict)

    assert result == {"a": 1}


def test_load_cfgs_all_nested_fields_overridden():
    """Test line 59: nested dict is empty after filtering all overridden fields."""
    override = ConfigWithNested(nested=NestedConfig(inner_value=999))
    default = ConfigWithNested()
    sweep_dcts = [
        {"nested": {"inner_value": 1}, "outer_value": 10},
        {"nested": {"inner_value": 2}, "outer_value": 10},
        {"nested": {"inner_value": 3}, "outer_value": 10},
        {"nested": {"inner_value": 1}, "outer_value": 20},
        {"nested": {"inner_value": 2}, "outer_value": 20},
        {"nested": {"inner_value": 3}, "outer_value": 20},
    ]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    # Should sweep outer_value only, nested should be filtered out entirely
    assert len(cfgs) == 6
    assert all(cfg.nested.inner_value == 999 for cfg in cfgs)
    assert cfgs[0].outer_value == 10
    assert cfgs[1].outer_value == 10


def test_load_cfgs_from_python_sweep():
    """Integration test: load sweep from Python file with make_cfgs()."""

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class ObjectiveConfig:
        sparsity_coeff: float = 1e-3

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class TrainConfig:
        lr: float = 1e-4
        objective: ObjectiveConfig = dataclasses.field(default_factory=ObjectiveConfig)

    # Load the sweep file
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "simple.py"
    sweep_dcts = load_sweep(sweep_fpath)

    # Test with first sweep config as dct
    override = TrainConfig()
    default = TrainConfig()
    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=[sweep_dcts[0]])

    assert len(cfgs) == 1
    assert isinstance(cfgs[0], TrainConfig)
    assert cfgs[0].lr == 1e-4
    assert cfgs[0].objective.sparsity_coeff == 4e-4
    assert len(errs) == 0


def test_load_cfgs_sweep_overrides_objective_fields():
    override = dataclasses.replace(train.Config(), objective=objectives.Matryoshka())
    default = train.Config()
    sweep_dcts = [{"objective": {"n_prefixes": 20}}]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert not errs
    assert len(cfgs) == 1
    assert isinstance(cfgs[0].objective, objectives.Matryoshka)
    assert cfgs[0].objective.n_prefixes == 20


def test_load_sweep_missing_function():
    """Test load_sweep() with missing make_cfgs() function."""
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "no_function.py"
    result = load_sweep(sweep_fpath)

    assert result == []


def test_load_sweep_raises_error():
    """Test load_sweep() when make_cfgs() raises an exception."""
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "raises_error.py"
    result = load_sweep(sweep_fpath)

    assert result == []


def test_load_sweep_wrong_return_type():
    """Test load_sweep() when make_cfgs() returns wrong type."""
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "wrong_return_type.py"
    result = load_sweep(sweep_fpath)

    assert result == []


def test_load_sweep_empty():
    """Test load_sweep() with empty list."""
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "empty.py"
    sweep_dcts = load_sweep(sweep_fpath)

    assert sweep_dcts == []


def test_load_sweep_with_imports():
    """Test load_sweep() with imports and helper functions."""
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "with_imports.py"
    sweep_dcts = load_sweep(sweep_fpath)

    assert len(sweep_dcts) == 4
    assert sweep_dcts[0] == {"lr": 1e-4, "sparsity": 4e-4}
    assert sweep_dcts[1] == {"lr": 1e-4, "sparsity": 8e-4}
    assert sweep_dcts[2] == {"lr": 3e-4, "sparsity": 4e-4}
    assert sweep_dcts[3] == {"lr": 3e-4, "sparsity": 8e-4}


def test_load_sweep_invalid_syntax():
    """Test load_sweep() with invalid Python syntax."""
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "invalid_syntax.py"
    result = load_sweep(sweep_fpath)

    assert result == []


def test_load_sweep_nonexistent_file():
    """Test load_sweep() with nonexistent file."""
    sweep_fpath = pathlib.Path(__file__).parent / "sweeps" / "does_not_exist.py"
    result = load_sweep(sweep_fpath)

    assert result == []


def test_load_cfgs_union_dataclass_field_sweep():
    """Test sweeping union dataclass fields when type is set via override.

    This mimics the actual usage pattern:
    1. User passes `sae.activation:topk` to tyro, setting the type to TopK
    2. Sweep dict provides only the parameters: `{"activation": {"top_k": 512}}`
    3. load_cfgs should update the existing TopK instance with the new top_k value

    This is the pattern used in fishvista_train_topk.py.
    """
    from saev.nn import modeling

    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class ConfigWithActivation:
        """Mimics SparseAutoencoderConfig with union activation field."""

        activation: modeling.ActivationConfig = modeling.Relu()
        d_sae: int = 16384

    # The key difference from a broken implicit test: override already has TopK set
    # (simulating tyro parsing `sae.activation:topk`)
    override = ConfigWithActivation(activation=modeling.TopK(top_k=32))
    default = ConfigWithActivation()

    # Sweep dict provides only the parameter to update
    sweep_dcts = [
        {"activation": {"top_k": 32}},
        {"activation": {"top_k": 128}},
        {"activation": {"top_k": 512}},
    ]

    cfgs, errs = load_cfgs(override, default=default, sweep_dcts=sweep_dcts)

    assert not errs, f"Expected no errors, got: {errs}"
    assert len(cfgs) == 3

    # All should remain TopK instances (not converted to something else)
    assert all(isinstance(cfg.activation, modeling.TopK) for cfg in cfgs), (
        f"Expected all TopK, got {[type(cfg.activation) for cfg in cfgs]}"
    )

    # And the top_k values should be updated correctly
    assert cfgs[0].activation.top_k == 32
    assert cfgs[1].activation.top_k == 128
    assert cfgs[2].activation.top_k == 512
