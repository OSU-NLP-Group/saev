import dataclasses
import pathlib

from saev.configs import expand, load_cfgs


@dataclasses.dataclass
class SimpleConfig:
    value: int = 1


@dataclasses.dataclass
class NestedConfig:
    inner_value: int = 5


@dataclasses.dataclass
class ConfigWithNested:
    nested: NestedConfig = dataclasses.field(default_factory=NestedConfig)
    outer_value: int = 10


@dataclasses.dataclass
class ItemConfig:
    item_id: int = 0
    name: str = "default"


@dataclasses.dataclass
class ConfigWithList:
    items: list[ItemConfig] = dataclasses.field(default_factory=list)
    count: int = 0


@dataclasses.dataclass
class ConfigWithTuple:
    tags: tuple[str, ...] = ()
    value: int = 1


@dataclasses.dataclass
class ConfigWithPath:
    path: pathlib.Path = pathlib.Path("/default")
    name: str = "test"


@dataclasses.dataclass
class ConfigWithOptionalPath:
    path: pathlib.Path | None = None
    name: str = "test"


@dataclasses.dataclass
class ConfigWithSeed:
    seed: int = 0
    value: int = 1


@dataclasses.dataclass
class NestedConfigWithSeed:
    inner_seed: int = 0
    inner_value: int = 5


@dataclasses.dataclass
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

    result = load_cfgs(override, default=default, dct=dct)

    assert isinstance(result, tuple)


def test_load_cfgs_returns_correct_types():
    override = SimpleConfig(value=2)
    default = SimpleConfig()
    dct = {}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert isinstance(cfgs, list)
    assert isinstance(errs, list)
    assert all(isinstance(cfg, SimpleConfig) for cfg in cfgs)
    assert all(isinstance(err, str) for err in errs)


def test_load_cfgs_override_overrides_default():
    override = SimpleConfig(value=10)
    default = SimpleConfig(value=1)
    dct = {}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(cfgs) == 1
    assert cfgs[0].value == 10


def test_load_cfgs_expands_dict_sweep():
    override = SimpleConfig()
    default = SimpleConfig()
    dct = {"value": [1, 2, 3]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(cfgs) == 3
    assert cfgs[0].value == 1
    assert cfgs[1].value == 2
    assert cfgs[2].value == 3


def test_load_cfgs_nested_dataclass():
    override = ConfigWithNested()
    default = ConfigWithNested()
    dct = {"nested": {"inner_value": [1, 2, 3]}}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(cfgs) == 3
    assert cfgs[0].nested.inner_value == 1
    assert cfgs[1].nested.inner_value == 2
    assert cfgs[2].nested.inner_value == 3
    assert all(cfg.outer_value == 10 for cfg in cfgs)


def test_load_cfgs_nested_and_outer_sweep():
    override = ConfigWithNested()
    default = ConfigWithNested()
    dct = {"nested": {"inner_value": [1, 2]}, "outer_value": [20, 30]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

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
    dct = {"tags": [["a", "b"], ["c", "d", "e"]]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(cfgs) == 2
    assert cfgs[0].tags == ("a", "b")
    assert cfgs[1].tags == ("c", "d", "e")


def test_load_cfgs_path_field():
    override = ConfigWithPath()
    default = ConfigWithPath()
    dct = {"path": ["/tmp/a", "/tmp/b"]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(cfgs) == 2
    assert cfgs[0].path == pathlib.Path("/tmp/a")
    assert cfgs[1].path == pathlib.Path("/tmp/b")
    assert isinstance(cfgs[0].path, pathlib.Path)
    assert isinstance(cfgs[1].path, pathlib.Path)


def test_load_cfgs_optional_path_field():
    override = ConfigWithOptionalPath()
    default = ConfigWithOptionalPath()
    dct = {"path": ["/tmp/a", None]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(cfgs) == 2
    assert cfgs[0].path == pathlib.Path("/tmp/a")
    assert cfgs[1].path is None


def test_load_cfgs_list_of_dataclasses():
    override = ConfigWithList()
    default = ConfigWithList()
    dct = {
        "items": [
            [{"item_id": 1, "name": "first"}],
            [{"item_id": 2, "name": "second"}, {"item_id": 3, "name": "third"}],
        ]
    }

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

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
    dct = {"value": [1, 2, 3]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert all(cfg.value == 100 for cfg in cfgs)


def test_load_cfgs_override_nested_beats_dict():
    override = ConfigWithNested(nested=NestedConfig(inner_value=999))
    default = ConfigWithNested()
    dct = {"nested": {"inner_value": [1, 2, 3]}}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert all(cfg.nested.inner_value == 999 for cfg in cfgs)


def test_load_cfgs_seed_increments():
    override = ConfigWithSeed()
    default = ConfigWithSeed()
    dct = {"value": [10, 20, 30]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(cfgs) == 3
    assert cfgs[0].seed == 0
    assert cfgs[1].seed == 1
    assert cfgs[2].seed == 2


def test_load_cfgs_nonexistent_field_creates_error():
    override = SimpleConfig()
    default = SimpleConfig()
    dct = {"nonexistent_field": [1, 2, 3]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(errs) > 0


def test_load_cfgs_mixed_types_same_field():
    override = SimpleConfig()
    default = SimpleConfig()
    dct = {"value": [1, "not_an_int", 3]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(errs) >= 1


def test_load_cfgs_deeply_nested():
    @dataclasses.dataclass
    class Level3:
        val: int = 0

    @dataclasses.dataclass
    class Level2:
        level3: Level3 = dataclasses.field(default_factory=Level3)

    @dataclasses.dataclass
    class Level1:
        level2: Level2 = dataclasses.field(default_factory=Level2)

    override = Level1()
    default = Level1()
    dct = {"level2": {"level3": {"val": [1, 2, 3]}}}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(cfgs) == 3
    assert cfgs[0].level2.level3.val == 1
    assert cfgs[1].level2.level3.val == 2
    assert cfgs[2].level2.level3.val == 3


def test_expand():
    cfg = {"lr": [1, 2, 3]}
    expected = [{"lr": 1}, {"lr": 2}, {"lr": 3}]
    actual = list(expand(cfg))

    assert expected == actual


def test_expand_two_fields():
    cfg = {"lr": [1, 2], "wd": [3, 4]}
    expected = [
        {"lr": 1, "wd": 3},
        {"lr": 1, "wd": 4},
        {"lr": 2, "wd": 3},
        {"lr": 2, "wd": 4},
    ]
    actual = list(expand(cfg))

    assert expected == actual


def test_expand_nested():
    cfg = {"sae": {"dim": [1, 2, 3]}}
    expected = [{"sae": {"dim": 1}}, {"sae": {"dim": 2}}, {"sae": {"dim": 3}}]
    actual = list(expand(cfg))

    assert expected == actual


def test_expand_nested_and_unnested():
    cfg = {"sae": {"dim": [1, 2]}, "lr": [3, 4]}
    expected = [
        {"sae": {"dim": 1}, "lr": 3},
        {"sae": {"dim": 1}, "lr": 4},
        {"sae": {"dim": 2}, "lr": 3},
        {"sae": {"dim": 2}, "lr": 4},
    ]
    actual = list(expand(cfg))

    assert expected == actual


def test_expand_nested_and_unnested_backwards():
    cfg = {"a": [False, True], "b": {"c": [False, True]}}
    expected = [
        {"a": False, "b": {"c": False}},
        {"a": False, "b": {"c": True}},
        {"a": True, "b": {"c": False}},
        {"a": True, "b": {"c": True}},
    ]
    actual = list(expand(cfg))

    assert expected == actual


def test_expand_multiple():
    cfg = {"a": [1, 2, 3], "b": {"c": [4, 5, 6]}}
    expected = [
        {"a": 1, "b": {"c": 4}},
        {"a": 1, "b": {"c": 5}},
        {"a": 1, "b": {"c": 6}},
        {"a": 2, "b": {"c": 4}},
        {"a": 2, "b": {"c": 5}},
        {"a": 2, "b": {"c": 6}},
        {"a": 3, "b": {"c": 4}},
        {"a": 3, "b": {"c": 5}},
        {"a": 3, "b": {"c": 6}},
    ]
    actual = list(expand(cfg))

    assert expected == actual


def test_load_cfgs_nested_override_filters_empty_nested_dict():
    """Test line 59: when filtered nested dict is empty, it's not included."""
    override = ConfigWithNested(nested=NestedConfig(inner_value=100))
    default = ConfigWithNested()
    dct = {"nested": {"inner_value": [1, 2, 3]}, "outer_value": [10, 20]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    # Should only sweep outer_value since nested.inner_value is overridden
    assert len(cfgs) == 2
    assert all(cfg.nested.inner_value == 100 for cfg in cfgs)


def test_expand_non_list_value():
    """Test lines 109-110: expand with non-list, non-dict values."""
    cfg = {"a": 1, "b": 2}
    expected = [{"a": 1, "b": 2}]
    actual = list(expand(cfg))

    assert expected == actual


def test_load_cfgs_dataclass_field_in_dict():
    """Test line 128: dict_to_dataclass called from _convert_value."""
    @dataclasses.dataclass
    class Inner:
        x: int = 0

    @dataclasses.dataclass
    class Outer:
        inner: Inner = dataclasses.field(default_factory=Inner)

    override = Outer()
    default = Outer()
    dct = {"inner": [{"x": 1}, {"x": 2}]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(cfgs) == 2
    assert cfgs[0].inner.x == 1
    assert cfgs[1].inner.x == 2


def test_load_cfgs_typing_union_path():
    """Test line 133: typing.Union style optional path (Python 3.9 style)."""
    import typing

    @dataclasses.dataclass
    class ConfigWithTypingUnionPath:
        path: typing.Union[pathlib.Path, None] = None

    override = ConfigWithTypingUnionPath()
    default = ConfigWithTypingUnionPath()
    dct = {"path": ["/tmp/x"]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    assert len(cfgs) == 1
    assert cfgs[0].path == pathlib.Path("/tmp/x")


def test_load_cfgs_nested_seed_increment():
    """Test lines 168-169: nested seed auto-increment when sweeping nested field."""
    @dataclasses.dataclass
    class NestedWithSeed:
        seed: int = 0
        value: int = 5

    @dataclasses.dataclass
    class OuterWithNestedSeed:
        seed: int = 0
        nested: NestedWithSeed = dataclasses.field(default_factory=NestedWithSeed)

    override = OuterWithNestedSeed()
    default = OuterWithNestedSeed()
    dct = {"nested": {"value": [1, 2, 3]}}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    # Nested seed should auto-increment
    assert len(cfgs) == 3
    assert cfgs[0].nested.seed == 0
    assert cfgs[1].nested.seed == 1
    assert cfgs[2].nested.seed == 2


def test_dict_to_dataclass_with_missing_fields():
    """Test line 191: dict_to_dataclass when field is not in data."""
    from saev.configs import dict_to_dataclass

    @dataclasses.dataclass
    class Config:
        a: int = 1
        b: int = 2

    result = dict_to_dataclass({"a": 10}, Config)

    assert result.a == 10
    assert result.b == 2


def test_dict_to_dataclass_non_dataclass():
    """Test line 184: dict_to_dataclass when cls is not a dataclass."""
    from saev.configs import dict_to_dataclass

    result = dict_to_dataclass({"a": 1}, dict)

    assert result == {"a": 1}


def test_load_cfgs_all_nested_fields_overridden():
    """Test line 59: nested dict is empty after filtering all overridden fields."""
    override = ConfigWithNested(nested=NestedConfig(inner_value=999))
    default = ConfigWithNested()
    dct = {"nested": {"inner_value": [1, 2, 3]}, "outer_value": [10, 20]}

    cfgs, errs = load_cfgs(override, default=default, dct=dct)

    # Should sweep outer_value only, nested should be filtered out entirely
    assert len(cfgs) == 2
    assert all(cfg.nested.inner_value == 999 for cfg in cfgs)
    assert cfgs[0].outer_value == 10
    assert cfgs[1].outer_value == 20
