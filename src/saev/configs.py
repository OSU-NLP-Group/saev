import dataclasses
import itertools
import pathlib
import types
import typing as tp
from collections.abc import Iterator

import beartype

T = tp.TypeVar("T")


@beartype.beartype
def load_cfgs(
    override: T, *, default: T, dct: dict[str, object]
) -> tuple[list[T], list[str]]:
    """
    Load a list of configs from a combination of sources.

    Args:
        override: Command-line overridden values.
        default: The default values for a config.
        dct: A dictionary from a .toml or .json file that has some list values to sweep over.

    Returns:
        A list of configs and a list of errors.
    """
    # Check that override and default are instances of a dataclass.
    assert dataclasses.is_dataclass(override) and not isinstance(override, type)
    assert dataclasses.is_dataclass(default) and not isinstance(default, type)

    # Find which fields were overridden (differ from default)
    overridden_fields = get_non_default_values(override, default)

    # Filter out overridden fields from the sweep dict
    filtered_dct = _filter_overridden_fields(dct, overridden_fields)

    # If there's nothing to sweep, return just the override
    if not filtered_dct:
        return [override], []

    # Use grid to expand the filtered dict
    return grid(override, filtered_dct)


@beartype.beartype
def _filter_overridden_fields(
    dct: dict[str, object], overridden: dict[str, object]
) -> dict[str, object]:
    """Remove fields from `dct` that are present in `overridden`."""
    result = {}
    for key, value in dct.items():
        if key not in overridden:
            result[key] = value
        elif isinstance(value, dict) and isinstance(overridden.get(key), dict):
            # Recursively filter nested dicts
            filtered = _filter_overridden_fields(value, overridden[key])
            if filtered:
                result[key] = filtered
    return result


@beartype.beartype
def grid(cfg: T, sweep_dct: dict[str, object]) -> tuple[list[T], list[str]]:
    """Generate configs from ``cfg`` according to ``sweep_dct``."""

    cfgs: list[T] = []
    errs: list[str] = []

    for d, dct in enumerate(expand(sweep_dct)):
        try:
            updates = _recursive_dataclass_update(cfg, dct, cfg, d)

            if hasattr(cfg, "seed") and "seed" not in updates:
                updates["seed"] = getattr(cfg, "seed", 0) + d

            cfgs.append(dataclasses.replace(cfg, **updates))
        except Exception as err:
            errs.append(str(err))

    return cfgs, errs


@beartype.beartype
def expand(config: dict[str, object]) -> Iterator[dict[str, object]]:
    """Expand a nested dict that may contain lists into many dicts."""

    yield from _expand_discrete(dict(config))


@beartype.beartype
def _expand_discrete(config: dict[str, object]) -> Iterator[dict[str, object]]:
    if not config:
        yield {}
        return

    key, value = config.popitem()

    if isinstance(value, list):
        for c in _expand_discrete(config):
            for v in value:
                yield {**c, key: v}
    elif isinstance(value, dict):
        for c, v in itertools.product(
            _expand_discrete(config), _expand_discrete(value)
        ):
            yield {**c, key: v}
    else:
        for c in _expand_discrete(config):
            yield {**c, key: value}


@beartype.beartype
def _convert_value(value: object, field_type: object) -> object:
    """Convert a value to the correct type based on field_type."""
    # Handle Optional types
    origin = tp.get_origin(field_type)
    args = tp.get_args(field_type)

    # Handle tuple[str, ...]
    if origin is tuple and args:
        return tuple(value) if isinstance(value, list) else value
    # Handle list[DataclassType]
    elif origin is list and args and dataclasses.is_dataclass(args[0]):
        return [dict_to_dataclass(item, args[0]) for item in value]
    # Handle regular dataclass fields
    elif dataclasses.is_dataclass(field_type):
        return dict_to_dataclass(value, field_type)
    # Handle pathlib.Path
    elif field_type is pathlib.Path:
        return pathlib.Path(value) if value is not None else value
    elif origin is tp.Union and pathlib.Path in args:
        return pathlib.Path(value) if value is not None else value
    elif origin is types.UnionType and pathlib.Path in args:
        return pathlib.Path(value) if value is not None else value
    else:
        # For basic types (int, str, bool, float, etc.), validate the type
        if isinstance(field_type, type) and not isinstance(value, field_type):
            raise TypeError(
                f"Expected {field_type.__name__}, got {type(value).__name__}"
            )
        return value


@beartype.beartype
def _recursive_dataclass_update(obj, updates: dict[str, object], base_cfg, d: int):
    """Recursively update nested dataclasses."""
    if not dataclasses.is_dataclass(obj):
        # If obj is not a dataclass, we can't update it recursively
        return updates

    result = {}
    for key, value in updates.items():
        if not hasattr(obj, key):
            # Key doesn't exist on this object, pass it through
            result[key] = value
            continue

        attr = getattr(obj, key)
        field_type = type(obj).__dataclass_fields__[key].type

        if dataclasses.is_dataclass(attr) and isinstance(value, dict):
            # Recursively update the nested dataclass
            nested_updates = _recursive_dataclass_update(attr, value, base_cfg, d)

            # Handle seed updates for nested objects
            if hasattr(attr, "seed") and "seed" not in nested_updates:
                base_seed = getattr(base_cfg, "seed", 0) if base_cfg else 0
                nested_updates["seed"] = getattr(attr, "seed", 0) + base_seed + d

            # Create a new instance of the nested dataclass with updates
            result[key] = dataclasses.replace(attr, **nested_updates)
        else:
            # Convert value to the correct type
            result[key] = _convert_value(value, field_type)

    return result


@beartype.beartype
def dict_to_dataclass(data: dict, cls: type[T]) -> T:
    """Recursively convert a dictionary to a dataclass instance."""
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}

    for field_name, field_type in field_types.items():
        if field_name not in data:
            continue

        value = data[field_name]

        # Handle Optional types
        origin = tp.get_origin(field_type)
        args = tp.get_args(field_type)

        # Handle tuple[str, ...]
        if origin is tuple and args:
            kwargs[field_name] = tuple(value) if isinstance(value, list) else value
        # Handle list[DataclassType]
        elif origin is list and args and dataclasses.is_dataclass(args[0]):
            kwargs[field_name] = [dict_to_dataclass(item, args[0]) for item in value]
        # Handle regular dataclass fields
        elif dataclasses.is_dataclass(field_type):
            kwargs[field_name] = dict_to_dataclass(value, field_type)
        # Handle pathlib.Path
        elif field_type is pathlib.Path:
            # Required Path field - always convert
            kwargs[field_name] = pathlib.Path(value) if value is not None else value
        elif origin is tp.Union and pathlib.Path in args:
            # Optional Path field (typing.Union style)
            kwargs[field_name] = pathlib.Path(value) if value is not None else value
        elif origin is types.UnionType and pathlib.Path in args:
            # Optional Path field (Python 3.10+ union style with |)
            kwargs[field_name] = pathlib.Path(value) if value is not None else value
        else:
            kwargs[field_name] = value

    return cls(**kwargs)


@beartype.beartype
def get_non_default_values(obj: T, default_obj: T) -> dict:
    """Recursively find fields that differ from defaults."""
    # Check that obj and default_obj are instances of a dataclass.
    assert dataclasses.is_dataclass(obj) and not isinstance(obj, type)
    assert dataclasses.is_dataclass(default_obj) and not isinstance(default_obj, type)

    obj_dict = dataclasses.asdict(obj)
    default_dict = dataclasses.asdict(default_obj)

    diff = {}
    for key, value in obj_dict.items():
        default_value = default_dict.get(key)
        if value != default_value:
            diff[key] = value

    return diff
