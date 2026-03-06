"""
Task-spec abstractions for Cambridge mimic-pair analysis.
"""

import dataclasses
import pathlib
import re

import beartype
import polars as pl
from tdiscovery.classification import LabelGrouping, apply_grouping, load_image_labels

DEFAULT_PAIR_SPECS = [
    "lativitta:malleti",
    "cyrbia:cythera",
    "notabilis:plesseni",
    "hydara:melpomene",
    "venus:vulcanus",
    "demophoon:rosina",
    "phyllis:nanna",
    "erato:thelxiopeia",
]
DEFAULT_VIEWS = ["dorsal", "ventral"]
TASK_NAME_RE = re.compile(
    r"^(?P<erato>[a-zA-Z0-9]+)_(?P<view_a>[a-zA-Z0-9]+)_vs_(?P<melp>[a-zA-Z0-9]+)_(?P<view_b>[a-zA-Z0-9]+)$"
)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TaskSpec:
    task_name: str
    source_col: str
    groups: dict[str, list[str]]
    n_erato: int
    n_melpomene: int
    n_total: int
    keep: bool


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class DecideTaskSpecsConfig:
    shards_dpath: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/79239bdd"
    )
    pair_specs: list[str] = dataclasses.field(
        default_factory=lambda: DEFAULT_PAIR_SPECS.copy()
    )
    views: list[str] = dataclasses.field(default_factory=lambda: DEFAULT_VIEWS.copy())
    task_names: list[str] = dataclasses.field(default_factory=list)
    min_samples_per_class: int = 50
    include_filtered: bool = False


@beartype.beartype
def parse_pair_spec(pair_spec: str) -> tuple[str, str]:
    erato_ssp, sep, melp_ssp = pair_spec.partition(":")
    msg = f"Pair spec must look like 'erato_ssp:melp_ssp', got '{pair_spec}'."
    assert sep == ":", msg
    erato_ssp = erato_ssp.strip()
    melp_ssp = melp_ssp.strip()
    msg = f"Pair spec has empty side: '{pair_spec}'."
    assert erato_ssp and melp_ssp, msg
    return erato_ssp, melp_ssp


@beartype.beartype
def get_task_name(erato_ssp: str, melp_ssp: str, view: str) -> str:
    return f"{erato_ssp}_{view}_vs_{melp_ssp}_{view}"


@beartype.beartype
def parse_task_name(task_name: str) -> tuple[str, str, str]:
    match = TASK_NAME_RE.fullmatch(task_name)
    msg = (
        "Task must match '{erato_ssp}_{view}_vs_{melp_ssp}_{view}', "
        f"got '{task_name}'."
    )
    assert match is not None, msg

    erato_ssp = str(match.group("erato"))
    melp_ssp = str(match.group("melp"))
    view_a = str(match.group("view_a"))
    view_b = str(match.group("view_b"))
    msg = f"Task has mismatched views: '{view_a}' vs '{view_b}'."
    assert view_a == view_b, msg
    return erato_ssp, melp_ssp, view_a


@beartype.beartype
def make_label_grouping(task_name: str) -> LabelGrouping:
    erato_ssp, melp_ssp, view = parse_task_name(task_name)
    return LabelGrouping(
        name=task_name,
        source_col="subspecies_view",
        groups={
            "erato": [f"{erato_ssp}_{view}"],
            "melpomene": [f"{melp_ssp}_{view}"],
        },
    )


@beartype.beartype
def dedup_keep_order(items: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


@beartype.beartype
def make_candidate_task_names(cfg: DecideTaskSpecsConfig) -> list[str]:
    if cfg.task_names:
        return dedup_keep_order(cfg.task_names)

    task_names = []
    for pair_spec in cfg.pair_specs:
        erato_ssp, melp_ssp = parse_pair_spec(pair_spec)
        for view in cfg.views:
            task_names.append(get_task_name(erato_ssp, melp_ssp, view))
    return dedup_keep_order(task_names)


@beartype.beartype
def get_empty_summary_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "task_name": pl.String,
            "n_erato": pl.Int64,
            "n_melpomene": pl.Int64,
            "n_total": pl.Int64,
            "keep": pl.Boolean,
            "source_col": pl.String,
            "erato_label": pl.String,
            "melpomene_label": pl.String,
        }
    )


@beartype.beartype
def decide_task_specs(
    cfg: DecideTaskSpecsConfig,
) -> tuple[list[TaskSpec], pl.DataFrame]:
    shards_dpath = cfg.shards_dpath.expanduser()
    msg = f"Shards directory missing: '{shards_dpath}'."
    assert shards_dpath.is_dir(), msg

    _, labels_by_col = load_image_labels(shards_dpath)
    msg = f"'subspecies_view' missing in labels loaded from '{shards_dpath}'."
    assert "subspecies_view" in labels_by_col, msg
    subspecies_view_n = labels_by_col["subspecies_view"]

    task_names = make_candidate_task_names(cfg)
    msg = "No task candidates. Set task_names or pair_specs."
    assert task_names, msg

    task_specs = []
    summary_rows = []
    for task_name in task_names:
        grouping = make_label_grouping(task_name)
        targets_n, class_names, include_n = apply_grouping(subspecies_view_n, grouping)
        class_to_i = {name: i for i, name in enumerate(class_names)}

        n_erato = (
            int((targets_n[include_n] == class_to_i["erato"]).sum())
            if "erato" in class_to_i
            else 0
        )
        n_melpomene = (
            int((targets_n[include_n] == class_to_i["melpomene"]).sum())
            if "melpomene" in class_to_i
            else 0
        )
        n_total = n_erato + n_melpomene
        keep = min(n_erato, n_melpomene) >= cfg.min_samples_per_class

        spec = TaskSpec(
            task_name=task_name,
            source_col=grouping.source_col,
            groups=grouping.groups,
            n_erato=n_erato,
            n_melpomene=n_melpomene,
            n_total=n_total,
            keep=keep,
        )
        summary_rows.append({
            "task_name": spec.task_name,
            "n_erato": spec.n_erato,
            "n_melpomene": spec.n_melpomene,
            "n_total": spec.n_total,
            "keep": spec.keep,
            "source_col": spec.source_col,
            "erato_label": spec.groups["erato"][0],
            "melpomene_label": spec.groups["melpomene"][0],
        })
        if keep or cfg.include_filtered:
            task_specs.append(spec)

    summary_df = (
        pl.DataFrame(summary_rows, infer_schema_length=None)
        if summary_rows
        else get_empty_summary_df()
    )
    summary_df = summary_df.sort(
        "keep", "n_total", "task_name", descending=[True, True, False]
    )
    return task_specs, summary_df
