"""
Checkpoint-discovery abstractions for Cambridge mimic-pair analysis.
"""

import dataclasses
import json
import logging
import pathlib

import beartype
import cloudpickle
import numpy as np
import polars as pl
import sklearn.metrics

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class DiscoverCheckpointsConfig:
    run_root_dpath: pathlib.Path
    shard_id: str
    task_name: str
    run_ids: list[str]
    c_values: list[float] = dataclasses.field(default_factory=list)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class CheckpointSummary:
    task_name: str
    patch_agg: str
    cls_key: str
    C: float
    test_acc: float
    balanced_acc: float
    features: list[int]
    weights: list[float]


@beartype.beartype
def get_empty_ckpt_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "run_id": pl.String,
            "C": pl.Float64,
            "n_features": pl.Int64,
            "balanced_acc": pl.Float64,
            "test_acc": pl.Float64,
            "features": pl.List(pl.Int64),
            "weights": pl.List(pl.Float64),
            "ckpt_fpath": pl.String,
        }
    )


@beartype.beartype
def c_is_selected(C: float, selected_c_values: list[float]) -> bool:
    if not selected_c_values:
        return True
    return any(abs(C - selected) < 1e-12 for selected in selected_c_values)


@beartype.beartype
def load_cls_checkpoint(ckpt_fpath: pathlib.Path) -> CheckpointSummary:
    with open(ckpt_fpath, "rb") as fd:
        header_line = fd.readline()
        header = json.loads(header_line.decode("utf8"))
        payload = cloudpickle.load(fd)

    cfg = header["cfg"]
    cls_cfg = cfg["cls"]
    task_cfg = cfg["task"]

    test_y_n = np.asarray(payload["test_y"])
    test_pred_n = np.asarray(payload["test_pred"])
    msg = f"test_y/test_pred shape mismatch in '{ckpt_fpath}'."
    assert test_y_n.shape == test_pred_n.shape, msg
    msg = f"No test labels in '{ckpt_fpath}'."
    assert test_y_n.size > 0, msg

    balanced_acc = float(sklearn.metrics.balanced_accuracy_score(test_y_n, test_pred_n))

    coef_cd = np.asarray(payload["classifier"].coef_)
    msg = f"Unexpected coef shape in '{ckpt_fpath}': {coef_cd.shape}."
    assert coef_cd.ndim == 2, msg
    msg = f"Expected binary sparse-linear classifier in '{ckpt_fpath}', got coef shape {coef_cd.shape}."
    assert coef_cd.shape[0] == 1, msg

    nonzero_mask_d = np.any(coef_cd != 0, axis=0)
    features_i = np.where(nonzero_mask_d)[0]
    weights_d = coef_cd[0, nonzero_mask_d]

    return CheckpointSummary(
        task_name=str(task_cfg["name"]),
        patch_agg=str(cfg["patch_agg"]),
        cls_key=str(cls_cfg["key"]),
        C=float(cls_cfg["C"]),
        test_acc=float(header["test_acc"]),
        balanced_acc=balanced_acc,
        features=features_i.astype(np.int64).tolist(),
        weights=weights_d.astype(np.float64).tolist(),
    )


@beartype.beartype
def discover_checkpoints(cfg: DiscoverCheckpointsConfig) -> pl.DataFrame:
    run_root_dpath = cfg.run_root_dpath.expanduser()
    msg = f"Runs root does not exist: '{run_root_dpath}'."
    assert run_root_dpath.is_dir(), msg

    msg = "Provide at least one run id."
    assert cfg.run_ids, msg

    task_name = cfg.task_name.strip()
    msg = "Task name is required."
    assert task_name, msg

    rows: list[dict[str, object]] = []
    for run_id in cfg.run_ids:
        ckpts_dpath = run_root_dpath / run_id / "inference" / cfg.shard_id
        if not ckpts_dpath.is_dir():
            continue

        pattern = f"cls_{task_name}_max_C*.pkl"
        for ckpt_fpath in sorted(ckpts_dpath.glob(pattern)):
            try:
                summary = load_cls_checkpoint(ckpt_fpath)
            except Exception as err:
                logger.warning("Skipping '%s': %s", ckpt_fpath, err)
                continue

            if summary.task_name != task_name:
                continue
            if summary.patch_agg != "max":
                continue
            if summary.cls_key != "sparse-linear":
                continue
            if not c_is_selected(summary.C, cfg.c_values):
                continue

            rows.append({
                "run_id": run_id,
                "C": summary.C,
                "n_features": len(summary.features),
                "balanced_acc": summary.balanced_acc,
                "test_acc": summary.test_acc,
                "features": summary.features,
                "weights": summary.weights,
                "ckpt_fpath": str(ckpt_fpath),
            })

    if not rows:
        return get_empty_ckpt_df()

    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "balanced_acc", "n_features", descending=[True, False]
    )
