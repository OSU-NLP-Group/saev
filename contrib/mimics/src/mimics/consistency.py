"""
Consistency scoring for Cambridge mimic-pair feature triage.
"""

import dataclasses
import json
import pathlib

import beartype
import numpy as np
import polars as pl
import scipy.sparse
import sklearn.metrics
from tdiscovery.classification import apply_grouping, load_image_labels

from . import tasks


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ScoreFeaturesConfig:
    run_root_dpath: pathlib.Path
    shards_dpath: pathlib.Path
    run_id: str
    shard_id: str
    task_name: str
    feature_ids: list[int]
    top_k: int = 8
    n_bootstrap: int = 64
    feature_chunk_size: int = 64
    seed: int = 0


@beartype.beartype
def get_empty_consistency_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "run_id": pl.String,
            "task": pl.String,
            "feature_id": pl.Int64,
            "consistency": pl.Float64,
            "selectivity": pl.Float64,
            "auroc": pl.Float64,
            "support_overall": pl.Float64,
            "support_erato": pl.Float64,
            "support_melpomene": pl.Float64,
            "topk_stability_erato": pl.Float64,
            "topk_stability_melpomene": pl.Float64,
            "strength_erato": pl.Float64,
            "strength_melpomene": pl.Float64,
            "n_nonzero_erato": pl.Int64,
            "n_nonzero_melpomene": pl.Int64,
        }
    )


@beartype.beartype
def get_consistency_fpath(
    run_root_dpath: pathlib.Path, *, run_id: str, shard_id: str
) -> pathlib.Path:
    run_id = run_id.strip()
    shard_id = shard_id.strip()
    msg = "run_id is required."
    assert run_id, msg
    msg = "shard_id is required."
    assert shard_id, msg

    return (
        run_root_dpath
        / run_id
        / "inference"
        / shard_id
        / "cambridge-mimics-consistency.parquet"
    )


@beartype.beartype
def load_consistency_df(
    run_root_dpath: pathlib.Path,
    *,
    run_id: str,
    shard_id: str,
    task_names: list[str] | None = None,
) -> pl.DataFrame:
    consistency_fpath = get_consistency_fpath(
        run_root_dpath, run_id=run_id, shard_id=shard_id
    )
    if not consistency_fpath.is_file():
        return get_empty_consistency_df()

    try:
        consistency_df = pl.read_parquet(consistency_fpath)
    except Exception:
        return get_empty_consistency_df()

    required_cols = {"task", "feature_id", "consistency", "selectivity"}
    if not required_cols <= set(consistency_df.columns):
        return get_empty_consistency_df()

    selected_task_names = sorted({
        task_name.strip() for task_name in (task_names or []) if task_name.strip()
    })
    if selected_task_names:
        consistency_df = consistency_df.filter(
            pl.col("task").is_in(selected_task_names)
        )
    if not consistency_df.height:
        return get_empty_consistency_df()

    return consistency_df.sort(
        "consistency", "selectivity", "feature_id", descending=[True, True, False]
    )


@beartype.beartype
def write_consistency_df(
    consistency_df: pl.DataFrame,
    run_root_dpath: pathlib.Path,
    *,
    run_id: str,
    shard_id: str,
) -> pathlib.Path:
    consistency_fpath = get_consistency_fpath(
        run_root_dpath, run_id=run_id, shard_id=shard_id
    )
    consistency_fpath.parent.mkdir(parents=True, exist_ok=True)
    consistency_df.write_parquet(consistency_fpath)
    return consistency_fpath


@beartype.beartype
def get_empty_task_specs_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "run_id": pl.String,
            "task": pl.String,
            "feature_ids": pl.List(pl.Int64),
            "n_features": pl.Int64,
        }
    )


@beartype.beartype
def discover_task_specs(
    run_root_dpath: pathlib.Path,
    *,
    shard_id: str,
    run_ids: list[str] | None = None,
    task_filter: str = "",
) -> pl.DataFrame:
    run_root_dpath = run_root_dpath.expanduser()
    msg = f"Runs root missing: '{run_root_dpath}'."
    assert run_root_dpath.is_dir(), msg
    shard_id = shard_id.strip()
    msg = "shard_id is required."
    assert shard_id, msg

    selected_run_ids = {run_id.strip() for run_id in (run_ids or []) if run_id.strip()}
    task_filter = task_filter.strip()

    feature_ids_by_key: dict[tuple[str, str], set[int]] = {}
    pattern = f"*/inference/{shard_id}/images/*/cambridge-mimics/_render.json"
    for render_meta_fpath in sorted(run_root_dpath.glob(pattern)):
        rel_parts = render_meta_fpath.relative_to(run_root_dpath).parts
        if len(rel_parts) < 7:
            continue
        run_id = str(rel_parts[0])
        if selected_run_ids and run_id not in selected_run_ids:
            continue

        try:
            feature_id = int(rel_parts[4])
        except ValueError:
            continue

        try:
            with open(render_meta_fpath) as fd:
                payload = json.load(fd)
        except Exception:
            continue
        task_name = str(payload.get("task", "")).strip()
        if not task_name:
            continue
        if task_filter and task_filter not in task_name:
            continue

        key = (run_id, task_name)
        if key not in feature_ids_by_key:
            feature_ids_by_key[key] = set()
        feature_ids_by_key[key].add(feature_id)

    if not feature_ids_by_key:
        return get_empty_task_specs_df()

    rows = []
    for (run_id, task_name), feature_ids in sorted(feature_ids_by_key.items()):
        sorted_feature_ids = sorted(feature_ids)
        rows.append({
            "run_id": run_id,
            "task": task_name,
            "feature_ids": sorted_feature_ids,
            "n_features": len(sorted_feature_ids),
        })

    return pl.DataFrame(rows, infer_schema_length=None).sort("run_id", "task")


@beartype.beartype
def _get_top_local_i(vals_n: np.ndarray, n_keep: int) -> np.ndarray:
    if not len(vals_n):
        return np.empty(0, dtype=np.int64)
    msg = "n_keep must be >= 1."
    assert n_keep >= 1, msg
    if n_keep >= len(vals_n):
        return np.argsort(-vals_n, kind="stable").astype(np.int64, copy=False)

    top_local_i = np.argpartition(vals_n, -n_keep)[-n_keep:]
    top_vals_n = vals_n[top_local_i]
    order_i = np.argsort(-top_vals_n, kind="stable")
    return top_local_i[order_i].astype(np.int64, copy=False)


@beartype.beartype
def _topk_strength(vals_n: np.ndarray, top_k: int) -> float:
    nonzero_vals_n = vals_n[vals_n > 0.0]
    if not len(nonzero_vals_n):
        return 0.0

    n_keep = min(top_k, len(nonzero_vals_n))
    top_local_i = _get_top_local_i(nonzero_vals_n, n_keep)
    top_vals_n = nonzero_vals_n[top_local_i]
    mean_val = float(top_vals_n.mean())
    if mean_val <= 0.0:
        return 0.0

    cv = float(top_vals_n.std() / (mean_val + 1e-12))
    return float(np.clip(1.0 - cv, 0.0, 1.0))


@beartype.beartype
def _pairwise_jaccard_mean(mask_bn: np.ndarray) -> float:
    n_boot, _ = mask_bn.shape
    if n_boot < 2:
        return 0.0

    mask_bi = mask_bn.astype(np.int16, copy=False)
    inter_bn = mask_bi @ mask_bi.T
    sizes_b = mask_bi.sum(axis=1)
    union_bn = sizes_b[:, None] + sizes_b[None, :] - inter_bn

    tri_i, tri_j = np.triu_indices(n_boot, k=1)
    inter_pairs_n = inter_bn[tri_i, tri_j]
    union_pairs_n = union_bn[tri_i, tri_j]
    valid_n = union_pairs_n > 0
    if not np.any(valid_n):
        return 0.0

    jaccard_n = inter_pairs_n[valid_n] / union_pairs_n[valid_n]
    return float(jaccard_n.mean())


@beartype.beartype
def _bootstrap_topk_stability(
    vals_n: np.ndarray,
    *,
    top_k: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, int]:
    nonzero_local_i = np.where(vals_n > 0.0)[0]
    n_nonzero = int(len(nonzero_local_i))
    if not n_nonzero:
        return 0.0, 0

    n_keep = min(top_k, n_nonzero)
    n_boot = max(2, n_bootstrap)
    mask_bn = np.zeros((n_boot, n_nonzero), dtype=bool)
    nonzero_vals_n = vals_n[nonzero_local_i]
    rng = np.random.default_rng(seed)
    for b_i in range(n_boot):
        sample_local_i = rng.integers(0, n_nonzero, size=n_nonzero, endpoint=False)
        sample_vals_n = nonzero_vals_n[sample_local_i]
        top_sample_local_i = _get_top_local_i(sample_vals_n, n_keep)
        picked_local_i = np.unique(sample_local_i[top_sample_local_i])
        mask_bn[b_i, picked_local_i] = True

    return _pairwise_jaccard_mean(mask_bn), n_nonzero


@beartype.beartype
def score_features(cfg: ScoreFeaturesConfig) -> pl.DataFrame:
    run_root_dpath = cfg.run_root_dpath.expanduser()
    shards_dpath = cfg.shards_dpath.expanduser()
    run_id = cfg.run_id.strip()
    shard_id = cfg.shard_id.strip()
    task_name = cfg.task_name.strip()

    msg = f"Runs root missing: '{run_root_dpath}'."
    assert run_root_dpath.is_dir(), msg
    msg = f"Shards directory missing: '{shards_dpath}'."
    assert shards_dpath.is_dir(), msg
    msg = "run_id is required."
    assert run_id, msg
    msg = "shard_id is required."
    assert shard_id, msg
    msg = "task_name is required."
    assert task_name, msg
    tasks.parse_task_name(task_name)
    msg = "top_k must be >= 1."
    assert cfg.top_k >= 1, msg
    msg = "n_bootstrap must be >= 2."
    assert cfg.n_bootstrap >= 2, msg
    msg = "feature_chunk_size must be >= 1."
    assert cfg.feature_chunk_size >= 1, msg
    if not cfg.feature_ids:
        return get_empty_consistency_df()

    unique_feature_ids = sorted({int(feature_id) for feature_id in cfg.feature_ids})
    msg = "feature_ids must be >= 0."
    assert min(unique_feature_ids) >= 0, msg

    _, labels_by_col = load_image_labels(shards_dpath)
    msg = f"'subspecies_view' missing in labels loaded from '{shards_dpath}'."
    assert "subspecies_view" in labels_by_col, msg
    subspecies_view_n = labels_by_col["subspecies_view"]
    n_images = len(subspecies_view_n)
    msg = "Expected at least one image."
    assert n_images > 0, msg

    grouping = tasks.make_label_grouping(task_name)
    targets_n, class_names, include_n = apply_grouping(subspecies_view_n, grouping)
    class_to_i = {name: i for i, name in enumerate(class_names)}
    msg = f"Expected classes erato/melpomene, got {class_names}."
    assert {"erato", "melpomene"} <= set(class_to_i), msg

    include_targets_n = targets_n[include_n]
    msg = "No images matched task grouping."
    assert len(include_targets_n) > 0, msg

    token_acts_fpath = (
        run_root_dpath / run_id / "inference" / shard_id / "token_acts.npz"
    )
    msg = f"Missing token activations: '{token_acts_fpath}'."
    assert token_acts_fpath.exists(), msg
    token_acts_csr = scipy.sparse.load_npz(token_acts_fpath)
    msg = "token_acts matrix must be 2D."
    assert token_acts_csr.ndim == 2, msg
    msg = f"token_acts rows {token_acts_csr.shape[0]} must be divisible by n_images {n_images}."
    assert token_acts_csr.shape[0] % n_images == 0, msg
    tokens_per_image = token_acts_csr.shape[0] // n_images
    msg = "tokens_per_image must be >= 1."
    assert tokens_per_image >= 1, msg
    max_feature_id = max(unique_feature_ids)
    msg = f"Feature id {max_feature_id} out of range for token_acts width {token_acts_csr.shape[1]}."
    assert max_feature_id < token_acts_csr.shape[1], msg

    binary_target_n = (include_targets_n == class_to_i["melpomene"]).astype(np.int8)
    token_acts_csc = token_acts_csr.tocsc()
    rows: list[dict[str, object]] = []
    n_features = len(unique_feature_ids)
    for start in range(0, n_features, cfg.feature_chunk_size):
        stop = min(start + cfg.feature_chunk_size, n_features)
        chunk_feature_ids = unique_feature_ids[start:stop]
        chunk_acts_np = token_acts_csc[:, chunk_feature_ids].toarray()
        msg = (
            f"Chunk shape mismatch for run '{run_id}': got {chunk_acts_np.shape}, "
            f"expected {(token_acts_csc.shape[0], len(chunk_feature_ids))}."
        )
        assert chunk_acts_np.shape == (
            token_acts_csc.shape[0],
            len(chunk_feature_ids),
        ), msg

        for local_i, feature_id in enumerate(chunk_feature_ids):
            acts_by_image_np = chunk_acts_np[:, local_i].reshape(
                n_images, tokens_per_image
            )
            max_acts_n = acts_by_image_np.max(axis=1)
            include_max_acts_n = max_acts_n[include_n]
            msg = (
                f"Activation shape mismatch for feature {feature_id}: "
                f"{include_max_acts_n.shape} vs {include_targets_n.shape}."
            )
            assert include_max_acts_n.shape == include_targets_n.shape, msg

            support_overall = float(np.mean(include_max_acts_n > 0.0))
            if np.unique(binary_target_n).size < 2:
                auroc = float("nan")
                selectivity = float("nan")
            else:
                auroc = float(
                    sklearn.metrics.roc_auc_score(binary_target_n, include_max_acts_n)
                )
                selectivity = float(np.clip(2.0 * abs(auroc - 0.5), 0.0, 1.0))

            class_stats_dct: dict[str, dict[str, float | int]] = {}
            class_components = []
            for class_name in ("erato", "melpomene"):
                class_i = int(class_to_i[class_name])
                class_mask_n = include_targets_n == class_i
                class_vals_n = include_max_acts_n[class_mask_n]
                msg = f"No rows for class '{class_name}' in task '{task_name}'."
                assert len(class_vals_n) > 0, msg

                support_c = float(np.mean(class_vals_n > 0.0))
                topk_stability_c, n_nonzero_c = _bootstrap_topk_stability(
                    class_vals_n,
                    top_k=cfg.top_k,
                    n_bootstrap=cfg.n_bootstrap,
                    seed=cfg.seed + feature_id * 131 + class_i,
                )
                strength_c = _topk_strength(class_vals_n, cfg.top_k)
                component_c = topk_stability_c * strength_c * float(np.sqrt(support_c))
                class_components.append(component_c)
                class_stats_dct[class_name] = {
                    "support": support_c,
                    "topk_stability": topk_stability_c,
                    "strength": strength_c,
                    "n_nonzero": int(n_nonzero_c),
                }

            consistency = float(np.mean(class_components))
            rows.append({
                "run_id": run_id,
                "task": task_name,
                "feature_id": int(feature_id),
                "consistency": consistency,
                "selectivity": selectivity,
                "auroc": auroc,
                "support_overall": support_overall,
                "support_erato": class_stats_dct["erato"]["support"],
                "support_melpomene": class_stats_dct["melpomene"]["support"],
                "topk_stability_erato": class_stats_dct["erato"]["topk_stability"],
                "topk_stability_melpomene": class_stats_dct["melpomene"][
                    "topk_stability"
                ],
                "strength_erato": class_stats_dct["erato"]["strength"],
                "strength_melpomene": class_stats_dct["melpomene"]["strength"],
                "n_nonzero_erato": class_stats_dct["erato"]["n_nonzero"],
                "n_nonzero_melpomene": class_stats_dct["melpomene"]["n_nonzero"],
            })

    if not rows:
        return get_empty_consistency_df()

    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "consistency", "selectivity", "feature_id", descending=[True, True, False]
    )
