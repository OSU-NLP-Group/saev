"""
Rendering abstractions for Cambridge mimic-pair analysis.
"""

import dataclasses
import datetime as dt
import json
import logging
import pathlib
import shutil
import typing as tp

import beartype
import numpy as np
import polars as pl
import scipy.sparse
from PIL import Image
from tdiscovery.classification import apply_grouping, load_image_labels

import saev.data
import saev.data.datasets
import saev.data.models
import saev.viz

from . import tasks

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RenderContext:
    img_ds: tp.Any
    n_images: int
    tokens_per_image: int
    patch_size: int
    subspecies_view_n: list[str]
    include_n: np.ndarray
    targets_n: np.ndarray
    class_to_i: dict[str, int]


@beartype.beartype
def get_feature_root_dpath(
    run_root_dpath: pathlib.Path,
    run_id: str,
    shard_id: str,
    feature_id: int,
) -> pathlib.Path:
    return (
        run_root_dpath
        / run_id
        / "inference"
        / shard_id
        / "images"
        / str(feature_id)
        / "cambridge-mimics"
    )


@beartype.beartype
def get_done_fpath(
    run_root_dpath: pathlib.Path,
    run_id: str,
    shard_id: str,
    feature_id: int,
) -> pathlib.Path:
    return (
        get_feature_root_dpath(run_root_dpath, run_id, shard_id, feature_id)
        / "_done.json"
    )


@beartype.beartype
def load_done_payload(done_fpath: pathlib.Path) -> dict[str, object] | None:
    if not done_fpath.exists():
        return None
    try:
        with open(done_fpath) as fd:
            return json.load(fd)
    except Exception:
        return None


@beartype.beartype
def get_empty_render_plan_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "run_id": pl.String,
            "feature_id": pl.Int64,
            "abs_weight": pl.Float64,
            "token_acts_exists": pl.Boolean,
            "done_exists": pl.Boolean,
            "done_n_per_class": pl.Int64,
            "done_task": pl.String,
            "needs_render": pl.Boolean,
            "status": pl.String,
            "render_root_dpath": pl.String,
            "sentinel_fpath": pl.String,
        }
    )


@beartype.beartype
def make_render_plan(
    feature_df: pl.DataFrame,
    *,
    run_root_dpath: pathlib.Path,
    shard_id: str,
    task_name: str,
    n_per_class: int,
    force_render: bool = False,
) -> pl.DataFrame:
    if not feature_df.height:
        return get_empty_render_plan_df()

    required_cols = {"run_id", "feature_id", "abs_weight"}
    missing = required_cols - set(feature_df.columns)
    msg = f"Missing columns in feature table: {sorted(missing)}."
    assert not missing, msg

    msg = "n_per_class must be >= 1."
    assert n_per_class >= 1, msg

    rows = []
    for row_dct in feature_df.iter_rows(named=True):
        run_id = str(row_dct["run_id"])
        feature_id = int(row_dct["feature_id"])
        abs_weight = float(row_dct["abs_weight"])
        token_acts_fpath = (
            run_root_dpath / run_id / "inference" / shard_id / "token_acts.npz"
        )

        done_fpath = get_done_fpath(run_root_dpath, run_id, shard_id, feature_id)
        done_payload = load_done_payload(done_fpath)
        done_n_per_class = None
        done_task = None
        if done_payload is not None:
            raw_done_n = done_payload.get("n_per_class")
            if isinstance(raw_done_n, int):
                done_n_per_class = int(raw_done_n)
            elif isinstance(raw_done_n, float) and raw_done_n.is_integer():
                done_n_per_class = int(raw_done_n)
            elif isinstance(raw_done_n, str):
                try:
                    done_n_per_class = int(raw_done_n)
                except ValueError:
                    done_n_per_class = None

            raw_done_task = done_payload.get("task")
            if isinstance(raw_done_task, str):
                done_task = raw_done_task

        token_acts_exists = token_acts_fpath.exists()
        is_up_to_date = (
            done_payload is not None
            and done_n_per_class == n_per_class
            and done_task == task_name
        )
        needs_render = token_acts_exists and (force_render or not is_up_to_date)
        if not token_acts_exists:
            status = "missing_token_acts"
        elif needs_render:
            status = "render"
        else:
            status = "skip"

        rows.append({
            "run_id": run_id,
            "feature_id": feature_id,
            "abs_weight": abs_weight,
            "token_acts_exists": token_acts_exists,
            "done_exists": done_payload is not None,
            "done_n_per_class": done_n_per_class,
            "done_task": done_task,
            "needs_render": needs_render,
            "status": status,
            "render_root_dpath": str(done_fpath.parent),
            "sentinel_fpath": str(done_fpath),
        })

    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "needs_render",
        "abs_weight",
        "run_id",
        "feature_id",
        descending=[True, True, False, False],
    )


@beartype.beartype
def get_pending_features_by_run(render_plan_df: pl.DataFrame) -> dict[str, list[int]]:
    if not render_plan_df.height:
        return {}

    pending_df = render_plan_df.filter(pl.col("status") == "render").sort(
        "run_id", "abs_weight", descending=[False, True]
    )
    if not pending_df.height:
        return {}

    run_to_features_dct: dict[str, list[int]] = {}
    seen_dct: dict[str, set[int]] = {}
    for row_dct in pending_df.iter_rows(named=True):
        run_id = str(row_dct["run_id"])
        feature_id = int(row_dct["feature_id"])
        if run_id not in run_to_features_dct:
            run_to_features_dct[run_id] = []
            seen_dct[run_id] = set()
        if feature_id in seen_dct[run_id]:
            continue
        run_to_features_dct[run_id].append(feature_id)
        seen_dct[run_id].add(feature_id)

    return run_to_features_dct


@beartype.beartype
def make_render_context(shards_dpath: pathlib.Path, task_name: str) -> RenderContext:
    task_grouping = tasks.make_label_grouping(task_name)

    md = saev.data.Metadata.load(shards_dpath)
    model_cls = tp.cast(
        tp.Callable[[str], tp.Any], saev.data.models.load_model_cls(md.family)
    )
    vit = model_cls(md.ckpt)
    resize_tr = vit.make_resize(md.ckpt, md.content_tokens_per_example, scale=1.0)

    data_cfg = tp.cast(saev.data.datasets.Config, md.make_data_cfg())
    img_ds = saev.data.datasets.get_dataset(
        data_cfg, data_transform=resize_tr, mask_transform=resize_tr
    )

    _, labels_by_col = load_image_labels(shards_dpath)
    msg = f"'subspecies_view' missing in labels loaded from '{shards_dpath}'."
    assert "subspecies_view" in labels_by_col, msg
    subspecies_view_n = labels_by_col["subspecies_view"]

    n_images = len(subspecies_view_n)
    msg = f"Image count mismatch: dataset={len(img_ds)} labels={n_images}."
    assert len(img_ds) == n_images, msg
    msg = f"Image count mismatch: metadata={md.n_examples} labels={n_images}."
    assert md.n_examples == n_images, msg

    targets_n, class_names, include_n = apply_grouping(subspecies_view_n, task_grouping)
    class_to_i = {name: i for i, name in enumerate(class_names)}
    msg = f"Expected classes erato/melpomene, got {class_names}."
    assert {"erato", "melpomene"} <= set(class_to_i), msg

    patch_size = int(vit.patch_size)
    msg = "Patch size must be >= 1."
    assert patch_size >= 1, msg

    return RenderContext(
        img_ds=img_ds,
        n_images=n_images,
        tokens_per_image=int(md.content_tokens_per_example),
        patch_size=patch_size,
        subspecies_view_n=subspecies_view_n,
        include_n=include_n,
        targets_n=targets_n,
        class_to_i=class_to_i,
    )


@beartype.beartype
def _setup_feature_root_dpath(feature_root_dpath: pathlib.Path) -> None:
    msg = f"Unexpected feature root path: '{feature_root_dpath}'."
    assert feature_root_dpath.name == "cambridge-mimics", msg
    if feature_root_dpath.exists():
        shutil.rmtree(feature_root_dpath)
    feature_root_dpath.mkdir(parents=True, exist_ok=True)


@beartype.beartype
def _save_highlighted_image(
    ctx: RenderContext,
    img_i: int,
    feature_acts_p: np.ndarray,
    *,
    upper: float,
    out_fpath: pathlib.Path,
) -> None:
    sample = ctx.img_ds[img_i]
    msg = f"Sample {img_i} does not have image data."
    assert isinstance(sample, dict) and "data" in sample, msg
    img = sample["data"]
    msg = f"Sample {img_i} image is not a PIL image."
    assert isinstance(img, Image.Image), msg
    img_with_highlights = saev.viz.add_highlights(
        img, feature_acts_p, ctx.patch_size, upper=upper
    )
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    img_with_highlights.save(out_fpath)


@beartype.beartype
def _write_feature_metadata(
    *,
    feature_root_dpath: pathlib.Path,
    run_id: str,
    feature_id: int,
    task_name: str,
    n_per_class: int,
    n_images_written: int,
    records: list[dict[str, object]],
) -> None:
    render_meta_fpath = feature_root_dpath / "_render.json"
    render_payload = {
        "run_id": run_id,
        "feature_id": feature_id,
        "task": task_name,
        "n_per_class": n_per_class,
        "n_images_written": n_images_written,
        "records": records,
    }
    with open(render_meta_fpath, "w") as fd:
        json.dump(render_payload, fd, indent=2)
        fd.write("\n")

    done_fpath = feature_root_dpath / "_done.json"
    done_payload = {
        "task": task_name,
        "n_per_class": n_per_class,
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "n_images_written": n_images_written,
    }
    with open(done_fpath, "w") as fd:
        json.dump(done_payload, fd, indent=2)
        fd.write("\n")


@beartype.beartype
def _render_feature(
    ctx: RenderContext,
    *,
    run_root_dpath: pathlib.Path,
    run_id: str,
    shard_id: str,
    feature_id: int,
    feature_acts_np: np.ndarray,
    n_per_class: int,
    task_name: str,
) -> int:
    feature_root_dpath = get_feature_root_dpath(
        run_root_dpath, run_id, shard_id, feature_id
    )
    _setup_feature_root_dpath(feature_root_dpath)

    acts_by_image_np = feature_acts_np.reshape(ctx.n_images, ctx.tokens_per_image)
    max_acts_n = acts_by_image_np.max(axis=1)
    upper = float(acts_by_image_np.max())

    n_images_written = 0
    records: list[dict[str, object]] = []
    for class_name in ("erato", "melpomene"):
        class_i = int(ctx.class_to_i[class_name])
        class_img_i = np.where((ctx.targets_n == class_i) & ctx.include_n)[0]
        class_max_n = max_acts_n[class_img_i]
        nonzero_mask_n = class_max_n > 0.0

        nonzero_img_i = class_img_i[nonzero_mask_n]
        nonzero_vals_n = class_max_n[nonzero_mask_n]
        if not len(nonzero_img_i):
            continue

        n_keep = min(n_per_class, len(nonzero_img_i))
        top_local_i = np.argsort(-nonzero_vals_n, kind="stable")[:n_keep]
        bottom_local_i = np.argsort(nonzero_vals_n, kind="stable")[:n_keep]

        for rank, local_i in enumerate(top_local_i.tolist()):
            img_i = int(nonzero_img_i[local_i])
            subspecies_view = str(ctx.subspecies_view_n[img_i])
            msg = f"Invalid subspecies_view for path: '{subspecies_view}'."
            assert subspecies_view and "/" not in subspecies_view, msg
            out_fpath = feature_root_dpath / subspecies_view / f"{rank}_sae_img.png"
            _save_highlighted_image(
                ctx,
                img_i,
                acts_by_image_np[img_i],
                upper=upper,
                out_fpath=out_fpath,
            )
            n_images_written += 1
            records.append({
                "direction": "top",
                "class_name": class_name,
                "rank": rank,
                "example_idx": img_i,
                "subspecies_view": subspecies_view,
                "max_activation": float(nonzero_vals_n[local_i]),
                "img_fpath": str(out_fpath),
            })

        for rank, local_i in enumerate(bottom_local_i.tolist()):
            img_i = int(nonzero_img_i[local_i])
            subspecies_view = str(ctx.subspecies_view_n[img_i])
            msg = f"Invalid subspecies_view for path: '{subspecies_view}'."
            assert subspecies_view and "/" not in subspecies_view, msg
            out_fpath = (
                feature_root_dpath / subspecies_view / "bottom" / f"{rank}_sae_img.png"
            )
            _save_highlighted_image(
                ctx,
                img_i,
                acts_by_image_np[img_i],
                upper=upper,
                out_fpath=out_fpath,
            )
            n_images_written += 1
            records.append({
                "direction": "bottom",
                "class_name": class_name,
                "rank": rank,
                "example_idx": img_i,
                "subspecies_view": subspecies_view,
                "max_activation": float(nonzero_vals_n[local_i]),
                "img_fpath": str(out_fpath),
            })

    _write_feature_metadata(
        feature_root_dpath=feature_root_dpath,
        run_id=run_id,
        feature_id=feature_id,
        task_name=task_name,
        n_per_class=n_per_class,
        n_images_written=n_images_written,
        records=records,
    )
    return n_images_written


@beartype.beartype
def render_run_features(
    ctx: RenderContext,
    *,
    run_root_dpath: pathlib.Path,
    run_id: str,
    shard_id: str,
    feature_ids: list[int],
    n_per_class: int,
    task_name: str,
    feature_chunk_size: int = 64,
) -> tuple[int, int]:
    if not feature_ids:
        return 0, 0

    msg = "feature_chunk_size must be >= 1."
    assert feature_chunk_size >= 1, msg
    msg = "n_per_class must be >= 1."
    assert n_per_class >= 1, msg

    token_acts_fpath = (
        run_root_dpath / run_id / "inference" / shard_id / "token_acts.npz"
    )
    msg = f"Missing token activations: '{token_acts_fpath}'."
    assert token_acts_fpath.exists(), msg

    token_acts_csr = scipy.sparse.load_npz(token_acts_fpath)
    n_tokens_expected = ctx.n_images * ctx.tokens_per_image
    msg = f"token_acts rows {token_acts_csr.shape[0]} != expected {n_tokens_expected}."
    assert token_acts_csr.shape[0] == n_tokens_expected, msg

    unique_feature_ids = sorted(set(feature_ids))
    max_feature_id = max(unique_feature_ids)
    msg = f"Feature id {max_feature_id} out of range for token_acts width {token_acts_csr.shape[1]}."
    assert max_feature_id < token_acts_csr.shape[1], msg

    token_acts_csc = token_acts_csr.tocsc()
    n_features_rendered = 0
    n_images_written = 0
    n_features = len(unique_feature_ids)

    for start in range(0, n_features, feature_chunk_size):
        stop = min(start + feature_chunk_size, n_features)
        chunk_feature_ids = unique_feature_ids[start:stop]
        chunk_acts_np = (
            token_acts_csc[:, chunk_feature_ids]
            .toarray()
            .astype(np.float32, copy=False)
        )
        msg = (
            f"Chunk shape mismatch: {chunk_acts_np.shape} != "
            f"({n_tokens_expected}, {len(chunk_feature_ids)})."
        )
        assert chunk_acts_np.shape == (n_tokens_expected, len(chunk_feature_ids)), msg

        for local_i, feature_id in enumerate(chunk_feature_ids):
            n_written = _render_feature(
                ctx,
                run_root_dpath=run_root_dpath,
                run_id=run_id,
                shard_id=shard_id,
                feature_id=feature_id,
                feature_acts_np=chunk_acts_np[:, local_i],
                n_per_class=n_per_class,
                task_name=task_name,
            )
            n_images_written += n_written
            n_features_rendered += 1

        logger.info(
            "Rendered run '%s' chunk [%d:%d] (%d features).",
            run_id,
            start,
            stop,
            len(chunk_feature_ids),
        )

    return n_features_rendered, n_images_written
