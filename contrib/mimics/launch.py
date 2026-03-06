"""
Launcher script for Cambridge mimicry utilities.
"""

import dataclasses
import logging
import pathlib
import typing as tp

import beartype
import polars as pl
import tyro
import tyro.extras
from mimics import checkpoints, consistency, features, render, tasks

import saev.configs

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("mimics.launch")

DEFAULT_RUN_IDS = ["zhul9opa", "gz2dikb3", "3rqci2h1", "r27w7pmf", "x4n29kua"]
DEFAULT_C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RenderCliConfig:
    """Configuration for local end-to-end Cambridge mimic rendering.

    This config controls checkpoint discovery, feature pooling, render planning,
    and image rendering for one task/run set.
    """

    run_root_dpath: pathlib.Path = pathlib.Path(
        "/fs/ess/PAS2136/samuelstevens/saev/runs"
    )
    """Root directory containing SAE run folders."""
    shards_dpath: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/79239bdd"
    )
    """Shards directory used for labels and image loading."""
    task_name: str = "lativitta_dorsal_vs_malleti_dorsal"
    """Mimic task name, format '{erato_ssp}_{view}_vs_{melp_ssp}_{view}'."""
    run_ids: list[str] = dataclasses.field(
        default_factory=lambda: DEFAULT_RUN_IDS.copy()
    )
    """Run IDs to search for classifier checkpoints."""
    c_values: list[float] = dataclasses.field(
        default_factory=lambda: DEFAULT_C_VALUES.copy()
    )
    """Allowed classifier C values. Empty means allow all."""
    n_features_min: int = 1
    """Minimum checkpoint sparsity (inclusive) for checkpoint filtering."""
    n_features_max: int = 30
    """Maximum checkpoint sparsity (inclusive) for checkpoint filtering."""
    top_k_ckpts: int = 20
    """Number of filtered checkpoints to keep before feature pooling."""
    n_per_class: int = 8
    """How many top and bottom non-zero images to render per class."""
    feature_chunk_size: int = 64
    """How many features to slice at once from CSC activations per run."""
    max_pooled_features: int = 0
    """Optional cap on pooled features (0 means no cap)."""
    force_render: bool = False
    """If True, re-render features even when _done.json is up to date."""
    dry_run: bool = False
    """If True, compute plan and log stats without writing images."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ConsistencyCliConfig:
    """Configuration for precomputing mimic feature consistency tables.

    This config controls discovery of rendered `(run, task, feature)` specs and
    computes consistency metrics into per-run parquet files for the viewer.
    """

    run_root_dpath: pathlib.Path = pathlib.Path(
        "/fs/ess/PAS2136/samuelstevens/saev/runs"
    )
    """Root directory containing SAE run folders."""
    shards_dpath: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/79239bdd"
    )
    """Shards directory used for labels during consistency scoring."""
    run_ids: list[str] = dataclasses.field(default_factory=list)
    """Optional run-id allowlist. Empty means include all discovered runs."""
    task_filter: str = ""
    """Optional substring filter applied to discovered task names."""
    top_k: int = 8
    """Top-k nonzero activations per class used for stability/strength terms."""
    n_bootstrap: int = 64
    """Bootstrap draws used by top-k stability estimation."""
    feature_chunk_size: int = 64
    """How many features to score at once from CSC activations."""
    force_recompute: bool = False
    """If True, recompute even when consistency rows already exist."""
    dry_run: bool = False
    """If True, plan and log pending work without writing outputs."""


@beartype.beartype
def render_worker_fn(cfg: RenderCliConfig) -> int:
    """Run one local render configuration."""

    run_root_dpath = cfg.run_root_dpath.expanduser()
    shards_dpath = cfg.shards_dpath.expanduser()
    shard_id = shards_dpath.name
    task_name = cfg.task_name.strip()

    msg = f"Runs root missing: '{run_root_dpath}'."
    assert run_root_dpath.is_dir(), msg
    msg = f"Shards directory missing: '{shards_dpath}'."
    assert shards_dpath.is_dir(), msg
    assert task_name, "task_name is required."
    tasks.parse_task_name(task_name)
    msg = "Provide at least one run id."
    assert cfg.run_ids, msg
    msg = "n_features_min must be >= 1."
    assert cfg.n_features_min >= 1, msg
    msg = "n_features_max must be >= n_features_min."
    assert cfg.n_features_max >= cfg.n_features_min, msg
    msg = "top_k_ckpts must be >= 1."
    assert cfg.top_k_ckpts >= 1, msg
    msg = "n_per_class must be >= 1."
    assert cfg.n_per_class >= 1, msg
    msg = "feature_chunk_size must be >= 1."
    assert cfg.feature_chunk_size >= 1, msg
    msg = "max_pooled_features must be >= 0."
    assert cfg.max_pooled_features >= 0, msg

    ckpt_cfg = checkpoints.DiscoverCheckpointsConfig(
        run_root_dpath=run_root_dpath,
        shard_id=shard_id,
        task_name=task_name,
        run_ids=cfg.run_ids,
        c_values=cfg.c_values,
    )
    ckpt_df = checkpoints.discover_checkpoints(ckpt_cfg)
    logger.info("Discovered %d checkpoints for task '%s'.", ckpt_df.height, task_name)
    if not ckpt_df.height:
        return 0

    filtered_ckpt_df = ckpt_df.filter(
        (pl.col("n_features") >= cfg.n_features_min)
        & (pl.col("n_features") <= cfg.n_features_max)
    ).sort("balanced_acc", "n_features", descending=[True, False])
    selected_ckpt_df = filtered_ckpt_df.head(cfg.top_k_ckpts)

    logger.info(
        "After filtering by n_features in [%d, %d], %d checkpoints remain.",
        cfg.n_features_min,
        cfg.n_features_max,
        filtered_ckpt_df.height,
    )
    logger.info(
        "Selected top-%d: %d checkpoints.", cfg.top_k_ckpts, selected_ckpt_df.height
    )
    if not selected_ckpt_df.height:
        return 0

    raw_feature_df, pooled_feature_df = features.make_feature_tables(selected_ckpt_df)
    if cfg.max_pooled_features:
        pooled_feature_df = pooled_feature_df.head(cfg.max_pooled_features)
    logger.info(
        "Feature pooling complete: raw=%d rows, pooled=%d rows.",
        raw_feature_df.height,
        pooled_feature_df.height,
    )
    if not pooled_feature_df.height:
        return 0

    render_plan_df = render.make_render_plan(
        pooled_feature_df,
        run_root_dpath=run_root_dpath,
        shard_id=shard_id,
        task_name=task_name,
        n_per_class=cfg.n_per_class,
        force_render=cfg.force_render,
    )
    n_render = int(render_plan_df.filter(pl.col("status") == "render").height)
    n_skip = int(render_plan_df.filter(pl.col("status") == "skip").height)
    n_missing_token_acts = int(
        render_plan_df.filter(pl.col("status") == "missing_token_acts").height
    )
    logger.info(
        "Render plan: render=%d, skip=%d, missing_token_acts=%d.",
        n_render,
        n_skip,
        n_missing_token_acts,
    )

    if cfg.dry_run:
        logger.info("Dry run enabled, skipping rendering.")
        return 0

    if not n_render:
        logger.info("No features need rendering.")
        return 0

    pending_by_run_dct = render.get_pending_features_by_run(render_plan_df)
    msg = "Render plan says work is needed, but no pending features were found."
    assert pending_by_run_dct, msg

    ctx = render.make_render_context(shards_dpath, task_name)
    n_runs = len(pending_by_run_dct)
    n_features_rendered = 0
    n_images_written = 0

    for i, run_id in enumerate(sorted(pending_by_run_dct.keys()), start=1):
        feature_ids = pending_by_run_dct[run_id]
        logger.info(
            "[%d/%d] Rendering run '%s' with %d features.",
            i,
            n_runs,
            run_id,
            len(feature_ids),
        )
        n_run_features, n_run_images = render.render_run_features(
            ctx,
            run_root_dpath=run_root_dpath,
            run_id=run_id,
            shard_id=shard_id,
            feature_ids=feature_ids,
            n_per_class=cfg.n_per_class,
            task_name=task_name,
            feature_chunk_size=cfg.feature_chunk_size,
        )
        n_features_rendered += n_run_features
        n_images_written += n_run_images
        logger.info(
            "[%d/%d] Finished run '%s': rendered %d features, wrote %d images.",
            i,
            n_runs,
            run_id,
            n_run_features,
            n_run_images,
        )

    logger.info(
        "Done. Rendered %d features and wrote %d images.",
        n_features_rendered,
        n_images_written,
    )
    return 0


@beartype.beartype
def _make_existing_key_set(existing_df: pl.DataFrame) -> set[tuple[str, int]]:
    required_cols = {"task", "feature_id"}
    if not required_cols <= set(existing_df.columns):
        return set()
    return {
        (str(row_dct["task"]), int(row_dct["feature_id"]))
        for row_dct in existing_df.select("task", "feature_id").iter_rows(named=True)
    }


@beartype.beartype
def consistency_worker_fn(cfg: ConsistencyCliConfig) -> int:
    """Run one local consistency configuration."""

    run_root_dpath = cfg.run_root_dpath.expanduser()
    shards_dpath = cfg.shards_dpath.expanduser()
    shard_id = shards_dpath.name

    msg = f"Runs root missing: '{run_root_dpath}'."
    assert run_root_dpath.is_dir(), msg
    msg = f"Shards directory missing: '{shards_dpath}'."
    assert shards_dpath.is_dir(), msg
    msg = "top_k must be >= 1."
    assert cfg.top_k >= 1, msg
    msg = "n_bootstrap must be >= 2."
    assert cfg.n_bootstrap >= 2, msg
    msg = "feature_chunk_size must be >= 1."
    assert cfg.feature_chunk_size >= 1, msg

    task_specs_df = consistency.discover_task_specs(
        run_root_dpath,
        shard_id=shard_id,
        run_ids=cfg.run_ids,
        task_filter=cfg.task_filter,
    )
    logger.info(
        "Discovered %d run/task specs from rendered outputs.", task_specs_df.height
    )
    if not task_specs_df.height:
        return 0

    run_ids = sorted(set(task_specs_df.get_column("run_id").to_list()))
    pending_specs: list[dict[str, object]] = []
    existing_df_by_run: dict[str, pl.DataFrame] = {}
    for run_id in run_ids:
        run_task_specs_df = task_specs_df.filter(pl.col("run_id") == run_id)
        token_acts_fpath = (
            run_root_dpath / run_id / "inference" / shard_id / "token_acts.npz"
        )
        if not token_acts_fpath.exists():
            logger.warning(
                "Skipping run '%s': missing token activations '%s'.",
                run_id,
                token_acts_fpath,
            )
            continue

        existing_df = consistency.load_consistency_df(
            run_root_dpath, run_id=run_id, shard_id=shard_id
        )
        existing_df_by_run[run_id] = existing_df
        existing_keys = (
            set() if cfg.force_recompute else _make_existing_key_set(existing_df)
        )
        n_expected = 0
        n_pending = 0
        for row_dct in run_task_specs_df.iter_rows(named=True):
            task_name = str(row_dct["task"])
            feature_ids = sorted({
                int(feature_id) for feature_id in row_dct["feature_ids"]
            })
            n_expected += len(feature_ids)
            if cfg.force_recompute:
                pending_feature_ids = feature_ids
            else:
                pending_feature_ids = [
                    feature_id
                    for feature_id in feature_ids
                    if (task_name, feature_id) not in existing_keys
                ]

            if not pending_feature_ids:
                continue
            n_pending += len(pending_feature_ids)
            pending_specs.append({
                "run_id": run_id,
                "task": task_name,
                "feature_ids": pending_feature_ids,
            })

        logger.info(
            "Run '%s' task-spec plan: expected=%d, pending=%d.",
            run_id,
            n_expected,
            n_pending,
        )

    if not pending_specs:
        logger.info("No consistency work is pending.")
        return 0

    if cfg.dry_run:
        logger.info("Dry run enabled, skipping scoring and writes.")
        return 0

    scored_df_l_by_run: dict[str, list[pl.DataFrame]] = {}
    n_specs = len(pending_specs)
    for i, task_spec in enumerate(pending_specs, start=1):
        run_id = str(task_spec["run_id"])
        task_name = str(task_spec["task"])
        feature_ids = [int(feature_id) for feature_id in task_spec["feature_ids"]]
        logger.info(
            "[%d/%d] Scoring run='%s' task='%s' features=%d.",
            i,
            n_specs,
            run_id,
            task_name,
            len(feature_ids),
        )
        score_cfg = consistency.ScoreFeaturesConfig(
            run_root_dpath=run_root_dpath,
            shards_dpath=shards_dpath,
            run_id=run_id,
            shard_id=shard_id,
            task_name=task_name,
            feature_ids=feature_ids,
            top_k=cfg.top_k,
            n_bootstrap=cfg.n_bootstrap,
            feature_chunk_size=cfg.feature_chunk_size,
        )
        try:
            scored_df = consistency.score_features(score_cfg)
        except Exception as err:
            logger.warning(
                "Failed scoring run='%s' task='%s': %s", run_id, task_name, err
            )
            continue
        if not scored_df.height:
            continue
        if run_id not in scored_df_l_by_run:
            scored_df_l_by_run[run_id] = []
        scored_df_l_by_run[run_id].append(scored_df)

    if not scored_df_l_by_run:
        logger.warning("No consistency scores were produced.")
        return 1

    n_runs_written = 0
    n_rows_written = 0
    for run_id, scored_df_l in sorted(scored_df_l_by_run.items()):
        new_df = pl.concat(scored_df_l, how="vertical_relaxed")
        existing_df = (
            consistency.get_empty_consistency_df()
            if cfg.force_recompute
            else existing_df_by_run.get(run_id, consistency.get_empty_consistency_df())
        )
        if existing_df.height:
            overlap_key_df = new_df.select("task", "feature_id").unique(
                maintain_order=True
            )
            existing_keep_df = existing_df.join(
                overlap_key_df, on=["task", "feature_id"], how="anti"
            )
        else:
            existing_keep_df = existing_df

        merged_df = pl.concat([existing_keep_df, new_df], how="vertical_relaxed").sort(
            "consistency", "selectivity", "feature_id", descending=[True, True, False]
        )
        out_fpath = consistency.write_consistency_df(
            merged_df, run_root_dpath, run_id=run_id, shard_id=shard_id
        )
        n_runs_written += 1
        n_rows_written += merged_df.height
        logger.info(
            "Wrote %d rows for run '%s' to '%s'.",
            merged_df.height,
            run_id,
            out_fpath,
        )

    logger.info(
        "Done. Wrote consistency tables for %d runs (%d total rows).",
        n_runs_written,
        n_rows_written,
    )
    return 0


@beartype.beartype
def render_cli(
    cfg: tp.Annotated[RenderCliConfig, tyro.conf.arg(name="")],
    sweep: pathlib.Path | None = None,
) -> int:
    """Render mimic feature overlays locally for one config or a sweep of configs.

    Args:
        cfg: Base render configuration. Used directly when `sweep` is not provided, or as the override baseline when `sweep` is provided.
        sweep: Optional path to a Python sweep file with `make_cfgs() -> list[dict]`.
    """

    if sweep is None:
        cfgs = [cfg]
    else:
        sweep_dcts = saev.configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            return 1

        cfgs, errs = saev.configs.load_cfgs(
            cfg, default=RenderCliConfig(), sweep_dcts=sweep_dcts
        )
        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return 1

    if not cfgs:
        logger.error("No render configs resolved.")
        return 1

    logger.info("Prepared %d config(s).", len(cfgs))
    for i, cfg_item in enumerate(cfgs, start=1):
        logger.info("Running config %d/%d locally.", i, len(cfgs))
        render_worker_fn(cfg_item)

    logger.info("Jobs done.")
    return 0


@beartype.beartype
def consistency_cli(
    cfg: tp.Annotated[ConsistencyCliConfig, tyro.conf.arg(name="")],
    sweep: pathlib.Path | None = None,
) -> int:
    """Precompute consistency tables for one config or a sweep.

    Args:
        cfg: Base consistency configuration. Used directly when `sweep` is not provided, or as the override baseline when `sweep` is provided.
        sweep: Optional path to a Python sweep file with `make_cfgs() -> list[dict]`.
    """

    if sweep is None:
        cfgs = [cfg]
    else:
        sweep_dcts = saev.configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            return 1

        cfgs, errs = saev.configs.load_cfgs(
            cfg, default=ConsistencyCliConfig(), sweep_dcts=sweep_dcts
        )
        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return 1

    if not cfgs:
        logger.error("No consistency configs resolved.")
        return 1

    logger.info("Prepared %d config(s).", len(cfgs))
    for i, cfg_item in enumerate(cfgs, start=1):
        logger.info("Running config %d/%d locally.", i, len(cfgs))
        consistency_worker_fn(cfg_item)

    logger.info("Jobs done.")
    return 0


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "render": render_cli,
        "consistency": consistency_cli,
    })
