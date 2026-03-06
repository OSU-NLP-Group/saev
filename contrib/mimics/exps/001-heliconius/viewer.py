import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import pathlib

    import marimo as mo
    import polars as pl

    mimics_consistency = None
    try:
        from mimics import consistency as _mimics_consistency
    except Exception:
        pass
    else:
        mimics_consistency = _mimics_consistency
    return json, mimics_consistency, mo, pathlib, pl


@app.cell
def _(mo):
    mo.md("""
    # Cambridge Mimicry Viewer

    Browse rendered mimic feature overlays written by `contrib/mimics/launch.py render`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Quick Start

    1. Render images first:
       `uv run python contrib/mimics/launch.py render --sweep contrib/mimics/exps/001-heliconius/render.py`
    2. Precompute consistency scores:
       `uv run python contrib/mimics/launch.py consistency`
    3. In this notebook, set `Runs Root` and `Shard ID`.
    4. Optionally set `Task contains` to filter tasks.
    5. Click **Refresh Index**.
    6. Switch `Feature order` to `consistency` to walk stable features first.
    7. Pick a `Run ID`, then use **Prev**/**Next** to choose a `Feature ID`.
    8. Pick a `Strip` (`erato/top`, `erato/bottom`, `melpomene/top`, `melpomene/bottom`).
    9. Adjust `Columns`.
    10. View the gallery at the bottom.
    """)
    return


@app.cell
def _(mo):
    run_root_dpath_ui = mo.ui.text(
        value="/fs/ess/PAS2136/samuelstevens/saev/runs",
        label="Runs Root",
        full_width=True,
    )
    shard_id_ui = mo.ui.text(value="79239bdd", label="Shard ID")
    task_filter_ui = mo.ui.text(
        value="",
        label="Task contains (optional)",
        full_width=True,
    )
    refresh_btn = mo.ui.run_button(label="Refresh Index")
    feature_order_ui = mo.ui.dropdown(
        ["feature_id", "consistency"], value="feature_id", label="Feature order"
    )
    n_cols_ui = mo.ui.slider(
        1,
        6,
        value=2,
        step=1,
        label="Columns",
        full_width=True,
    )
    strip_ui = mo.ui.dropdown(
        [
            "erato/top",
            "erato/bottom",
            "melpomene/top",
            "melpomene/bottom",
        ],
        value="erato/top",
        label="Strip",
    )
    return (
        feature_order_ui,
        n_cols_ui,
        refresh_btn,
        run_root_dpath_ui,
        shard_id_ui,
        strip_ui,
        task_filter_ui,
    )


@app.cell
def _(mo, refresh_btn):
    mo.hstack([refresh_btn], justify="start")
    return


@app.cell
def _(mo, run_root_dpath_ui, shard_id_ui, task_filter_ui):
    mo.md(f"""
    {run_root_dpath_ui}
    {shard_id_ui}
    {task_filter_ui}
    """)
    return


@app.cell
def _(json, mimics_consistency, pathlib, pl):
    def get_empty_index_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "run_id": pl.String,
                "feature_id": pl.Int64,
                "task": pl.String,
                "n_per_class": pl.Int64,
                "n_records": pl.Int64,
                "n_images_written": pl.Int64,
                "done_exists": pl.Boolean,
                "render_meta_fpath": pl.String,
            }
        )

    def get_empty_records_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "direction": pl.String,
                "class_name": pl.String,
                "rank": pl.Int64,
                "example_idx": pl.Int64,
                "subspecies_view": pl.String,
                "max_activation": pl.Float64,
                "img_fpath": pl.String,
            }
        )

    def get_empty_consistency_df() -> pl.DataFrame:
        if mimics_consistency is not None:
            return mimics_consistency.get_empty_consistency_df()
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

    def get_consistency_fpath(
        run_root_dpath: pathlib.Path, *, run_id: str, shard_id: str
    ) -> pathlib.Path:
        if mimics_consistency is not None and run_id and shard_id:
            return mimics_consistency.get_consistency_fpath(
                run_root_dpath, run_id=run_id, shard_id=shard_id
            )
        return (
            run_root_dpath
            / run_id
            / "inference"
            / shard_id
            / "cambridge-mimics-consistency.parquet"
        )

    def load_consistency_df(
        run_root_dpath: pathlib.Path,
        *,
        run_id: str,
        shard_id: str,
        task_names: list[str],
    ) -> tuple[pl.DataFrame, pathlib.Path]:
        run_id = run_id.strip()
        shard_id = shard_id.strip()
        consistency_fpath = get_consistency_fpath(
            run_root_dpath, run_id=run_id, shard_id=shard_id
        )
        if not run_id or not shard_id or mimics_consistency is None:
            return get_empty_consistency_df(), consistency_fpath
        consistency_df = mimics_consistency.load_consistency_df(
            run_root_dpath,
            run_id=run_id,
            shard_id=shard_id,
            task_names=task_names,
        )
        return consistency_df, consistency_fpath

    def load_render_index(
        run_root_dpath: pathlib.Path, *, shard_id: str, task_filter: str
    ) -> pl.DataFrame:
        if not run_root_dpath.is_dir():
            return get_empty_index_df()

        rows = []
        task_filter = task_filter.strip()
        pattern = f"*/inference/{shard_id}/images/*/cambridge-mimics/_render.json"
        for render_meta_fpath in sorted(run_root_dpath.glob(pattern)):
            try:
                with open(render_meta_fpath) as fd:
                    payload = json.load(fd)
            except Exception:
                continue

            task = str(payload.get("task", ""))
            if task_filter and task_filter not in task:
                continue

            feature_root_dpath = render_meta_fpath.parent
            done_fpath = feature_root_dpath / "_done.json"

            feature_id = int(feature_root_dpath.parent.name)
            run_id = str(feature_root_dpath.parent.parent.parent.parent.parent.name)

            records = payload.get("records", [])
            rows.append({
                "run_id": run_id,
                "feature_id": feature_id,
                "task": task,
                "n_per_class": int(payload.get("n_per_class", 0)),
                "n_records": int(len(records)),
                "n_images_written": int(payload.get("n_images_written", 0)),
                "done_exists": done_fpath.exists(),
                "render_meta_fpath": str(render_meta_fpath),
            })

        if not rows:
            return get_empty_index_df()

        return pl.DataFrame(rows, infer_schema_length=None).sort(
            "task", "run_id", "feature_id"
        )

    def load_feature_records(render_meta_fpath: pathlib.Path) -> pl.DataFrame:
        if not render_meta_fpath.exists():
            return get_empty_records_df()

        try:
            with open(render_meta_fpath) as fd:
                payload = json.load(fd)
        except Exception:
            return get_empty_records_df()

        records = payload.get("records", [])
        if not isinstance(records, list) or not records:
            return get_empty_records_df()

        rows = []
        for record in records:
            try:
                rows.append({
                    "direction": str(record.get("direction", "")),
                    "class_name": str(record.get("class_name", "")),
                    "rank": int(record.get("rank", -1)),
                    "example_idx": int(record.get("example_idx", -1)),
                    "subspecies_view": str(record.get("subspecies_view", "")),
                    "max_activation": float(record.get("max_activation", 0.0)),
                    "img_fpath": str(record.get("img_fpath", "")),
                })
            except Exception:
                continue

        if not rows:
            return get_empty_records_df()

        return pl.DataFrame(rows, infer_schema_length=None).sort(
            "class_name", "direction", "rank"
        )

    return (
        get_empty_index_df,
        load_consistency_df,
        load_feature_records,
        load_render_index,
    )


@app.cell
def _(
    get_empty_index_df,
    load_render_index,
    mo,
    pathlib,
    refresh_btn,
    run_root_dpath_ui,
    shard_id_ui,
    task_filter_ui,
):
    run_root_dpath = pathlib.Path(run_root_dpath_ui.value).expanduser()
    if not refresh_btn.value:
        _ = mo.md("Click **Refresh Index** to scan rendered outputs.")
        index_df = get_empty_index_df()
    else:
        index_df = load_render_index(
            run_root_dpath,
            shard_id=shard_id_ui.value,
            task_filter=task_filter_ui.value,
        )
    return (index_df,)


@app.cell
def _(index_df, mo):
    mo.md(f"""
    Indexed rendered features: `{index_df.height}`
    """)
    return


@app.cell
def _(index_df):
    index_df.head(25)
    return


@app.cell
def _(index_df):
    run_id_options = sorted(set(index_df.get_column("run_id").to_list()))
    return (run_id_options,)


@app.cell
def _(mo, run_id_options):
    if run_id_options:
        run_id_ui = mo.ui.dropdown(
            run_id_options,
            value=run_id_options[0],
            label="Run ID",
            searchable=True,
            full_width=True,
        )
    else:
        run_id_ui = mo.ui.dropdown([""], value="", label="Run ID", full_width=True)
    return (run_id_ui,)


@app.cell
def _(index_df, pl, run_id_ui):
    run_index_df = index_df.filter(pl.col("run_id") == str(run_id_ui.value))
    return (run_index_df,)


@app.cell
def _(
    load_consistency_df,
    pathlib,
    run_id_ui,
    run_index_df,
    run_root_dpath_ui,
    shard_id_ui,
):
    _run_root_dpath = pathlib.Path(run_root_dpath_ui.value).expanduser()
    task_names = sorted(set(run_index_df.get_column("task").to_list()))
    consistency_df, consistency_fpath = load_consistency_df(
        _run_root_dpath,
        run_id=str(run_id_ui.value),
        shard_id=str(shard_id_ui.value),
        task_names=task_names,
    )
    return consistency_df, consistency_fpath


@app.cell
def _(consistency_df, consistency_fpath, mo, run_id_ui):
    run_id = str(run_id_ui.value).strip()
    if not run_id:
        summary = mo.md("Consistency rows: select a run.")
    elif not consistency_fpath.is_file():
        summary = mo.md(f"""
        Consistency rows: not found at `{consistency_fpath}`.
        Run:
        `uv run python contrib/mimics/launch.py consistency --run-ids {run_id}`
        """)
    else:
        summary = mo.md(f"""
        Consistency rows: `{consistency_df.height}`
        Source: `{consistency_fpath}`
        """)
    summary
    return


@app.cell
def _(consistency_df):
    consistency_df.head(25)
    return


@app.cell
def _(consistency_df, feature_order_ui, pl, run_index_df):
    feature_ids = run_index_df.get_column("feature_id").to_list()
    feature_ids = sorted({int(feature_id) for feature_id in feature_ids})
    if feature_order_ui.value != "consistency" or not consistency_df.height:
        feature_id_options = feature_ids
    else:
        ranking_df = (
            consistency_df
            .group_by("feature_id")
            .agg(pl.col("consistency").max().alias("consistency"))
            .sort("consistency", "feature_id", descending=[True, False])
        )
        ranked_feature_ids = [
            int(feature_id)
            for feature_id in ranking_df.get_column("feature_id").to_list()
        ]
        seen = set(ranked_feature_ids)
        feature_id_options = ranked_feature_ids + [
            feature_id for feature_id in feature_ids if feature_id not in seen
        ]
    return (feature_id_options,)


@app.cell
def _(mo):
    get_feature_i, set_feature_i = mo.state(0)
    return get_feature_i, set_feature_i


@app.cell
def _(feature_order_ui, mo, n_cols_ui, run_id_ui, strip_ui):
    mo.hstack([run_id_ui, strip_ui, feature_order_ui, n_cols_ui], justify="start")
    return


@app.cell
def _(feature_id_options, get_feature_i, set_feature_i):
    n_features = len(feature_id_options)
    if n_features:
        feature_i = int(get_feature_i())
        if feature_i < 0:
            feature_i = 0
            set_feature_i(feature_i)
        if feature_i >= n_features:
            feature_i = n_features - 1
            set_feature_i(feature_i)
        selected_feature_id = int(feature_id_options[feature_i])
    else:
        feature_i = 0
        selected_feature_id = -1
    return feature_i, n_features, selected_feature_id


@app.cell
def _(feature_id_options, mo, set_feature_i):
    def on_prev(_):
        if not feature_id_options:
            return
        set_feature_i(lambda i: (i - 1) % len(feature_id_options))

    def on_next(_):
        if not feature_id_options:
            return
        set_feature_i(lambda i: (i + 1) % len(feature_id_options))

    prev_feature_btn = mo.ui.button(label="Prev", on_change=on_prev)
    next_feature_btn = mo.ui.button(label="Next", on_change=on_next)
    return next_feature_btn, prev_feature_btn


@app.cell
def _(
    feature_i,
    mo,
    n_features,
    next_feature_btn,
    prev_feature_btn,
    selected_feature_id,
):
    if not n_features:
        feature_nav = mo.hstack(
            [prev_feature_btn, next_feature_btn, mo.md("Feature ID: none")],
            justify="start",
        )
    else:
        feature_nav = mo.hstack(
            [
                prev_feature_btn,
                next_feature_btn,
                mo.md(
                    f"Feature ID: `{selected_feature_id}` ({feature_i + 1}/{n_features})"
                ),
            ],
            justify="start",
        )
    feature_nav
    return


@app.cell
def _(index_df, pathlib, pl, run_id_ui, selected_feature_id):
    selected_df = (
        index_df
        .filter(
            (pl.col("run_id") == str(run_id_ui.value))
            & (pl.col("feature_id") == int(selected_feature_id))
        )
        .sort("task")
        .head(1)
    )

    if not selected_df.height:
        render_meta_fpath = pathlib.Path("/")
    else:
        render_meta_fpath = pathlib.Path(
            str(selected_df.get_column("render_meta_fpath").item())
        )
    return render_meta_fpath, selected_df


@app.cell
def _(selected_df):
    _ = selected_df
    return


@app.cell
def _(load_feature_records, render_meta_fpath):
    records_df = load_feature_records(render_meta_fpath)
    return (records_df,)


@app.cell
def _(records_df):
    _ = records_df
    return


@app.cell
def _(mo, pathlib):
    def make_img_card(img_fpath: pathlib.Path, *, label: str):
        if not img_fpath.exists():
            return mo.vstack([mo.md(label), mo.md("Missing image")], gap=0.25)
        return mo.vstack([mo.image(img_fpath, width="100%"), mo.md(label)], gap=0.25)

    return (make_img_card,)


@app.cell
def _(make_img_card, mo, n_cols_ui, pathlib, pl, records_df, strip_ui):
    if not records_df.height:
        gallery_view = mo.md("No image records for this selection.")
    else:
        n_cols = int(n_cols_ui.value)
        msg = "n_cols must be >= 1."
        assert n_cols >= 1, msg
        class_name, direction = str(strip_ui.value).split("/", maxsplit=1)
        subset_df = records_df.filter(
            (pl.col("class_name") == class_name) & (pl.col("direction") == direction)
        ).sort("rank")

        cards = []
        for row in subset_df.iter_rows(named=True):
            img_fpath = pathlib.Path(str(row["img_fpath"]))
            label = (
                f"rank={int(row['rank'])}  \n"
                f"ssp={str(row['subspecies_view'])}  \n"
                f"max={float(row['max_activation']):.4f}"
            )
            cards.append(make_img_card(img_fpath, label=label))

        if not cards:
            gallery_view = mo.md("No images for this strip.")
        else:
            rows = []
            for start in range(0, len(cards), n_cols):
                rows.append(mo.hstack(cards[start : start + n_cols], widths="equal"))
            gallery_view = mo.vstack(
                [mo.md(f"### {class_name} / {direction}"), *rows], gap=0.8
            )

    gallery_view
    return


@app.cell
def _(records_df):
    records_df.head(25)
    return


if __name__ == "__main__":
    app.run()
