import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import beartype
    import marimo as mo
    import mimics.checkpoints as mimics_checkpoints
    import mimics.features as mimics_features
    import mimics.render as mimics_render
    import mimics.tasks as mimics_tasks
    import numpy as np
    import polars as pl
    from tdiscovery.classification import (
        LabelGrouping,
        apply_grouping,
        load_image_labels,
    )

    import saev.data
    import saev.data.datasets
    import saev.data.models

    return (
        LabelGrouping,
        apply_grouping,
        beartype,
        load_image_labels,
        mimics_checkpoints,
        mimics_features,
        mimics_render,
        mimics_tasks,
        mo,
        np,
        pathlib,
        pl,
        saev,
    )


@app.cell
def _(mo):
    mo.md("""
    # Cambridge Mimicry Exploration

    This notebook is the step-by-step implementation workspace for `contrib/mimics/spec.md`.
    """)
    return


@app.cell
def _(mo, pathlib):
    spec_fpath = pathlib.Path(__file__).resolve().parents[1] / "spec.md"
    if not spec_fpath.exists():
        _ = mo.md(f"Spec file not found: `{spec_fpath}`")
    else:
        _ = mo.md(spec_fpath.read_text())
    return


@app.cell
def _(mo):
    run_root_dpath_ui = mo.ui.text(
        value="/fs/ess/PAS2136/samuelstevens/saev/runs",
        label="Runs Root",
        full_width=True,
    )
    run_ids_ui = mo.ui.text_area(
        value="zhul9opa\ngz2dikb3\n3rqci2h1\nr27w7pmf\nx4n29kua",
        label="Run IDs (newline or comma separated)",
        full_width=True,
    )
    shards_dpath_ui = mo.ui.text(
        value="/fs/scratch/PAS2136/samuelstevens/saev/shards/79239bdd",
        label="Shards Directory",
        full_width=True,
    )
    c_values_ui = mo.ui.text(
        value="0.00001,0.0001,0.001,0.01,0.1",
        label="C Values (optional, comma separated)",
        full_width=True,
    )
    min_samples_per_class_ui = mo.ui.number(
        start=1, step=1, value=50, label="Min samples per class for task specs"
    )
    load_task_specs_btn = mo.ui.run_button(label="Load Task Specs")
    load_ckpts_btn = mo.ui.run_button(label="Step 1: Load Classifier Checkpoints")
    return (
        c_values_ui,
        load_ckpts_btn,
        load_task_specs_btn,
        min_samples_per_class_ui,
        run_ids_ui,
        run_root_dpath_ui,
        shards_dpath_ui,
    )


@app.cell
def _(
    c_values_ui,
    load_ckpts_btn,
    load_task_specs_btn,
    min_samples_per_class_ui,
    mo,
    run_ids_ui,
    run_root_dpath_ui,
    shards_dpath_ui,
):
    mo.hstack(
        [min_samples_per_class_ui, load_task_specs_btn, load_ckpts_btn],
        justify="start",
    )
    mo.md(f"""{run_root_dpath_ui}
    {run_ids_ui}
    {shards_dpath_ui}
    {c_values_ui}
    """)
    return


@app.cell
def _(beartype):
    @beartype.beartype
    def parse_items(raw: str) -> list[str]:
        return [
            item
            for item in (piece.strip() for piece in raw.replace(",", "\n").splitlines())
            if item
        ]

    @beartype.beartype
    def parse_c_values(raw: str) -> list[float]:
        return [float(item) for item in parse_items(raw)]

    return parse_c_values, parse_items


@app.cell
def _(
    load_task_specs_btn,
    mimics_tasks,
    min_samples_per_class_ui,
    mo,
    pathlib,
    shards_dpath_ui,
):
    _shards_dpath = pathlib.Path(shards_dpath_ui.value).expanduser()

    if not load_task_specs_btn.value:
        _ = mo.md("Click **Load Task Specs** to compute task candidates in memory.")
        task_specs = []
        task_summary_df = mimics_tasks.get_empty_summary_df()
    else:
        _cfg = mimics_tasks.DecideTaskSpecsConfig(
            shards_dpath=_shards_dpath,
            min_samples_per_class=int(min_samples_per_class_ui.value),
        )
        task_specs, task_summary_df = mimics_tasks.decide_task_specs(_cfg)

    task_name_options = [spec.task_name for spec in task_specs]
    return task_name_options, task_specs, task_summary_df


@app.cell
def _(mo, task_name_options):
    if task_name_options:
        task_name_ui = mo.ui.dropdown(
            task_name_options,
            value=task_name_options[0],
            label="Task Name",
            searchable=True,
            full_width=True,
        )
    else:
        task_name_ui = mo.ui.dropdown(
            [""],
            value="",
            label="Task Name",
            searchable=True,
            full_width=True,
        )
    return (task_name_ui,)


@app.cell
def _(mo, task_name_ui):
    mo.hstack([task_name_ui], justify="start")
    return


@app.cell
def _(task_summary_df):
    task_summary_df
    return


@app.cell
def _(task_specs):
    task_spec_by_name = {spec.task_name: spec for spec in task_specs}
    return (task_spec_by_name,)


@app.cell
def _(
    LabelGrouping,
    c_values_ui,
    load_ckpts_btn,
    mimics_checkpoints,
    mo,
    parse_c_values,
    parse_items,
    pathlib,
    run_ids_ui,
    run_root_dpath_ui,
    shards_dpath_ui,
    task_name_ui,
    task_spec_by_name,
):
    task_name = str(task_name_ui.value).strip()
    task_spec = task_spec_by_name.get(task_name)
    run_root_dpath = pathlib.Path(run_root_dpath_ui.value).expanduser()
    shards_dpath = pathlib.Path(shards_dpath_ui.value).expanduser()
    shard_id = shards_dpath.name
    run_ids = parse_items(run_ids_ui.value)
    c_values = parse_c_values(c_values_ui.value)
    task_grouping = (
        LabelGrouping(
            name=task_spec.task_name,
            source_col=task_spec.source_col,
            groups=task_spec.groups,
        )
        if task_spec is not None
        else LabelGrouping(
            name="",
            source_col="subspecies_view",
            groups={"erato": [], "melpomene": []},
        )
    )

    if not load_ckpts_btn.value:
        _ = mo.md("Click **Step 1: Load Classifier Checkpoints** to scan runs.")
        ckpt_df = mimics_checkpoints.get_empty_ckpt_df()
    else:
        _msg = "Select a task from the task-spec dropdown before loading checkpoints."
        assert task_spec is not None, _msg
        _msg = f"Runs root does not exist: '{run_root_dpath}'."
        assert run_root_dpath.is_dir(), _msg
        _msg = f"Shards directory does not exist: '{shards_dpath}'."
        assert shards_dpath.is_dir(), _msg
        _msg = "Provide at least one run id."
        assert run_ids, _msg

        _cfg = mimics_checkpoints.DiscoverCheckpointsConfig(
            run_root_dpath=run_root_dpath,
            shard_id=shard_id,
            task_name=task_spec.task_name,
            run_ids=run_ids,
            c_values=c_values,
        )
        ckpt_df = mimics_checkpoints.discover_checkpoints(_cfg)
        task_name = task_spec.task_name
    return (
        ckpt_df,
        run_ids,
        run_root_dpath,
        shard_id,
        shards_dpath,
        task_grouping,
        task_name,
    )


@app.cell
def _(ckpt_df, mo, run_ids, task_name):
    if not ckpt_df.height:
        _ = mo.md(
            f"Step 1 scanned `{len(run_ids)}` runs for task `{task_name}` and found no matching checkpoints."
        )
    else:
        _ = mo.md(
            f"## Step 1: Classifier Table\nLoaded `{ckpt_df.height}` checkpoints across `{len(set(ckpt_df.get_column('run_id').to_list()))}` runs."
        )
    return


@app.cell
def _(ckpt_df, mo):
    max_n_features = (
        int(ckpt_df.get_column("n_features").max()) if ckpt_df.height > 0 else 1
    )
    max_k_ckpts = int(ckpt_df.height) if ckpt_df.height > 0 else 1
    n_features_range_ui = mo.ui.range_slider(
        start=1,
        stop=max_n_features,
        step=1,
        value=[1, min(30, max_n_features)],
        label="n_features range",
    )
    top_k_ckpts_ui = mo.ui.slider(
        1,
        max_k_ckpts,
        value=min(20, max_k_ckpts),
        label="Top-K checkpoints to keep after filtering",
        full_width=True,
    )
    return n_features_range_ui, top_k_ckpts_ui


@app.cell
def _(mo, n_features_range_ui, top_k_ckpts_ui):
    mo.hstack([n_features_range_ui, top_k_ckpts_ui], justify="start")
    return


@app.cell
def _(ckpt_df, n_features_range_ui, pl, top_k_ckpts_ui):
    n_features_min = int(n_features_range_ui.value[0])
    n_features_max = int(n_features_range_ui.value[1])

    filtered_ckpt_df = ckpt_df.filter(
        (pl.col("n_features") >= n_features_min)
        & (pl.col("n_features") <= n_features_max)
    ).sort("balanced_acc", "n_features", descending=[True, False])
    selected_ckpt_df = filtered_ckpt_df.head(int(top_k_ckpts_ui.value))

    step1_display_df = filtered_ckpt_df.select(
        "run_id",
        "C",
        "n_features",
        "balanced_acc",
        "test_acc",
        "ckpt_fpath",
    )
    step1_selected_display_df = selected_ckpt_df.select(
        "run_id", "C", "n_features", "balanced_acc", "test_acc"
    )
    return selected_ckpt_df, step1_display_df, step1_selected_display_df


@app.cell
def _(mo, step1_display_df, step1_selected_display_df):
    mo.md(
        f"Filtered checkpoints: `{step1_display_df.height}` | Top-K selected for feature pooling: `{step1_selected_display_df.height}`"
    )
    step1_display_df
    mo.md("Top-K used in Step 2:")
    step1_selected_display_df
    return


@app.cell
def _(mimics_features, selected_ckpt_df):
    raw_feature_df, feature_df = mimics_features.make_feature_tables(selected_ckpt_df)
    return feature_df, raw_feature_df


@app.cell
def _(feature_df, mo, raw_feature_df):
    mo.md(
        f"## Step 2: Pooled Features\nRaw rows: `{raw_feature_df.height}` | pooled unique `(run_id, feature_id)`: `{feature_df.height}`."
    )
    return


@app.cell
def _(feature_df):
    feature_df
    return


@app.cell
def _(mo):
    n_per_class_ui = mo.ui.number(
        start=1, stop=20, value=8, step=1, label="N per class for top/bottom strips"
    )
    build_render_plan_btn = mo.ui.run_button(label="Step 3: Build Render Plan")
    return build_render_plan_btn, n_per_class_ui


@app.cell
def _(build_render_plan_btn, mo, n_per_class_ui):
    mo.hstack([n_per_class_ui, build_render_plan_btn], justify="start")
    return


@app.cell
def _(
    apply_grouping,
    build_render_plan_btn,
    feature_df,
    load_image_labels,
    mimics_render,
    mo,
    n_per_class_ui,
    np,
    pl,
    run_root_dpath,
    saev,
    shard_id,
    shards_dpath,
    task_grouping,
):
    if not build_render_plan_btn.value:
        _ = mo.md("Click **Step 3: Build Render Plan** to prepare rendering inputs.")
        render_plan_df = mimics_render.get_empty_render_plan_df()
        render_prep_dct = {
            "n_images_total": 0,
            "n_images_in_task": 0,
            "n_erato": 0,
            "n_melpomene": 0,
            "content_tokens_per_example": 0,
            "patch_size": 0,
            "n_features": 0,
            "n_need_render": 0,
            "n_skip_render": 0,
            "n_missing_token_acts": 0,
        }
    elif feature_df.height == 0:
        _ = mo.md("Step 2 has no pooled features yet.")
        render_plan_df = mimics_render.get_empty_render_plan_df()
        render_prep_dct = {
            "n_images_total": 0,
            "n_images_in_task": 0,
            "n_erato": 0,
            "n_melpomene": 0,
            "content_tokens_per_example": 0,
            "patch_size": 0,
            "n_features": 0,
            "n_need_render": 0,
            "n_skip_render": 0,
            "n_missing_token_acts": 0,
        }
    else:
        n_per_class = int(n_per_class_ui.value)
        shard_md = saev.data.Metadata.load(shards_dpath)
        vit = saev.data.models.load_model_cls(shard_md.family)(shard_md.ckpt)
        patch_size = int(vit.patch_size)

        _, labels_by_col = load_image_labels(shards_dpath)
        _msg = "Expected 'subspecies_view' in loaded labels."
        assert "subspecies_view" in labels_by_col, _msg
        subspecies_view_n = labels_by_col["subspecies_view"]
        _msg = (
            f"Label count mismatch: {len(subspecies_view_n)} != {shard_md.n_examples}."
        )
        assert len(subspecies_view_n) == shard_md.n_examples, _msg

        targets_n, class_names, include_n = apply_grouping(
            subspecies_view_n, task_grouping
        )
        class_to_i = {name: i for i, name in enumerate(class_names)}
        _msg = f"Expected erato/melpomene classes, got {class_names}."
        assert {"erato", "melpomene"} <= set(class_to_i), _msg

        grouped_targets_n = targets_n[include_n]
        n_erato = int((grouped_targets_n == class_to_i["erato"]).sum())
        n_melpomene = int((grouped_targets_n == class_to_i["melpomene"]).sum())

        render_plan_df = mimics_render.make_render_plan(
            feature_df,
            run_root_dpath=run_root_dpath,
            shard_id=shard_id,
            task_name=task_grouping.name,
            n_per_class=n_per_class,
        )
        render_prep_dct = {
            "n_images_total": int(shard_md.n_examples),
            "n_images_in_task": int(np.sum(include_n)),
            "n_erato": n_erato,
            "n_melpomene": n_melpomene,
            "content_tokens_per_example": int(shard_md.content_tokens_per_example),
            "patch_size": patch_size,
            "n_features": int(feature_df.height),
            "n_need_render": int(
                render_plan_df.filter(pl.col("status") == "render").height
            ),
            "n_skip_render": int(
                render_plan_df.filter(pl.col("status") == "skip").height
            ),
            "n_missing_token_acts": int(
                render_plan_df.filter(pl.col("status") == "missing_token_acts").height
            ),
        }
    return render_plan_df, render_prep_dct


@app.cell
def _(mo, render_prep_dct):
    mo.md(
        f"""## Step 3: Render Plan

    - n_images_total: `{render_prep_dct["n_images_total"]}`
    - n_images_in_task: `{render_prep_dct["n_images_in_task"]}`
    - class counts: erato=`{render_prep_dct["n_erato"]}`, melpomene=`{render_prep_dct["n_melpomene"]}`
    - tokens_per_image: `{render_prep_dct["content_tokens_per_example"]}`
    - patch_size: `{render_prep_dct["patch_size"]}`
    - pooled features: `{render_prep_dct["n_features"]}`
    - need render: `{render_prep_dct["n_need_render"]}`
    - already up to date: `{render_prep_dct["n_skip_render"]}`
    - missing token_acts.npz: `{render_prep_dct["n_missing_token_acts"]}`
    """
    )
    return


@app.cell
def _(render_plan_df):
    render_plan_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 4 (Next)

    The renderer now lives in `contrib/mimics/launch.py render`.
    Next notebook pass should add a scrollable browser that reads rendered images from disk.
    """)
    return


if __name__ == "__main__":
    app.run()
