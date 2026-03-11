import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib
    import typing as tp

    import marimo as mo
    import numpy as np
    import polars as pl
    import scipy.sparse
    from PIL import Image
    from tdiscovery.classification import apply_grouping, load_image_labels
    from tdiscovery.visuals import make_seg

    import saev.data
    import saev.data.datasets
    import saev.data.models
    import saev.viz

    return (
        Image,
        apply_grouping,
        load_image_labels,
        make_seg,
        mo,
        np,
        pathlib,
        pl,
        saev,
        scipy,
        tp,
    )


@app.cell
def _(pathlib):
    run_root_dpath = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
    shards_dpath = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/a6be28a1"
    )
    shard_id = "a6be28a1"

    all_run_ids = [
        "u9i54ybv",
        "4gi47eoy",
        "iifhcp9z",
        "1pwpq6ue",
        "oqm8s8sq",
        "zgsifqrc",
        "kmlavddy",
        "qohobmx1",
        "uu4l9a1c",
        "qpsn8p39",
        "yo4x94aj",
        "nn46e0xn",
        "7yhyyxim",
        "7ywv33hl",
        "37yshx8v",
        "d9wkqlds",
    ]
    return all_run_ids, run_root_dpath, shard_id, shards_dpath


@app.cell
def _(mo):
    mo.md("""
    # 002-wider-saes: Feature Viewer

    Browse SAE features scored for mimic pair discrimination. Select a run and task, then pick features by selectivity to view per-class activation grids.
    """)
    return


@app.cell
def _(all_run_ids, mo, pl, run_root_dpath, shard_id):
    _dfs = []
    for _rid in all_run_ids:
        _pq = (
            run_root_dpath / _rid / "inference" / shard_id / "cambridge-mimics.parquet"
        )
        if _pq.exists():
            _dfs.append(pl.read_parquet(_pq))
    scores_df = pl.concat(_dfs) if _dfs else pl.DataFrame()
    mo.stop(scores_df.height == 0, mo.md("No score parquets found."))

    scores_df = scores_df.with_columns(
        pl.col("auroc").sub(0.5).abs().mul(2).clip(0, 1).alias("selectivity"),
        (pl.col("support_erato") - pl.col("support_melpomene"))
        .abs()
        .alias("support_diff"),
    )
    return (scores_df,)


@app.cell
def _(mo, scores_df):
    _tasks = sorted(scores_df.get_column("task").unique().to_list())
    task_dropdown = mo.ui.dropdown(
        _tasks, value=_tasks[0], label="Task", searchable=True
    )
    n_per_class_slider = mo.ui.slider(3, 20, value=6, label="Images per class")
    show_seg_toggle = mo.ui.switch(value=False, label="Show seg masks")
    mo.hstack(
        [task_dropdown, n_per_class_slider, show_seg_toggle],
        justify="start",
    )
    return n_per_class_slider, show_seg_toggle, task_dropdown


@app.cell
def _(mo, pl, scores_df, task_dropdown):
    filtered_df = scores_df.filter(
        pl.col("task") == task_dropdown.value
    ).sort("selectivity", descending=True)
    mo.stop(filtered_df.height == 0, mo.md("No features for this task."))

    feature_table = mo.ui.table(
        filtered_df
        .head(200)
        .select(
            "run_id",
            "feature_id",
            "selectivity",
            "auroc",
            "support_diff",
            "support_erato",
            "support_melpomene",
            "mean_act_erato",
            "mean_act_melpomene",
        )
        .to_pandas(),
        selection="single",
        label="Top features by selectivity",
    )
    feature_table
    return (feature_table,)


@app.cell
def _(feature_table, mo):
    _sel = feature_table.value
    mo.stop(
        _sel is None or len(_sel) == 0,
        mo.md("Select a feature from the table above."),
    )
    selected_run_id = str(_sel.iloc[0]["run_id"])
    selected_feature_id = int(_sel.iloc[0]["feature_id"])
    mo.md(f"Selected: run=`{selected_run_id}` feature=**{selected_feature_id}**")
    return selected_feature_id, selected_run_id


@app.cell
def _(
    apply_grouping,
    load_image_labels,
    mo,
    run_root_dpath,
    saev,
    scipy,
    selected_run_id,
    shard_id,
    shards_dpath,
    task_dropdown,
    tp,
):
    # Load render context: image dataset, labels, patch size.
    _md = saev.data.Metadata.load(shards_dpath)
    _model_cls = tp.cast(
        tp.Callable[[str], tp.Any], saev.data.models.load_model_cls(_md.family)
    )
    _vit = _model_cls(_md.ckpt)
    _resize_tr = _vit.make_resize(_md.ckpt, _md.content_tokens_per_example, scale=1.0)
    _data_cfg = tp.cast(saev.data.datasets.Config, _md.make_data_cfg())
    img_ds = saev.data.datasets.get_dataset(
        _data_cfg, data_transform=_resize_tr, mask_transform=_resize_tr
    )
    patch_size = int(_vit.patch_size)
    n_patches = _md.content_tokens_per_example
    pixel_agg = _md.pixel_agg
    bg_label = _data_cfg.bg_label

    _, _labels_by_col = load_image_labels(shards_dpath)
    sv_labels = _labels_by_col["subspecies_view"]
    n_images = len(sv_labels)

    from mimics import tasks as _tasks_mod

    _grouping = _tasks_mod.make_label_grouping(task_dropdown.value)
    _targets, _names, include_mask = apply_grouping(sv_labels, _grouping)
    _c2i = {name: i for i, name in enumerate(_names)}
    erato_i = _c2i["erato"]
    melp_i = _c2i["melpomene"]
    _inc_tgt = _targets[include_mask]
    erato_class_mask = _inc_tgt == erato_i
    melp_class_mask = _inc_tgt == melp_i

    # Load token_acts for selected run.
    _ta_fpath = (
        run_root_dpath / selected_run_id / "inference" / shard_id / "token_acts.npz"
    )
    msg = f"Missing token_acts: '{_ta_fpath}'."
    assert _ta_fpath.exists(), msg
    ta_csc = scipy.sparse.load_npz(str(_ta_fpath)).tocsc()
    tpi = ta_csc.shape[0] // n_images
    assert ta_csc.shape[0] == n_images * tpi

    mo.md(
        f"Loaded token_acts for `{selected_run_id}` (d_sae={ta_csc.shape[1]}, tpi={tpi}, patch_size={patch_size})."
    )
    return (
        bg_label,
        erato_class_mask,
        img_ds,
        include_mask,
        melp_class_mask,
        n_images,
        n_patches,
        patch_size,
        pixel_agg,
        sv_labels,
        ta_csc,
        tpi,
    )


@app.cell
def _(
    Image,
    bg_label,
    erato_class_mask,
    img_ds,
    include_mask,
    make_seg,
    melp_class_mask,
    mo,
    n_images,
    n_patches,
    n_per_class_slider,
    np,
    patch_size,
    pixel_agg,
    run_root_dpath,
    saev,
    selected_feature_id,
    selected_run_id,
    shard_id,
    show_seg_toggle,
    sv_labels,
    ta_csc,
    tpi,
):
    # Extract per-patch activations for this feature.
    _col = ta_csc[:, selected_feature_id].toarray().ravel()
    _by_image = _col.reshape(n_images, tpi)
    _max_acts = _by_image.max(axis=1)
    _upper = float(_by_image.max())

    _inc_acts = _by_image[include_mask]
    _inc_max = _max_acts[include_mask]
    _inc_i = np.where(include_mask)[0]

    _n = n_per_class_slider.value
    _n_cols = 3

    import glasbey

    _palette = [
        tuple(rgb) for rgb in glasbey.create_palette(palette_size=256, as_hex=False)
    ]

    # Cache dir: runs/{run_id}/inference/{shard_id}/images/{feature_id}/
    _cache_dpath = (
        run_root_dpath
        / selected_run_id
        / "inference"
        / shard_id
        / "images"
        / str(selected_feature_id)
    )
    _cache_dpath.mkdir(parents=True, exist_ok=True)

    def _render_images(img_i, patches):
        """Render highlighted + seg images, caching to disk."""
        _sae_fpath = _cache_dpath / f"{img_i}_sae_img.png"
        _seg_fpath = _cache_dpath / f"{img_i}_seg.png"

        if _sae_fpath.exists() and _seg_fpath.exists():
            return Image.open(_sae_fpath), Image.open(_seg_fpath)

        _sample = img_ds[img_i]
        assert isinstance(_sample, dict) and "data" in _sample
        _img = _sample["data"]
        assert isinstance(_img, Image.Image)

        if _sae_fpath.exists():
            _sae = Image.open(_sae_fpath)
        else:
            _sae = saev.viz.add_highlights(_img, patches, patch_size, upper=_upper)
            _sae.save(_sae_fpath)

        _seg = None
        if _seg_fpath.exists():
            _seg = Image.open(_seg_fpath)
        else:
            _seg_raw = _sample.get("patch_labels", None)
            if _seg_raw is not None:
                _seg = make_seg(
                    _seg_raw, n_patches, patch_size, pixel_agg, bg_label, _palette
                )
                _seg.save(_seg_fpath)

        return _sae, _seg

    def _get_top_bot(class_mask):
        _cls_max = _inc_max[class_mask]
        _order = np.argsort(-_cls_max)
        _top_i = _order[:_n]
        _nonzero = _order[_cls_max[_order] > 0]
        _bot_i = _nonzero[-_n:] if len(_nonzero) >= _n else _nonzero
        return _top_i, _bot_i

    _e_top, _e_bot = _get_top_bot(erato_class_mask)
    _m_top, _m_bot = _get_top_bot(melp_class_mask)

    # Pre-render all images with a single progress bar.
    _render_jobs = []
    for _indices, _mask in [
        (_e_top, erato_class_mask),
        (_m_top, melp_class_mask),
        (_e_bot, erato_class_mask),
        (_m_bot, melp_class_mask),
    ]:
        _cls_acts = _inc_acts[_mask]
        _cls_global_i = _inc_i[_mask]
        for _idx in _indices:
            _img_i = int(_cls_global_i[_idx])
            _patches = _cls_acts[_idx].astype(np.float32)
            _render_jobs.append((_img_i, _patches))

    _rendered = {}
    for _img_i, _patches in mo.status.progress_bar(
        _render_jobs, title="Rendering images"
    ):
        if _img_i not in _rendered:
            _rendered[_img_i] = _render_images(_img_i, _patches)

    def _build_grid(indices, class_mask):
        """Build grid from pre-rendered images."""
        _cls_max = _inc_max[class_mask]
        _cls_global_i = _inc_i[class_mask]

        _imgs = []
        for _idx in indices:
            _img_i = int(_cls_global_i[_idx])
            _sae, _seg = _rendered[_img_i]
            _act_val = float(_cls_max[_idx])
            _sv = sv_labels[_img_i]
            _ssp, _view = _sv.rsplit("_", 1)
            _cells = [mo.image(_sae)]
            if show_seg_toggle.value and _seg is not None:
                _cells.append(mo.image(_seg))
            _cells.append(mo.md(f"{_ssp} ({_view}) | act={_act_val:.3f}"))
            _imgs.append(mo.vstack(_cells))

        _rows = []
        for _r in range(0, len(_imgs), _n_cols):
            _rows.append(mo.hstack(_imgs[_r : _r + _n_cols], justify="start"))
        return mo.vstack(_rows)

    mo.vstack([
        mo.md(f"## Feature {selected_feature_id} (upper={_upper:.3f})"),
        mo.md(f"### Erato Top {len(_e_top)}"),
        _build_grid(_e_top, erato_class_mask),
        mo.md(f"### Melpomene Top {len(_m_top)}"),
        _build_grid(_m_top, melp_class_mask),
        mo.md(f"### Erato Bottom {len(_e_bot)}"),
        _build_grid(_e_bot, erato_class_mask),
        mo.md(f"### Melpomene Bottom {len(_m_bot)}"),
        _build_grid(_m_bot, melp_class_mask),
    ])
    return


if __name__ == "__main__":
    app.run()
