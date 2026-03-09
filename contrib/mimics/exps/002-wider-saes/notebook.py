import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import wandb
    from adjustText import adjust_text

    import saev.colors

    figs_dpath = pathlib.Path("contrib/mimics/exps/002-wider-saes/figs")
    figs_dpath.mkdir(exist_ok=True)
    return adjust_text, figs_dpath, mo, np, pl, plt, saev, wandb


@app.cell
def _(mo):
    mo.md("""
    # 002-wider-saes: Pareto Selection

    Pick Pareto-optimal runs (NMSE vs L0) from the `mimics-32k-384p-v1.6` sweep.
    """)
    return


@app.cell
def _(mo, pl, wandb):
    tags = ["mimics-16k-384p-v1.6", "mimics-32k-384p-v1.6"]
    # Only consider runs after the W_enc normalization bugfix (2026-03-07T23:30:00Z).
    created_gte = "2026-03-07T23:30:00"
    api = wandb.Api()
    runs = list(
        api.runs(
            path="samuelstevens/saev",
            filters={
                "tags": {"$in": tags},
                "created_at": {"$gte": created_gte},
            },
        )
    )

    rows = []
    for run in mo.status.progress_bar(runs, title="Loading WandB runs"):
        cfg = run.config
        sae = cfg.get("sae", {})
        activation = sae.get("activation", {})
        train_data = cfg.get("train_data", {})

        l0 = run.summary.get("eval/l0")
        nmse = run.summary.get("eval/normalized_mse")
        if l0 is None or nmse is None:
            continue

        n_dead = run.summary.get("eval/n_dead", 0)
        d_sae = sae.get("d_sae")
        rows.append({
            "id": run.id,
            "layer": train_data.get("layer"),
            "d_sae": d_sae,
            "k": activation.get("top_k"),
            "lr": cfg.get("lr"),
            "l0": float(l0),
            "nmse": float(nmse),
            "n_dead": int(n_dead),
            "dead_pct": int(n_dead) / d_sae * 100 if d_sae else 0.0,
        })

    df = pl.DataFrame(rows)
    mo.stop(df.height == 0, mo.md(f"No runs found for tags `{tags}`."))
    return (df,)


@app.cell
def _(df, pl):
    # Mark Pareto-optimal per (layer, d_sae): sort by L0 ascending, keep cumulative-min NMSE.
    pareto_df = df.sort("l0", "nmse").with_columns(
        (pl.col("nmse") == pl.col("nmse").cum_min().over("layer", "d_sae")).alias(
            "is_pareto"
        )
    )
    pareto_df
    return (pareto_df,)


@app.cell
def _(adjust_text, figs_dpath, pareto_df, pl, plt, saev):
    _layers = sorted(pareto_df.get_column("layer").unique().to_list())
    _widths = sorted(pareto_df.get_column("d_sae").unique().to_list())
    _width_colors = {
        _widths[i]: c
        for i, c in zip(
            range(len(_widths)),
            [
                saev.colors.CYAN_RGB01,
                saev.colors.SCARLET_RGB01,
                saev.colors.SEA_RGB01,
                saev.colors.ORANGE_RGB01,
            ],
        )
    }

    _fig, _axes = plt.subplots(1, len(_layers), figsize=(6 * len(_layers), 4), dpi=300)
    if len(_layers) == 1:
        _axes = [_axes]

    for _ax, _layer in zip(_axes, _layers):
        _texts = []
        for _width in _widths:
            _color = _width_colors[_width]
            _width_df = pareto_df.filter(
                (pl.col("layer") == _layer) & (pl.col("d_sae") == _width)
            )
            _pareto = _width_df.filter(pl.col("is_pareto")).sort("l0")
            _rest = _width_df.filter(~pl.col("is_pareto"))

            if _rest.height > 0:
                _ax.scatter(
                    _rest.get_column("l0").to_numpy(),
                    _rest.get_column("nmse").to_numpy(),
                    color=_color,
                    marker="o",
                    s=12,
                    alpha=0.3,
                )

            _xs = _pareto.get_column("l0").to_numpy()
            _ys = _pareto.get_column("nmse").to_numpy()
            _label = f"{_width // 1024}K"
            _ax.plot(_xs, _ys, color=_color, marker="o", alpha=0.8, label=_label)

            for _x, _y, _rid in zip(_xs, _ys, _pareto.get_column("id").to_list()):
                _texts.append(
                    _ax.text(_x, _y, _rid, fontsize=5, ha="left", va="bottom")
                )

        _ax.set_title(f"Layer {_layer}")
        _ax.set_xlabel("L0")
        _ax.set_ylabel("NMSE")
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.grid(True, linewidth=0.3, alpha=0.7)
        _ax.legend(fontsize="small")
        _ax.spines[["right", "top"]].set_visible(False)
        adjust_text(_texts, ax=_ax)

    _fig.savefig(figs_dpath / "pareto.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(mo, pareto_df, pl):
    frontier = pareto_df.filter(pl.col("is_pareto")).sort("layer", "d_sae", "l0")
    mo.md(f"""
    ## Pareto-optimal runs

    {mo.as_html(frontier.select("id", "layer", "d_sae", "k", "lr", "l0", "nmse"))}
    """)
    return (frontier,)


@app.cell
def _(frontier):
    # Copy-pasteable run IDs per layer.
    for _layer in sorted(frontier.get_column("layer").unique().to_list()):
        ids = frontier.filter(frontier["layer"] == _layer).get_column("id").to_list()
        print(f"Layer {_layer}: {ids}")
    return


@app.cell
def _(df, figs_dpath, np, pl, plt, saev):
    # Dead latent % for the best LR (lowest NMSE) per (layer, d_sae, k).
    _dead_agg = (
        df
        .sort("nmse")
        .group_by("layer", "d_sae", "k")
        .first()
        .sort("layer", "k", "d_sae")
    )

    _layers = sorted(_dead_agg.get_column("layer").unique().to_list())
    _ks = sorted(_dead_agg.get_column("k").unique().to_list())
    _widths = sorted(_dead_agg.get_column("d_sae").unique().to_list())
    _width_colors = {
        _w: _c
        for _w, _c in zip(
            _widths,
            [saev.colors.CYAN_RGB01, saev.colors.SCARLET_RGB01],
        )
    }

    _fig, _axes = plt.subplots(
        1, len(_layers), figsize=(6 * len(_layers), 4), dpi=150, sharey=True
    )
    if len(_layers) == 1:
        _axes = [_axes]

    _bar_width = 0.35
    _x = np.arange(len(_ks))

    for _ax, _layer in zip(_axes, _layers):
        for _i, _width in enumerate(_widths):
            _sub = _dead_agg.filter(
                (pl.col("layer") == _layer) & (pl.col("d_sae") == _width)
            ).sort("k")
            _vals = _sub.get_column("dead_pct").to_numpy()
            _label = f"{_width // 1024}K"
            _ax.bar(
                _x + _i * _bar_width,
                _vals,
                _bar_width,
                label=_label,
                color=_width_colors[_width],
                alpha=0.8,
            )

        _ax.set_title(f"Layer {_layer}")
        _ax.set_xlabel("k")
        _ax.set_ylabel("Dead latents (%, best LR)")
        _ax.set_xticks(_x + _bar_width / 2)
        _ax.set_xticklabels([str(_k) for _k in _ks])
        _ax.legend(fontsize="small")
        _ax.spines[["right", "top"]].set_visible(False)

    _fig.savefig(figs_dpath / "dead_latents.png", bbox_inches="tight")
    _fig
    return


@app.cell
def _(frontier, mo, pl):
    # Check if any Pareto-optimal runs hit the LR sweep boundaries.
    _lr_min, _lr_max = 1e-4, 1e-2
    _at_min = frontier.filter(pl.col("lr") <= _lr_min)
    _at_max = frontier.filter(pl.col("lr") >= _lr_max)
    mo.md(f"""
    ## LR boundary check

    LR sweep range: [{_lr_min}, {_lr_max}]

    **Pareto runs at min LR ({_lr_min}):** {_at_min.height}
    {mo.as_html(_at_min.select("id", "layer", "d_sae", "k", "lr", "l0", "nmse")) if _at_min.height > 0 else "None"}

    **Pareto runs at max LR ({_lr_max}):** {_at_max.height}
    {mo.as_html(_at_max.select("id", "layer", "d_sae", "k", "lr", "l0", "nmse")) if _at_max.height > 0 else "None"}
    """)
    return


@app.cell
def _(frontier, mo, pathlib, pl):
    # Load pre-computed scores from parquet files written by score.py.
    _run_root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
    _shard_id = "a6be28a1"

    _dfs = []
    for _rid in frontier.get_column("id").to_list():
        _pq = _run_root / _rid / "inference" / _shard_id / "cambridge-mimics.parquet"
        if _pq.exists():
            _dfs.append(pl.read_parquet(_pq))

    scores_df = pl.concat(_dfs) if _dfs else pl.DataFrame()
    mo.stop(
        scores_df.height == 0, mo.md("No score parquets found. Run `launch.py score`.")
    )
    scores_df
    return (scores_df,)


@app.cell
def _(frontier, mo, pl, scores_df):
    # Top features by AUROC per (run, task), with run metadata.
    _top = (
        scores_df
        .with_columns(pl.col("auroc").sub(0.5).abs().alias("selectivity"))
        .sort("selectivity", descending=True)
        .group_by("run_id", "task")
        .head(5)
        .join(
            frontier.select("id", "layer", "d_sae", "k"),
            left_on="run_id",
            right_on="id",
        )
        .sort("selectivity", descending=True)
    )
    mo.md(f"""
    ## Top discriminative features

    Top 5 features per (run, task) by selectivity = |AUROC - 0.5|.

    {mo.as_html(_top.head(50))}
    """)
    return


if __name__ == "__main__":
    app.run()
