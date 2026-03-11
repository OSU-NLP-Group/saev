import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import collections
    import concurrent.futures

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import wandb

    return beartype, collections, concurrent, mo, pl, plt, wandb


@app.cell
def _():
    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "saev"
    WANDB_TAGS = ["pe-core", "auxk-comparison-v0.3"]

    # Shard hashes for identifying model from val shards path
    PE_CORE_IN1K_VAL = "a7f78fe3"
    DINOV3_IN1K_VAL = "3e27794f"
    return DINOV3_IN1K_VAL, PE_CORE_IN1K_VAL, WANDB_PROJECT, WANDB_TAGS, WANDB_USERNAME


@app.cell
def _(
    DINOV3_IN1K_VAL,
    PE_CORE_IN1K_VAL,
    WANDB_PROJECT,
    WANDB_TAGS,
    WANDB_USERNAME,
    beartype,
    concurrent,
    mo,
    pl,
    wandb,
):
    @beartype.beartype
    def _identify_model(shards_path: str) -> str | None:
        if PE_CORE_IN1K_VAL in shards_path:
            return "PE-core"
        if DINOV3_IN1K_VAL in shards_path:
            return "DINOv3"
        return None

    @beartype.beartype
    def _row_from_run(wandb_run) -> dict[str, object] | None:
        row = {"id": wandb_run.id}

        row.update(**{
            f"summary/{key}": value for key, value in wandb_run.summary.items()
        })

        config = dict(wandb_run.config)

        try:
            train_data = config.pop("train_data")
            val_data = config.pop("val_data")
        except KeyError as err:
            print(f"Run {wandb_run.id} missing config section: {err}.")
            return None

        row.update(**{
            f"config/train_data/{key}": value for key, value in train_data.items()
        })
        row.update(**{
            f"config/val_data/{key}": value for key, value in val_data.items()
        })
        row.update(**{f"config/{key}": value for key, value in config.items()})

        return row

    @beartype.beartype
    def _finalize_df(rows: list[dict[str, object]]):
        df = pl.DataFrame(rows, infer_schema_length=None)

        df = (
            df
            .unnest("config/sae", "config/train_data/metadata", separator="/")
            .unnest("config/sae/activation", separator="/")
            .unnest(
                "config/sae/activation/aux",
                "config/sae/activation/sparsity",
                separator="/",
            )
        )

        # Identify model from val shards path
        df = df.with_columns(
            pl
            .col("config/val_data/shards")
            .map_elements(_identify_model, return_dtype=pl.Utf8)
            .alias("model")
        )

        # Keep only PE-core and DINOv3 IN1K runs
        df = df.filter(pl.col("model").is_not_null())

        # Compute Pareto per model and layer
        group_cols = ("model", "config/val_data/layer")
        x_col = "summary/eval/l0"
        y_col = "summary/eval/normalized_mse"
        pareto_ids = set()
        for _keys, group_df in df.group_by(group_cols):
            group_df = group_df.filter(
                pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null()
            ).sort(x_col, y_col)

            if group_df.height == 0:
                continue

            ids = group_df.get_column("id").to_list()
            ys = group_df.get_column(y_col).to_list()

            min_y = float("inf")
            for rid, y in zip(ids, ys):
                if y < min_y:
                    pareto_ids.add(rid)
                    min_y = y

        df = df.with_columns(pl.col("id").is_in(pareto_ids).alias("is_pareto"))

        return df

    @beartype.beartype
    def _fetch_wandb_runs():
        all_runs = []
        seen_ids = set()
        for tag in WANDB_TAGS:
            runs = list(
                wandb.Api().runs(
                    path=f"{WANDB_USERNAME}/{WANDB_PROJECT}", filters={"tags": tag}
                )
            )
            for run in runs:
                if run.id not in seen_ids:
                    all_runs.append(run)
                    seen_ids.add(run.id)
        if not all_runs:
            raise ValueError("No runs found.")
        return all_runs

    @beartype.beartype
    def make_df_parallel(n_workers: int = 16):
        runs = _fetch_wandb_runs()

        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            fut_to_run_id = {pool.submit(_row_from_run, run): run.id for run in runs}
            for fut in mo.status.progress_bar(
                concurrent.futures.as_completed(fut_to_run_id),
                total=len(fut_to_run_id),
                remove_on_exit=True,
            ):
                try:
                    result = fut.result()
                except Exception as err:
                    print(f"Run {fut_to_run_id[fut]} blew up: {err}")
                    continue
                if result is None:
                    continue
                rows.append(result)

        assert rows, "No valid runs."
        return _finalize_df(rows)

    sae_df = make_df_parallel()
    return (sae_df,)


@app.cell
def _(sae_df):
    sae_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## L0 vs Normalized MSE (Pareto Curves)
    """)
    return


@app.cell
def _(collections, pl, plt, sae_df):
    def plot_pareto(df: pl.DataFrame):
        x_col = "summary/eval/l0"
        y_col = "summary/eval/normalized_mse"
        k_col = "config/sae/activation/top_k"

        layers = [21, 23]
        models = ["PE-core", "DINOv3"]
        model_colors = {"PE-core": "#1f77b4", "DINOv3": "#ff7f0e"}
        model_markers = {"PE-core": "o", "DINOv3": "s"}

        k_values = sorted(df.get_column(k_col).drop_nulls().unique().to_list())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, layout="constrained")

        pareto_ckpts = collections.defaultdict(list)

        for ax, layer in zip(axes, layers):
            for model in models:
                color = model_colors[model]
                marker = model_markers[model]

                group = df.filter(
                    (pl.col("config/sae/activation/key") == "top-k")
                    & (pl.col("config/val_data/layer") == layer)
                    & (pl.col("model") == model)
                )
                if group.height == 0:
                    continue

                group = group.sort(by=x_col)
                pareto = group.filter(pl.col("is_pareto"))

                if pareto.height > 0:
                    ids = pareto.get_column("id").to_list()
                    xs = pareto.get_column(x_col).to_numpy()
                    ys = pareto.get_column(y_col).to_numpy()

                    ax.plot(
                        xs,
                        ys,
                        alpha=0.7,
                        label=model,
                        color=color,
                        marker=marker,
                        linestyle="-",
                    )
                    pareto_ckpts[(model, layer)].extend(ids)
                    pareto_k_values = set(pareto.get_column(k_col).to_list())
                else:
                    pareto_k_values = set()

                for k in k_values:
                    if k in pareto_k_values:
                        continue
                    k_group = group.filter(pl.col(k_col) == k)
                    if k_group.height == 0:
                        continue
                    best = k_group.sort(y_col).head(1)
                    x = best.get_column(x_col).item()
                    y = best.get_column(y_col).item()
                    ax.scatter([x], [y], alpha=0.4, color=color, marker=marker, s=30)

            ax.set_xlabel("L$_0$ (average active features)")
            ax.set_ylabel("Normalized MSE")
            ax.set_title(f"Layer {layer}")
            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()

        fig.suptitle("SAE Pareto Curves: PE-core vs DINOv3 on ImageNet-1K")

        return fig, dict(pareto_ckpts)

    _fig, pareto_ckpts = plot_pareto(sae_df)
    _fig
    return (pareto_ckpts,)


@app.cell
def _(mo):
    mo.md("""
    ## Pareto Checkpoint IDs
    """)
    return


@app.cell
def _(pareto_ckpts):
    pareto_ckpts
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary Statistics
    """)
    return


@app.cell
def _(pl, sae_df):
    summary = (
        sae_df
        .group_by("model", "config/val_data/layer", "config/sae/activation/top_k")
        .agg(
            pl.col("summary/eval/l0").mean().alias("mean_l0"),
            pl.col("summary/eval/normalized_mse").mean().alias("mean_nmse"),
            pl.col("summary/eval/normalized_mse").min().alias("min_nmse"),
            pl.len().alias("n_runs"),
        )
        .sort("model", "config/val_data/layer", "config/sae/activation/top_k")
    )
    summary
    return


@app.cell
def _(mo):
    mo.md("""
    ## Pareto from Inference metrics.json

    Reads `metrics.json` from on-disk inference outputs instead of W&B summary values. Uses IN1K val shards for both models.
    """)
    return


@app.cell
def _(PE_CORE_IN1K_VAL, DINOV3_IN1K_VAL, beartype, pl, sae_df):
    import json
    import os

    RUN_ROOT = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    EVAL_SHARDS = {"PE-core": PE_CORE_IN1K_VAL, "DINOv3": DINOV3_IN1K_VAL}

    @beartype.beartype
    def _read_metrics(run_id: str, model: str) -> dict[str, float] | None:
        eval_hash = EVAL_SHARDS.get(model)
        if eval_hash is None:
            return None
        metrics_fpath = os.path.join(
            RUN_ROOT, run_id, "inference", eval_hash, "metrics.json"
        )
        if not os.path.isfile(metrics_fpath):
            return None
        with open(metrics_fpath) as fd:
            return json.load(fd)

    rows = []
    for row in sae_df.iter_rows(named=True):
        rid, model = row["id"], row["model"]
        metrics = _read_metrics(rid, model)
        if metrics is None:
            rows.append({"id": rid, "disk/normalized_mse": None})
        else:
            rows.append({"id": rid, "disk/normalized_mse": metrics["normalized_mse"]})

    disk_df = pl.DataFrame(rows, infer_schema_length=None)
    disk_sae_df = sae_df.join(disk_df, on="id", how="left")

    # Pareto on disk metrics
    pareto_ids_disk = set()
    for _keys, group in disk_sae_df.filter(
        pl.col("disk/normalized_mse").is_not_null()
    ).group_by("model", "config/val_data/layer"):
        group = group.sort("config/sae/activation/top_k", "disk/normalized_mse")
        min_y = float("inf")
        for rid, y in zip(
            group.get_column("id").to_list(),
            group.get_column("disk/normalized_mse").to_list(),
        ):
            if y < min_y:
                pareto_ids_disk.add(rid)
                min_y = y

    disk_sae_df = disk_sae_df.with_columns(
        pl.col("id").is_in(pareto_ids_disk).alias("is_pareto_disk")
    )

    disk_sae_df.filter(pl.col("disk/normalized_mse").is_not_null()).select(
        "id",
        "model",
        "config/val_data/layer",
        "config/sae/activation/top_k",
        "disk/normalized_mse",
        "is_pareto_disk",
    )
    return (disk_sae_df,)


@app.cell
def _(collections, pl, plt, disk_sae_df):
    def plot_pareto_disk(df: pl.DataFrame):
        x_col = "config/sae/activation/top_k"
        y_col = "disk/normalized_mse"

        layers = [21, 23]
        models = ["PE-core", "DINOv3"]
        model_colors = {"PE-core": "#1f77b4", "DINOv3": "#ff7f0e"}
        model_markers = {"PE-core": "o", "DINOv3": "s"}

        df = df.filter(pl.col(y_col).is_not_null())
        k_values = sorted(df.get_column(x_col).drop_nulls().unique().to_list())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, layout="constrained")
        pareto_ckpts_disk = collections.defaultdict(list)

        for ax, layer in zip(axes, layers):
            for model in models:
                color = model_colors[model]
                marker = model_markers[model]

                group = df.filter(
                    (pl.col("config/sae/activation/key") == "top-k")
                    & (pl.col("config/val_data/layer") == layer)
                    & (pl.col("model") == model)
                )
                if group.height == 0:
                    continue

                group = group.sort(by=x_col)
                pareto = group.filter(pl.col("is_pareto_disk"))

                if pareto.height > 0:
                    ids = pareto.get_column("id").to_list()
                    xs = pareto.get_column(x_col).to_numpy()
                    ys = pareto.get_column(y_col).to_numpy()

                    ax.plot(
                        xs,
                        ys,
                        alpha=0.7,
                        label=model,
                        color=color,
                        marker=marker,
                        linestyle="-",
                    )
                    pareto_ckpts_disk[(model, layer)].extend(ids)
                    pareto_k_values = set(pareto.get_column(x_col).to_list())
                else:
                    pareto_k_values = set()

                for k in k_values:
                    if k in pareto_k_values:
                        continue
                    k_group = group.filter(pl.col(x_col) == k)
                    if k_group.height == 0:
                        continue
                    best = k_group.sort(y_col).head(1)
                    ax.scatter(
                        [best.get_column(x_col).item()],
                        [best.get_column(y_col).item()],
                        alpha=0.4,
                        color=color,
                        marker=marker,
                        s=30,
                    )

            ax.set_xlabel("$k$ (top-k = L$_0$)")
            ax.set_ylabel("Normalized MSE (from metrics.json)")
            ax.set_title(f"Layer {layer}")
            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()

        fig.suptitle("SAE Pareto Curves (disk metrics): PE-core vs DINOv3 on IN1K val")
        return fig, dict(pareto_ckpts_disk)

    _fig_disk, pareto_ckpts_disk = plot_pareto_disk(disk_sae_df)
    _fig_disk
    return (pareto_ckpts_disk,)


@app.cell
def _(pareto_ckpts_disk):
    pareto_ckpts_disk
    return


@app.cell
def _(mo):
    mo.md("""
    ## W&B vs Disk NMSE

    After the W_enc/W_dec shared-storage fix, NMSE from training (W&B `eval/normalized_mse`) and NMSE from inference (`metrics.json`) should agree. Points on the y=x line means the fix worked.
    """)
    return


@app.cell
def _(pl, plt, disk_sae_df):

    def plot_wandb_vs_disk(df: pl.DataFrame):
        wandb_col = "summary/eval/normalized_mse"
        disk_col = "disk/normalized_mse"

        df = df.filter(pl.col(wandb_col).is_not_null() & pl.col(disk_col).is_not_null())

        models = ["PE-core", "DINOv3"]
        model_colors = {"PE-core": "#1f77b4", "DINOv3": "#ff7f0e"}
        model_markers = {"PE-core": "o", "DINOv3": "s"}

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150, layout="constrained")

        for model in models:
            group = df.filter(pl.col("model") == model)
            if group.height == 0:
                continue
            xs = group.get_column(wandb_col).to_numpy()
            ys = group.get_column(disk_col).to_numpy()
            ax.scatter(
                xs,
                ys,
                alpha=0.6,
                label=model,
                color=model_colors[model],
                marker=model_markers[model],
                s=30,
            )

        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("W&B eval/normalized_mse (training)")
        ax.set_ylabel("metrics.json normalized_mse (inference)")
        ax.set_title("NMSE: training vs inference (should be y=x)")
        ax.grid(True, linewidth=0.3, alpha=0.7)
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend()
        return fig

    _fig_scatter = plot_wandb_vs_disk(disk_sae_df)
    _fig_scatter
    return


if __name__ == "__main__":
    app.run()
