import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import collections
    import concurrent.futures
    import json
    import os.path
    import pathlib

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import wandb
    from jaxtyping import Float, jaxtyped

    import saev.data as saev_data

    return (
        Float,
        beartype,
        collections,
        concurrent,
        jaxtyped,
        json,
        mo,
        np,
        os,
        pathlib,
        pl,
        plt,
        saev_data,
        wandb,
    )


@app.cell
def _(pathlib):
    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "saev"
    WANDB_TAGS = ["pe-spatial", "auxk-comparison-v0.3"]

    # Shard hashes for identifying model and dataset
    PE_SPATIAL_IN1K_VAL = "5630ff85"
    DINOV3_IN1K_VAL = "3e27794f"
    PE_SPATIAL_ADE20K_TRAIN = "4b279034"
    PE_SPATIAL_ADE20K_VAL = "245e4d61"
    DINOV3_ADE20K_TRAIN = "614861a0"
    DINOV3_ADE20K_VAL = "3802cb66"

    runs_root_dpath = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
    shards_root_dpath = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")
    return (
        DINOV3_ADE20K_TRAIN,
        DINOV3_ADE20K_VAL,
        DINOV3_IN1K_VAL,
        PE_SPATIAL_ADE20K_TRAIN,
        PE_SPATIAL_ADE20K_VAL,
        PE_SPATIAL_IN1K_VAL,
        WANDB_PROJECT,
        WANDB_TAGS,
        WANDB_USERNAME,
        runs_root_dpath,
        shards_root_dpath,
    )


@app.cell
def _(
    DINOV3_IN1K_VAL,
    PE_SPATIAL_IN1K_VAL,
    WANDB_PROJECT,
    WANDB_TAGS,
    WANDB_USERNAME,
    beartype,
    concurrent,
    load_freqs,
    load_mean_values,
    mo,
    pl,
    wandb,
):
    @beartype.beartype
    def _row_from_run(wandb_run) -> dict[str, object] | None:
        row = {"id": wandb_run.id}

        row.update(**{
            f"summary/{key}": value for key, value in wandb_run.summary.items()
        })

        try:
            row["summary/eval/freqs"] = load_freqs(wandb_run)
        except Exception as err:
            print(f"Run {wandb_run.id} failed loading freqs: {err}")
            return None

        try:
            row["summary/eval/mean_values"] = load_mean_values(wandb_run)
        except Exception as err:
            print(f"Run {wandb_run.id} failed loading mean values: {err}")
            return None

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
    def _identify_model(shards_path: str) -> str | None:
        """Identify model from validation shards path."""
        if PE_SPATIAL_IN1K_VAL in shards_path:
            return "PE-spatial"
        elif DINOV3_IN1K_VAL in shards_path:
            return "DINOv3"
        return None

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

        # Add model column based on val shards
        df = df.with_columns(
            pl
            .col("config/val_data/shards")
            .map_elements(_identify_model, return_dtype=pl.Utf8)
            .alias("model")
        )

        # Filter to only ImageNet-1K runs (PE-spatial or DINOv3)
        df = df.filter(pl.col("model").is_not_null())

        # Compute Pareto per model and layer
        group_cols = ("model", "config/val_data/layer", "config/sae/activation/key")
        x_col = "summary/eval/l0"
        y_col = "summary/eval/normalized_mse"
        pareto_ids = set()
        for keys, group_df in df.group_by(group_cols):
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
        models = ["PE-spatial", "DINOv3"]
        k_values = sorted(df.get_column(k_col).unique().to_list())

        # Colors and markers for each model
        model_colors = {"PE-spatial": "#1f77b4", "DINOv3": "#ff7f0e"}
        model_markers = {"PE-spatial": "o", "DINOv3": "s"}

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

                # Plot pareto front line
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

                # For k values without pareto points, plot best as faded point
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

        fig.suptitle("SAE Pareto Curves: PE-spatial vs DINOv3 ViT-L/14 on ImageNet-1K")

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
    ## Probe1D results (ADE20K)
    """)
    return


@app.cell
def _(
    DINOV3_ADE20K_TRAIN,
    DINOV3_ADE20K_VAL,
    PE_SPATIAL_ADE20K_TRAIN,
    PE_SPATIAL_ADE20K_VAL,
    beartype,
    json,
    np,
    pl,
    runs_root_dpath,
    sae_df,
    saev_data,
    shards_root_dpath,
):
    models = ["PE-spatial", "DINOv3"]
    ade20k_shards_by_model = {
        "PE-spatial": (PE_SPATIAL_ADE20K_TRAIN, PE_SPATIAL_ADE20K_VAL),
        "DINOv3": (DINOV3_ADE20K_TRAIN, DINOV3_ADE20K_VAL),
    }

    @beartype.beartype
    def load_probe_loss(run_id: str, shards_hash: str) -> np.ndarray | None:
        metrics_fpath = (
            runs_root_dpath / run_id / "inference" / shards_hash / "probe1d_metrics.npz"
        )
        if not metrics_fpath.exists():
            return None

        with np.load(metrics_fpath) as fd:
            loss = fd["loss"]

        assert loss.ndim == 2
        return loss

    @beartype.beartype
    def load_nmse(run_id: str, shards_hash: str) -> float | None:
        metrics_fpath = (
            runs_root_dpath / run_id / "inference" / shards_hash / "metrics.json"
        )
        if not metrics_fpath.exists():
            return None
        metrics = json.loads(metrics_fpath.read_text())
        nmse = metrics.get("normalized_mse")
        if nmse is None:
            return None
        return float(nmse)

    @beartype.beartype
    def make_wandb_nmse_map(keys: list[str]) -> dict[str, float | None]:
        for key in keys:
            if key not in sae_df.columns:
                continue
            sub = sae_df.select("id", key)
            ids = sub.get_column("id").to_list()
            values = sub.get_column(key).to_list()
            return {
                run_id: (float(val) if val is not None else None)
                for run_id, val in zip(ids, values)
            }
        return {}

    @beartype.beartype
    def get_baseline_ce_mean(shards_hash: str) -> float:
        shards_dpath = shards_root_dpath / shards_hash
        md = saev_data.Metadata.load(shards_dpath)
        labels = np.memmap(
            shards_dpath / "labels.bin",
            mode="r",
            dtype=np.uint8,
            shape=(md.n_examples, md.content_tokens_per_example),
        ).reshape(-1)
        n_samples = labels.size
        n_classes = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
        prob = counts / n_samples
        prob = np.clip(prob, 1e-12, 1 - 1e-12)
        ce = -(prob * np.log(prob) + (1 - prob) * np.log(1 - prob))
        return float(ce.mean())

    @beartype.beartype
    def get_mean_purity(
        top_labels_dk: np.ndarray, best_i: np.ndarray, *, k: int
    ) -> float:
        assert top_labels_dk.ndim == 2
        labels_ck = top_labels_dk[best_i, :k]
        purity_c: list[float] = []
        for row in labels_ck:
            _, counts = np.unique(row, return_counts=True)
            purity_c.append(float(counts.max()) / k)
        return float(np.mean(purity_c))

    in1k_train_nmse_map = make_wandb_nmse_map([
        "summary/train/normalized_mse",
        "summary/metrics/normalized_mse",
    ])
    in1k_val_nmse_map = make_wandb_nmse_map(["summary/eval/normalized_mse"])
    run_meta = (
        sae_df
        .filter(
            pl.col("model").is_in(models)
            & (pl.col("config/sae/activation/aux/key") == "auxk")
            & pl.col("config/val_data/layer").is_in([21, 23])
            & pl.col("config/sae/activation/top_k").is_not_null()
        )
        .select(
            pl.col("id").alias("run_id"),
            pl.col("model"),
            pl.col("config/val_data/layer").cast(pl.Int64).alias("layer"),
            pl.col("config/sae/activation/top_k").cast(pl.Int64).alias("top_k"),
        )
        .unique()
        .sort("model", "layer", "top_k", "run_id")
    )
    baseline_ce_cache: dict[str, float] = {}

    rows: list[dict[str, object]] = []
    for run_id, model, layer, top_k in run_meta.iter_rows():
        train_shards, val_shards = ade20k_shards_by_model[model]
        train_loss = load_probe_loss(run_id, train_shards)
        val_loss = load_probe_loss(run_id, val_shards)

        row = {
            "run_id": run_id,
            "layer": layer,
            "top_k": top_k,
            "model": model,
            "train_shards": train_shards,
            "val_shards": val_shards,
            "has_train": train_loss is not None,
            "has_val": val_loss is not None,
            "in1k_train_nmse": in1k_train_nmse_map.get(run_id),
            "in1k_val_nmse": in1k_val_nmse_map.get(run_id),
            "ade20k_train_nmse": load_nmse(run_id, train_shards),
            "ade20k_val_nmse": load_nmse(run_id, val_shards),
            "train_probe_r": None,
            "val_probe_r": None,
            "val_mean_ap": None,
            "val_mean_precision": None,
            "val_mean_recall": None,
            "val_mean_f1": None,
            "val_coverage_at_0_3": None,
            "val_coverage_at_0_5": None,
            "val_coverage_at_0_7": None,
            "val_purity_at_16": None,
            "val_purity_at_64": None,
            "val_purity_at_256": None,
        }

        if train_loss is None or val_loss is None:
            row["status"] = "missing"
            rows.append(row)
            continue

        assert train_loss.shape == val_loss.shape
        n_latents, n_classes = train_loss.shape
        best_i = np.argmin(train_loss, axis=0)
        train_ce = train_loss[best_i, np.arange(n_classes)]
        val_ce = val_loss[best_i, np.arange(n_classes)]

        if train_shards not in baseline_ce_cache:
            baseline_ce_cache[train_shards] = get_baseline_ce_mean(train_shards)
        if val_shards not in baseline_ce_cache:
            baseline_ce_cache[val_shards] = get_baseline_ce_mean(val_shards)

        train_base_ce = baseline_ce_cache[train_shards]
        val_base_ce = baseline_ce_cache[val_shards]

        row.update({
            "status": "ok",
            "n_latents": int(n_latents),
            "n_classes": int(n_classes),
            "train_ce_mean": float(train_ce.mean()),
            "val_ce_mean": float(val_ce.mean()),
            "val_ce_median": float(np.median(val_ce)),
            "val_ce_min": float(val_ce.min()),
            "train_probe_r": float(1 - train_ce.mean() / train_base_ce),
            "val_probe_r": float(1 - val_ce.mean() / val_base_ce),
        })

        metrics_fpath = (
            runs_root_dpath
            / run_id
            / "inference"
            / val_shards
            / f"probe1d_metrics__train-{train_shards}.npz"
        )
        if metrics_fpath.exists():
            with np.load(metrics_fpath) as fd:
                ap_c = fd["ap"]
                prec_c = fd["precision"]
                recall_c = fd["recall"]
                f1_c = fd["f1"]
                top_labels_dk = fd["top_labels"]

            row.update({
                "val_mean_ap": float(ap_c.mean()),
                "val_mean_precision": float(prec_c.mean()),
                "val_mean_recall": float(recall_c.mean()),
                "val_mean_f1": float(f1_c.mean()),
            })

            for tau in [0.3, 0.5, 0.7]:
                row[f"val_coverage_at_{str(tau).replace('.', '_')}"] = float(
                    (ap_c > tau).mean()
                )

            for k in [16, 64, 256]:
                row[f"val_purity_at_{k}"] = get_mean_purity(top_labels_dk, best_i, k=k)
        rows.append(row)

    probe_df = pl.DataFrame(rows, infer_schema_length=None).sort(
        "status", "layer", "top_k"
    )
    probe_df
    return (probe_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Val probe r by layer and k
    """)
    return


@app.cell
def _(pl, plt, probe_df):
    def _():
        df = probe_df.filter(
            (pl.col("status") == "ok") & pl.col("val_probe_r").is_not_null()
        )
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(8, 3.5),
            dpi=200,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        if df.height == 0:
            for ax in axes:
                ax.text(0.5, 0.5, "No probe metrics yet", ha="center", va="center")
                ax.set_axis_off()
            return fig

        layer_values = sorted(df.get_column("layer").unique().to_list())
        model_values = ["PE-spatial", "DINOv3"]
        model_values = [m for m in model_values if m in df.get_column("model")]
        if not model_values:
            model_values = sorted(df.get_column("model").unique().to_list())

        for ax, model in zip(axes, model_values):
            sub = df.filter(pl.col("model") == model)
            if sub.height == 0:
                ax.text(0.5, 0.5, f"No {model} probe metrics", ha="center", va="center")
                ax.set_axis_off()
                continue

            k_values = sorted(sub.get_column("top_k").unique().to_list())
            for k in k_values:
                sub_k = sub.filter(pl.col("top_k") == k).sort("layer")
                xs = sub_k.get_column("layer").to_list()
                ys = sub_k.get_column("val_probe_r").to_list()
                ax.plot(xs, ys, marker="o", label=f"k={k}")

            ax.set_title(model)
            ax.set_xlabel("Layer")
            ax.set_xticks(layer_values)
            ax.grid(True, linewidth=0.3, alpha=0.6)
            ax.spines[["right", "top"]].set_visible(False)
            ax.legend(title="top_k")

        axes[0].set_ylabel("Val probe r")
        return fig

    _()
    return


@app.cell
def _(Float, beartype, jaxtyped, json, np, os):
    @jaxtyped(typechecker=beartype.beartype)
    def load_freqs(run) -> Float[np.ndarray, " d_sae"]:
        try:
            for artifact in run.logged_artifacts():
                if "evalfreqs" not in artifact.name:
                    continue

                dpath = artifact.download()
                fpath = os.path.join(dpath, "eval", "freqs.table.json")
                with open(fpath) as fd:
                    raw = json.load(fd)
                return np.array(raw["data"], dtype=float).reshape(-1)
        except Exception as err:
            raise RuntimeError(f"Wandb sucks: {err}") from err

        raise ValueError(f"freqs not found in run '{run.id}'")

    @jaxtyped(typechecker=beartype.beartype)
    def load_mean_values(run) -> Float[np.ndarray, " d_sae"]:
        try:
            for artifact in run.logged_artifacts():
                if "evalmean_values" not in artifact.name:
                    continue

                dpath = artifact.download()
                fpath = os.path.join(dpath, "eval", "mean_values.table.json")
                with open(fpath) as fd:
                    raw = json.load(fd)
                return np.array(raw["data"], dtype=float).reshape(-1)
        except Exception as err:
            raise RuntimeError(f"Wandb sucks: {err}") from err

        raise ValueError(f"mean_values not found in run '{run.id}'")

    return load_freqs, load_mean_values


if __name__ == "__main__":
    app.run()
