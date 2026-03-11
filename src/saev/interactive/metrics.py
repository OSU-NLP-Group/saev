import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import collections
    import itertools
    import json
    import os
    import pickle

    import altair as alt
    import beartype
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import wandb
    from adjustText import adjust_text
    from jaxtyping import Float, jaxtyped

    import saev.colors

    return (
        Float,
        adjust_text,
        alt,
        base64,
        beartype,
        collections,
        itertools,
        jaxtyped,
        json,
        mo,
        mpl,
        np,
        os,
        pickle,
        pl,
        plt,
        saev,
        wandb,
    )


@app.cell
def _(mo):
    mo.md("""
    # SAE Metrics Explorer

    This notebook helps you analyze and compare SAE training runs from WandB.

    1. Edit the configuration cell at the top to set your WandB username and project
    2. Use the filters to narrow down which models to compare

    ## Troubleshooting

    - **No runs found**: Check your WandB username, project name, and tag filter
    """)
    return


@app.cell
def _():
    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "saev"
    return WANDB_PROJECT, WANDB_USERNAME


@app.cell
def _(mo):
    tag_input = mo.ui.text(label="Sweep Tag:")
    return (tag_input,)


@app.cell
def _(WANDB_PROJECT, WANDB_USERNAME, mo, tag_input):
    mo.vstack([
        mo.md(
            f"Look at [{WANDB_USERNAME}/{WANDB_PROJECT} on WandB](https://wandb.ai/{WANDB_USERNAME}/{WANDB_PROJECT}/table) to pick your tag."
        ),
        tag_input,
    ])
    return


@app.cell
def _(df, mo):
    # All unique (model, layer) pairs
    pairs = (
        df
        .select(["model_key", "config/val_data/layer"])
        .unique()
        .sort(by=["model_key", "config/val_data/layer"])
        .iter_rows()
    )

    pair_elems = {
        f"{model}|{layer}": mo.ui.switch(value=False, label=f"{model} / layer {layer}")
        for model, layer in pairs
    }
    pair_dict = mo.ui.dictionary(pair_elems)  # ★ reactive wrapper ★

    # Global toggle for non-frontier ("rest") points
    show_rest = mo.ui.switch(value=True, label="Show non-frontier points")
    show_ids = mo.ui.switch(value=True, label="Annotate Pareto points")

    def _make_grid(elems, ncols: int, gap: float):
        return mo.hstack(
            [
                mo.vstack(elems[i : i + ncols], gap=gap, justify="start")
                for i in range(0, len(elems), ncols)
            ],
            gap=gap,
        )

    elems = [*pair_dict.elements.values(), show_rest, show_ids]
    ui_grid = _make_grid(elems, 3, 0.5)
    ui_grid
    return pair_dict, show_ids, show_rest


@app.cell
def _(
    adjust_text,
    collections,
    df,
    itertools,
    mo,
    pair_dict,
    pl,
    plt,
    saev,
    show_ids,
    show_rest,
):
    def plot_layerwise(
        df: pl.DataFrame,
        show_rest: bool,
        show_ids: bool,
        l0_col: str = "summary/eval/l0",
        mse_col: str = "summary/eval/mse",
        layer_col: str = "config/val_data/layer",
        model_col: str = "model_key",
    ):
        """
        Plot Pareto frontiers (L0 vs MSE) for every (layer, model) pair using **polars** only.
        """

        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

        linestyles = ["-", "--", ":", "-."]
        colors = [
            saev.colors.CYAN_RGB01,
            saev.colors.SEA_RGB01,
            saev.colors.CREAM_RGB01,
            saev.colors.GOLD_RGB01,
            saev.colors.ORANGE_RGB01,
            saev.colors.RUST_RGB01,
            saev.colors.SCARLET_RGB01,
            saev.colors.RED_RGB01,
        ]
        markers = ["X", "o", "+"]

        models = sorted(df.select(model_col).unique().get_column(model_col).to_list())

        texts = []

        pareto_ckpts = collections.defaultdict(list)

        for model, marker in zip(models, itertools.cycle(markers)):
            model_df = df.filter((pl.col(model_col) == model))

            layers = model_df.select(layer_col).unique().get_column(layer_col).to_list()
            for layer, color, linestyle in zip(
                sorted(layers),
                itertools.cycle(colors),
                itertools.cycle(linestyles),
            ):
                group = model_df.filter(
                    (pl.col(layer_col) == layer)
                    & (pl.col("config/objective/n_prefixes").is_not_null())
                ).sort(l0_col)

                pareto = group.filter(pl.col("is_pareto"))
                if pareto.height == 0:
                    continue

                ids = pareto.get_column("id").to_list()
                xs = pareto.get_column(l0_col).to_numpy()
                ys = pareto.get_column(mse_col).to_numpy()

                line, *_ = ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    alpha=0.8,
                    label=f"{model} / layer {layer}",
                )

                edge_mask = (
                    pl.col("is_lr_min") | pl.col("is_lr_max")
                    # | pl.col("is_lambda_min")
                    # | pl.col("is_lambda_max")
                )
                edge_df = pareto.filter(edge_mask)

                if edge_df.height > 0:
                    edge_xs = edge_df.get_column(l0_col).to_numpy()
                    edge_ys = edge_df.get_column(mse_col).to_numpy()
                    ax.scatter(
                        edge_xs,
                        edge_ys,
                        facecolors="none",
                        edgecolors="tab:red",
                        marker=marker,
                        s=60,
                        linewidths=1.2,
                        zorder=line.get_zorder() + 1,
                    )

                if show_ids:
                    lr_min = pareto.get_column("is_lr_min").to_list()
                    lr_max = pareto.get_column("is_lr_max").to_list()
                    lam_min = [False] * len(lr_min)
                    lam_max = [False] * len(lr_max)
                    # lam_min = pareto.get_column("is_lambda_min").to_list()
                    # lam_max = pareto.get_column("is_lambda_max").to_list()

                    for x, y, rid, is_lr_min, is_lr_max, is_lam_min, is_lam_max in zip(
                        xs, ys, ids, lr_min, lr_max, lam_min, lam_max
                    ):
                        edge_parts = []
                        if is_lr_min:
                            edge_parts.append("LR min")
                        if is_lr_max:
                            edge_parts.append("LR max")
                        if is_lam_min:
                            edge_parts.append("$\\lambda$ min")
                        if is_lam_max:
                            edge_parts.append("$\\lambda$ max")

                        label = (
                            rid
                            if not edge_parts
                            else f"{rid} ({', '.join(edge_parts)})"
                        )
                        color_text = "tab:red" if edge_parts else "black"
                        texts.append(
                            ax.text(
                                x,
                                y,
                                label,
                                fontsize=6,
                                color=color_text,
                                ha="left",
                                va="bottom",
                            )
                        )

                        pareto_ckpts[f"{model}/{layer}"].append(rid)

                if show_rest:
                    rest = group.filter(~pl.col("is_pareto"))
                    if rest.height > 0:
                        ax.scatter(
                            rest.get_column(l0_col).to_numpy(),
                            rest.get_column(mse_col).to_numpy(),
                            color=color,
                            marker=marker,
                            s=12,
                            linewidths=0,
                            alpha=0.5,
                        )

        ax.set_xlabel("L0 sparsity (lower is better)")
        ax.set_ylabel("Reconstruction MSE (lower is better)")
        ax.grid(True, linewidth=0.3, alpha=0.7)
        ax.legend(fontsize="small", ncols=2)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xscale("log")
        ax.set_xlim(1e-1, 1e5)
        ax.set_yscale("log")

        adjust_text(texts)

        return fig, pareto_ckpts

    selected_keys = [k for k, v in pair_dict.value.items() if v]
    if selected_keys:
        models, layers = zip(*[k.rsplit("|", 1) for k in selected_keys])
        pairs_df = pl.DataFrame({
            "model_key": list(models),
            "config/val_data/layer": list(map(int, layers)),
        })
        filtered_df = df.join(
            pairs_df, on=["model_key", "config/val_data/layer"], how="inner"
        )
    else:
        filtered_df = pl.DataFrame(schema=df.schema)

    mo.stop(filtered_df.height == 0, mo.md("No runs match the current filters."))

    mo.vstack(plot_layerwise(filtered_df, show_rest.value, show_ids.value))
    return filtered_df, layers, models


@app.cell
def _(alt, df, layers, mo, pl):
    chart = mo.ui.altair_chart(
        alt
        .Chart(
            df.filter(
                pl.col("config/val_data/layer").is_in([int(layer) for layer in layers])
            ).select(
                "summary/eval/l0",
                "summary/eval/mse",
                "id",
                "config/objective/sparsity_coeff",
                pl.col("config/lr").log10().alias("config/lr"),
                "config/sae/d_sae",
                "config/val_data/layer",
                "model_key",
            )
        )
        .mark_point()
        .encode(
            x=alt.X("summary/eval/l0"),
            y=alt.Y("summary/eval/mse"),
            tooltip=["id", "config/lr"],
            color="config/lr:Q",
            shape="config/val_data/layer:N",
            # shape="config/objective/sparsity_coeff:N",
            # shape="config/sae/d_sae:N",
            # shape="model_key",
        )
    )
    chart
    return (chart,)


@app.cell
def _(
    WANDB_PROJECT,
    WANDB_USERNAME,
    chart,
    df,
    load_freqs,
    load_mean_values,
    mo,
    np,
    pl,
    plot_dist,
    plt,
    wandb,
):
    mo.stop(
        len(chart.value) < 2,
        mo.md(
            "Select two or more points. Exactly one point is not supported because of a [Polars bug](https://github.com/pola-rs/polars/issues/19855)."
        ),
    )

    selected_df = (
        df
        .select("id", "summary/eval/l0")
        .join(chart.value.select("id"), on="id", how="inner")
        .sort(by="summary/eval/l0")
        .head(4)
    )
    wandb_path = f"{WANDB_USERNAME}/{WANDB_PROJECT}"
    rows = []
    for run_id, l0 in selected_df.iter_rows():
        run = wandb.Api().run(f"{wandb_path}/{run_id}")
        try:
            freqs = load_freqs(run)
            mean_values = load_mean_values(run)
        except ValueError:
            print(f"Run {run_id} did not log eval/freqs or eval/mean_values.")
            continue
        except RuntimeError:
            print(f"Wandb blew up on run {run_id}.")
            continue
        rows.append({
            "id": run_id,
            "summary/eval/freqs": freqs,
            "summary/eval/mean_values": mean_values,
            "summary/eval/l0": l0,
        })
    mo.stop(
        not rows,
        mo.md(
            "No selected runs have eval/freqs and eval/mean_values artifacts available."
        ),
    )
    sub_df = pl.DataFrame(rows).sort(by="summary/eval/l0")

    scatter_fig, scatter_axes = plt.subplots(
        ncols=len(sub_df), figsize=(12, 3), squeeze=False, sharey=True, sharex=True
    )

    hist_fig, hist_axes = plt.subplots(
        ncols=len(sub_df),
        nrows=2,
        figsize=(12, 6),
        squeeze=False,
        sharey=True,
        sharex=True,
    )

    # Always one row
    scatter_axes = scatter_axes.reshape(-1)
    hist_axes = hist_axes.T

    for (id, freqs, values, _), scatter_ax, (freq_hist_ax, values_hist_ax) in zip(
        sub_df.iter_rows(), scatter_axes, hist_axes
    ):
        plot_dist(
            freqs.astype(float),
            (-1.0, 1.0),
            values.astype(float),
            (-2.0, 2.0),
            scatter_ax,
        )
        # ax.scatter(freqs, values, marker=".", alpha=0.03)
        # ax.set_yscale("log")
        # ax.set_xscale("log")
        scatter_ax.set_title(id)

        # Plot feature
        bins = np.linspace(-6, 1, 100)
        freq_hist_ax.hist(np.log10(freqs.astype(float)), bins=bins)
        freq_hist_ax.set_title(f"{id} Feat. Freq. Dist.")

        values_hist_ax.hist(np.log10(values.astype(float)), bins=bins)
        values_hist_ax.set_title(f"{id} Mean Val. Distribution")

    scatter_fig.tight_layout()
    hist_fig.tight_layout()
    return hist_fig, scatter_fig


@app.cell
def _(scatter_fig):
    scatter_fig
    return


@app.cell
def _(hist_fig):
    hist_fig
    return


@app.cell
def _(chart, df, pl):
    df.join(chart.value.select("id"), on="id", how="inner").sort(
        by="summary/eval/l0"
    ).select("id", pl.selectors.starts_with("config/"))
    return


@app.cell
def _(Float, beartype, jaxtyped, np):
    @jaxtyped(typechecker=beartype.beartype)
    def plot_dist(
        freqs: Float[np.ndarray, " d_sae"],
        freqs_log_range: tuple[float, float],
        values: Float[np.ndarray, " d_sae"],
        values_log_range: tuple[float, float],
        ax,
    ):
        log_sparsity = np.log10(freqs + 1e-9)
        log_values = np.log10(values + 1e-9)

        mask = np.ones(len(log_sparsity)).astype(bool)
        min_log_freq, max_log_freq = freqs_log_range
        mask[log_sparsity < min_log_freq] = False
        mask[log_sparsity > max_log_freq] = False
        min_log_value, max_log_value = values_log_range
        mask[log_values < min_log_value] = False
        mask[log_values > max_log_value] = False

        n_shown = mask.sum()
        ax.scatter(
            log_sparsity[mask],
            log_values[mask],
            marker=".",
            alpha=0.1,
            color="tab:blue",
            label=f"Shown ({n_shown})",
        )
        n_filtered = (~mask).sum()
        ax.scatter(
            log_sparsity[~mask],
            log_values[~mask],
            marker=".",
            alpha=0.1,
            color="tab:red",
            label=f"Filtered ({n_filtered})",
        )

        ax.axvline(min_log_freq, linewidth=0.5, color="tab:red")
        ax.axvline(max_log_freq, linewidth=0.5, color="tab:red")
        ax.axhline(min_log_value, linewidth=0.5, color="tab:red")
        ax.axhline(max_log_value, linewidth=0.5, color="tab:red")

        ax.set_xlabel("Feature Frequency (log10)")
        ax.set_ylabel("Mean Activation Value (log10)")

    return (plot_dist,)


@app.cell
def _(
    WANDB_PROJECT,
    WANDB_USERNAME,
    beartype,
    get_data_key,
    get_model_key,
    json,
    mo,
    os,
    pl,
    tag_input,
    wandb,
):
    class MetadataAccessError(Exception):
        """Exception raised when metadata cannot be accessed or parsed."""

        pass

    @beartype.beartype
    def find_metadata(shard_root: str, wandb_metadata: dict | None = None):
        # First check if metadata is available from WandB config
        if wandb_metadata is not None:
            return wandb_metadata

        # Fall back to reading from file system for backward compatibility
        if not os.path.exists(shard_root):
            msg = f"""
    ERROR: Shard root '{shard_root}' not found. You need to either:

    1. Run this notebook on the same machine where the shards are located.
    2. Copy the shards to this machine at path: {shard_root}
    3. Update the filtering criteria to only show checkpoints with available data""".strip()
            raise MetadataAccessError(msg)

        metadata_path = os.path.join(shard_root, "metadata.json")
        if not os.path.exists(metadata_path):
            raise MetadataAccessError("Missing metadata.json file")

        try:
            with open(metadata_path) as fd:
                return json.load(fd)
        except json.JSONDecodeError:
            raise MetadataAccessError("Malformed metadata.json file")

    @beartype.beartype
    def make_df(tag: str):
        empty_df = pl.DataFrame({
            "id": [],
            "model_key": [],
            "config/val_data/layer": [],
            "data_key": [],
            "summary/eval/l0": [],
            "summary/eval/mse": [],
        })

        if not tag:
            return empty_df

        def is_scalar(value: object) -> bool:
            if value is None:
                return True
            if isinstance(value, bool):
                return True
            if isinstance(value, str):
                return True
            if isinstance(value, int | float):
                return True
            return False

        path = f"{WANDB_USERNAME}/{WANDB_PROJECT}"
        # Most sweeps use config.tag and this is usually much narrower.
        runs = list(wandb.Api().runs(path=path, filters={"config.tag": tag}))
        if not runs:
            runs = list(wandb.Api().runs(path=path, filters={"tags": {"$in": [tag]}}))

        # runs = runs[:10]

        rows = []
        for run in mo.status.progress_bar(
            runs,
            remove_on_exit=True,
            title="Loading",
            subtitle="Parsing runs from WandB",
        ):
            row = {}
            row["id"] = run.id

            try:
                for key, value in run.summary.items():
                    if not is_scalar(value):
                        continue
                    row[f"summary/{key}"] = value
            except AttributeError as err:
                print(f"Run {run.id} has a problem in run.summary._json_dict: {err}")
                continue
            # config
            cfg = dict(run.config)

            for cfg_key in ("train_data", "val_data", "sae"):
                nested_cfg = cfg.pop(cfg_key, {})
                if not isinstance(nested_cfg, dict):
                    continue
                for key, value in nested_cfg.items():
                    if not is_scalar(value):
                        continue
                    row[f"config/{cfg_key}/{key}"] = value

            objective_cfg = cfg.pop("objective", {})
            if isinstance(objective_cfg, dict):
                for key, value in objective_cfg.items():
                    if not is_scalar(value):
                        continue
                    row[f"config/objective/{key}"] = value

            for key, value in cfg.items():
                if not is_scalar(value):
                    continue
                row[f"config/{key}"] = value

            try:
                # Check if metadata is available in WandB config
                wandb_metadata = row.get("config/train_data/metadata")
                metadata = find_metadata(
                    row["config/train_data/shards"], wandb_metadata
                )
            except MetadataAccessError as err:
                print(f"Bad run {run.id}: {err}")
                continue

            row["model_key"] = get_model_key(metadata)

            data_key = get_data_key(metadata)
            if data_key is None:
                print(f"Bad run {run.id}: unknown data.")
                continue
            row["data_key"] = data_key

            row["config/d_model"] = metadata["d_model"]
            rows.append(row)

        if not rows:
            return empty_df

        df = pl.DataFrame(rows).with_row_index("_row_idx")

        group_cols = ("model_key", "config/val_data/layer", "data_key")
        lr_col = "config/lr"
        # lambda_col = "config/objective/sparsity_coeff"

        bounds_df = df.group_by(group_cols, maintain_order=False).agg(
            pl.col(lr_col).min().alias("lr_min"),
            pl.col(lr_col).max().alias("lr_max"),
            # pl.col(lambda_col).min().alias("lambda_min"),
            # pl.col(lambda_col).max().alias("lambda_max"),
        )

        df = (
            df
            .join(bounds_df, on=group_cols, how="left")
            .with_columns(
                (pl.col(lr_col) == pl.col("lr_min")).alias("is_lr_min"),
                (pl.col(lr_col) == pl.col("lr_max")).alias("is_lr_max"),
                # (pl.col(lambda_col) == pl.col("lambda_min")).alias("is_lambda_min"),
                # (pl.col(lambda_col) == pl.col("lambda_max")).alias("is_lambda_max"),
            )
            .sort(group_cols + ("summary/eval/l0", "summary/eval/mse"))
            .with_columns(
                (
                    pl.col("summary/eval/mse")
                    == pl.col("summary/eval/mse").cum_min().over(group_cols)
                ).alias("is_pareto"),
            )
            .sort("_row_idx")
            .drop("_row_idx")
        )

        return df

    df = make_df(tag_input.value)
    mo.stop(
        df.height == 0,
        mo.md(f"No runs found for tag: `{tag_input.value}`."),
    )
    return (df,)


@app.cell
def _(base64, beartype, pickle, saev):
    @beartype.beartype
    def get_model_key(metadata: dict[str, object]) -> str:
        family = next(
            metadata[key]
            for key in ("vit_family", "model_family", "family")
            if key in metadata
        )

        ckpt = next(
            metadata[key]
            for key in ("vit_ckpt", "model_ckpt", "ckpt")
            if key in metadata
        )

        if family == "dinov2" and ckpt == "dinov2_vitb14_reg":
            return "DINOv2 ViT-B/14 (reg)"
        if family == "dinov2" and ckpt == "dinov2_vitl14_reg":
            return "DINOv2 ViT-L/14 (reg)"
        if family == "dinov3" and "vitl" in ckpt:
            return "DINOv3 ViT-L/16"
        if family == "dinov3" and "vitb" in ckpt:
            return "DINOv3 ViT-B/16"
        if family == "dinov3" and "vits" in ckpt:
            return "DINOv3 ViT-S/16"
        if family == "clip" and ckpt == "ViT-B-16/openai":
            return "CLIP ViT-B/16"
        if family == "clip" and ckpt == "hf-hub:imageomics/bioclip":
            return "BioCLIP ViT-B/16"
        if family == "clip" and ckpt == "hf-hub:imageomics/bioclip-2":
            return "BioCLIP 2 ViT-L/14"
        if family == "siglip" and ckpt == "hf-hub:timm/ViT-L-16-SigLIP2-256":
            return "SigLIP2 ViT-L/16"

        print(f"Unknown model: {(family, ckpt)}")
        return ckpt

    @beartype.beartype
    def get_data_key(metadata: dict[str, object]) -> str | None:
        data_cfg = pickle.loads(base64.b64decode(metadata["data"].encode("utf8")))

        if isinstance(data_cfg, saev.data.datasets.ImgSegFolder) and "ADE" in str(
            data_cfg.root
        ):
            return f"ADE20K/{data_cfg.split}"

        if isinstance(data_cfg, saev.data.datasets.Imagenet):
            return f"IN1K/{data_cfg.split}"

        if isinstance(
            data_cfg, saev.data.datasets.ImgFolder
        ) and "fish-vista-imgfolder" in str(data_cfg.root):
            return "FishVista (Img)"

        print(f"Unknown data: {data_cfg}")
        return None

    return get_data_key, get_model_key


@app.cell
def _(Float, json, np, os):
    def load_freqs(run) -> Float[np.ndarray, " d_sae"]:
        try:
            for artifact in run.logged_artifacts():
                if "evalfreqs" not in artifact.name:
                    continue

                dpath = artifact.download()
                fpath = os.path.join(dpath, "eval", "freqs.table.json")
                print(fpath)
                with open(fpath) as fd:
                    raw = json.load(fd)
                return np.array(raw["data"]).reshape(-1)
        except Exception as err:
            raise RuntimeError("Wandb sucks.") from err

        raise ValueError(f"freqs not found in run '{run.id}'")

    def load_mean_values(run) -> Float[np.ndarray, " d_sae"]:
        try:
            for artifact in run.logged_artifacts():
                if "evalmean_values" not in artifact.name:
                    continue

                dpath = artifact.download()
                fpath = os.path.join(dpath, "eval", "mean_values.table.json")
                print(fpath)
                with open(fpath) as fd:
                    raw = json.load(fd)
                return np.array(raw["data"]).reshape(-1)
        except Exception as err:
            raise RuntimeError("Wandb sucks.") from err

        raise ValueError(f"mean_values not found in run '{run.id}'")

    return load_freqs, load_mean_values


@app.cell
def _(alt, df, layers, mo, pl):
    mo.ui.altair_chart(
        alt
        .Chart(
            df.filter(
                pl.col("config/val_data/layer").is_in([int(layer) for layer in layers])
            ).select(
                "summary/eval/l0",
                "summary/eval/mse",
                "id",
                "config/objective/sparsity_coeff",
                "config/lr",
                "config/sae/d_sae",
                "config/val_data/layer",
                "model_key",
                "is_pareto",
            )
        )
        .mark_point()
        .encode(
            x=alt.X("summary/eval/l0").scale(type="log"),
            y=alt.Y("summary/eval/mse").scale(type="log"),
            tooltip=["id", "config/lr", "config/objective/sparsity_coeff", "is_pareto"],
            color="config/lr:Q",
            shape="is_pareto:N",
            # shape="config/val_data/layer:N",
            # shape="config/objective/sparsity_coeff:N",
            # shape="config/sae/d_sae:N",
            # shape="model_key",
        )
    )
    return


@app.cell
def _(alt, df, layers, mo, models, pl):
    mo.ui.altair_chart(
        alt
        .Chart(
            df.filter(
                pl.col("config/val_data/layer").is_in([int(layer) for layer in layers])
                & pl.col("model_key").is_in(models)
            ).select(
                "summary/eval/l0",
                "summary/eval/mse",
                "id",
                "config/objective/sparsity_coeff",
                "config/lr",
                "config/sae/d_sae",
                "config/val_data/layer",
                "model_key",
                "is_pareto",
            )
        )
        .mark_point()
        .encode(
            x=alt.X("summary/eval/l0").scale(type="log", domain=(1e-1, 1e4)),
            y=alt.Y("summary/eval/mse").scale(type="log"),
            tooltip=["id", "config/lr", "config/objective/sparsity_coeff", "is_pareto"],
            color="config/objective/sparsity_coeff:Q",
            shape="is_pareto:N",
            # shape="config/val_data/layer:N",
            # shape="config/objective/sparsity_coeff:N",
            # shape="config/sae/d_sae:N",
            # shape="model_key",
        )
    )
    return


@app.cell
def _(adjust_text, filtered_df, itertools, mo, pl, plt, saev):
    def plot_lambdawise(df: pl.DataFrame):
        l0_col = "summary/eval/l0"
        mse_col = "summary/eval/mse"
        layer_col = "config/val_data/layer"
        lam_col = "config/objective/sparsity_coeff"
        model_col = "model_key"

        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

        linestyles = ["-", "--", ":", "-."]
        colors = [
            saev.colors.CYAN_RGB01,
            saev.colors.SEA_RGB01,
            saev.colors.CREAM_RGB01,
            saev.colors.GOLD_RGB01,
            saev.colors.ORANGE_RGB01,
            saev.colors.RUST_RGB01,
            saev.colors.SCARLET_RGB01,
            saev.colors.RED_RGB01,
        ]
        markers = ["X", "o", "+"]

        models = sorted(df.select(model_col).unique().get_column(model_col).to_list())
        lams = sorted(df.select(lam_col).unique().get_column(lam_col).to_list())

        texts = []

        for model, marker in zip(models, itertools.cycle(markers)):
            model_df = df.filter(pl.col(model_col) == model)

            layers = model_df.select(layer_col).unique().get_column(layer_col).to_list()
            for layer, linestyle in zip(sorted(layers), itertools.cycle(linestyles)):
                for lam, color in zip(lams, itertools.cycle(colors)):
                    group = model_df.filter(
                        (pl.col(layer_col) == layer) & (pl.col(lam_col) == lam)
                    ).sort(l0_col)

                    line, *_ = ax.plot(
                        group.get_column(l0_col).to_numpy(),
                        group.get_column(mse_col).to_numpy(),
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        alpha=0.8,
                        label=f"Layer {layer} / $\\lambda$ {lam:.1g}",
                    )

        ax.set_xlabel("L0 sparsity (lower is better)")
        ax.set_ylabel("Reconstruction MSE (lower is better)")
        ax.grid(True, linewidth=0.3, alpha=0.7)
        ax.legend(fontsize="small", ncols=2)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xscale("log")
        # ax.set_xlim(1e-1)
        # ax.set_yscale("log")
        # ax.set_ylim(1e1)

        adjust_text(texts)

        return fig

    mo.stop(filtered_df.height == 0, mo.md("No runs match the current filters."))

    plot_lambdawise(filtered_df)
    return


@app.cell
def _(df):
    print("rm -r " + " ".join(df.get_column("id").to_list()))
    return


@app.cell
def _(df, pl):
    df.select(pl.col("^config/objective.*$"))
    return


@app.cell
def _(collections, df, functools, get_nmse, pl, plt, saev):
    def plot_for_publication(
        df: pl.DataFrame,
        l0_col: str = "summary/eval/l0",
        layer_col: str = "config/val_data/layer",
    ):
        mse_col = "val_nmse"
        # mse_col = "summary/eval/mse"

        fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300, layout="constrained")

        pareto_ckpts = collections.defaultdict(list)

        point_alpha = 0.8

        ax.scatter(
            [1.0],
            [0.672],
            color=saev.colors.SEA_RGB01,
            marker="x",
            s=64,
            linewidth=2.5,
            alpha=point_alpha,
            label="$k$-Means",
        )

        ax.plot(
            [
                1,
                4,
                16,
                64,
                256,
                1024,
            ],
            [
                1.0,
                0.9900600238458658,
                0.8605523198923893,
                0.6878579596261332,
                0.35252687913334674,
                1e-9,
            ],
            color=saev.colors.ORANGE_RGB01,
            marker="^",
            alpha=point_alpha,
            label="PCA",
        )

        vanilla_df = df.filter(
            (
                pl.col("objective") == "vanilla"
                # & pl.col("is_pareto")
            )
        ).sort(by=l0_col)

        ids = vanilla_df.get_column("id").to_list()
        xs = vanilla_df.get_column(l0_col).to_numpy()
        ys = vanilla_df.get_column(mse_col).to_numpy()

        line, *_ = ax.plot(
            xs,
            ys,
            color=saev.colors.BLUE_RGB01,
            linestyle="--",
            marker="^",
            alpha=point_alpha,
            label="SAE",
        )

        mat_df = df.filter(
            (
                pl.col("objective") == "matryoshka"
                # & pl.col('is_pareto')
            )
        ).sort(by=l0_col)

        ids = mat_df.get_column("id").to_list()
        xs = mat_df.get_column(l0_col).to_numpy()
        ys = mat_df.get_column(mse_col).to_numpy()

        line, *_ = ax.plot(
            xs,
            ys,
            color=saev.colors.SCARLET_RGB01,
            linestyle="-.",
            marker="s",
            alpha=point_alpha,
            label="Matryoshka",
        )

        ax.set_xlabel("L0 ($\\downarrow$)")
        ax.set_ylabel("Normalized MSE ($\\downarrow$)")
        ax.grid(True, linewidth=0.3, alpha=0.7)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xscale("log")
        # ax.set_xlim(1e0, 1e3)
        ax.legend()

        fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vitl16_in1k_baselines.pdf"
        )
        return fig

    plot_for_publication(
        df
        .filter(
            (pl.col("model_key") == "DINOv3 ViT-L/16")
            & (pl.col("config/train_data/layer") == 23)
        )
        .with_columns(
            pl
            .col("id")
            .map_elements(
                functools.partial(get_nmse, shards="614861a0"), return_dtype=pl.Float64
            )
            .alias("train_nmse"),
            pl
            .col("id")
            .map_elements(
                functools.partial(get_nmse, shards="3802cb66"), return_dtype=pl.Float64
            )
            .alias("val_nmse"),
            pl
            .when(pl.col("config/objective/n_prefixes").is_null())
            .then(pl.lit("vanilla"))
            .otherwise(pl.lit("matryoshka"))
            .alias("objective"),
        )
        .filter(pl.col("train_nmse") < 1)
    )
    return


@app.cell
def _(df, get_nmse, pl, plt, saev):
    def _(
        df: pl.DataFrame,
        l0_col: str = "summary/eval/l0",
        layer_col: str = "config/val_data/layer",
    ):
        mse_col = "ade20k_val_nmse"
        mse_col = "summary/eval/mse"

        fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300, layout="constrained")

        point_alpha = 0.8

        vitl_df = df.filter(
            (
                pl.col("model_key") == "DINOv3 ViT-L/16"  # & pl.col("is_pareto")
            )
        ).sort(by=l0_col)

        ids = vitl_df.get_column("id").to_list()
        xs = vitl_df.get_column(l0_col).to_numpy()
        ys = vitl_df.get_column(mse_col).to_numpy()

        line, *_ = ax.plot(
            xs,
            ys,
            color=saev.colors.SEA_RGB01,
            linestyle="-",
            marker="o",
            alpha=point_alpha,
            label="ViT-L/16",
            clip_on=False,
        )

        vitb_df = df.filter(
            (
                pl.col("model_key") == "DINOv3 ViT-B/16"  # & pl.col("is_pareto")
            )
        ).sort(by=l0_col)

        ids = vitb_df.get_column("id").to_list()
        xs = vitb_df.get_column(l0_col).to_numpy()
        ys = vitb_df.get_column(mse_col).to_numpy()

        line, *_ = ax.plot(
            xs,
            ys,
            color=saev.colors.ORANGE_RGB01,
            linestyle="--",
            marker="^",
            alpha=point_alpha,
            label="ViT-B/16",
            clip_on=False,
        )

        vits_df = df.filter(
            (
                pl.col("model_key") == "DINOv3 ViT-S/16"  # & pl.col("is_pareto")
            )
        ).sort(by=l0_col)

        ids = vits_df.get_column("id").to_list()
        xs = vits_df.get_column(l0_col).to_numpy()
        ys = vits_df.get_column(mse_col).to_numpy()

        line, *_ = ax.plot(
            xs,
            ys,
            color=saev.colors.BLUE_RGB01,
            linestyle="-.",
            marker="s",
            alpha=point_alpha,
            label="ViT-S/16",
            clip_on=False,
        )

        ax.set_xlabel("L$_0$ ($\\downarrow$)")
        ax.set_ylabel("MSE ($\\downarrow$)")
        ax.grid(True, linewidth=0.3, alpha=0.7)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xscale("log")
        # ax.set_xlim(1e0, 2e3)
        ax.set_ylim(9e1, 2e3)
        ax.set_yscale("log")
        ax.legend(loc="lower right")

        fig.savefig("contrib/trait_discovery/docs/assets/dinov3_sizes_in1k_sae.pdf")
        return fig

    _(
        df
        .filter(
            (
                (pl.col("model_key") == "DINOv3 ViT-L/16")
                & (pl.col("config/train_data/layer") == 23)
                & (pl.col("data_key") == "IN1K/train")
                & pl.col("config/objective/n_prefixes").is_not_null()
            )
            | (
                (pl.col("model_key") == "DINOv3 ViT-B/16")
                & (pl.col("config/train_data/layer") == 11)
                & (pl.col("data_key") == "IN1K/train")
                & pl.col("config/objective/n_prefixes").is_not_null()
            )
            | (
                (pl.col("model_key") == "DINOv3 ViT-S/16")
                & (pl.col("config/train_data/layer") == 11)
                & (pl.col("data_key") == "IN1K/train")
                & pl.col("config/objective/n_prefixes").is_not_null()
            )
        )
        .with_columns(
            pl
            .when(pl.col("model_key") == "DINOv3 ViT-S/16")
            .then(pl.lit("5e195bbf"))
            .when(pl.col("model_key") == "DINOv3 ViT-B/16")
            .then(pl.lit("66a5d2c1"))
            .when(pl.col("model_key") == "DINOv3 ViT-L/16")
            .then(pl.lit("3802cb66"))
            .alias("ade20k_val_shards"),
        )
        .with_columns(
            pl
            .struct("id", "ade20k_val_shards")
            .map_elements(
                lambda cols: get_nmse(cols["id"], cols["ade20k_val_shards"]),
                return_dtype=pl.Float64,
            )
            .alias("ade20k_val_nmse"),
        )
        .filter(pl.col("ade20k_val_nmse") < 1)
    )
    return


@app.cell
def _(df, get_nmse, mpl, np, pl, plt):
    def _(
        df: pl.DataFrame,
        l0_col: str = "summary/eval/l0",
        layer_col: str = "config/val_data/layer",
    ):
        # mse_col = "ade20k_val_nmse"
        mse_col = "summary/eval/mse"

        fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300, layout="constrained")

        point_alpha = 0.8

        layers = (13, 15, 17, 19, 21, 23)
        colormap = mpl.colormaps.get_cmap("plasma")
        colors = colormap(np.linspace(0, 1, len(layers)))[:, :3]

        for layer, color in zip(layers, colors):
            layer_df = df.filter(
                (
                    pl.col("config/train_data/layer") == layer
                    # & pl.col("is_pareto")
                )
            ).sort(by=l0_col)

            ids = layer_df.get_column("id").to_list()
            xs = layer_df.get_column(l0_col).to_numpy()
            ys = layer_df.get_column(mse_col).to_numpy()

            line, *_ = ax.plot(
                xs,
                ys,
                color=color,
                marker="o",
                alpha=point_alpha,
                label=f"Layer {layer + 1}",
                clip_on=False,
            )

        ax.set_xlabel("L0 ($\\downarrow$)")
        ax.set_ylabel("MSE ($\\downarrow$)")
        ax.grid(True, linewidth=0.3, alpha=0.7)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xscale("log")
        ax.set_xlim(1e0)
        ax.set_ylim(1e0)
        # ax.set_ylim(-0.05, 1.05)
        ax.set_yscale("log")
        ax.legend(loc="lower right")

        fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vitl_layers_in1k_sae.pdf"
        )
        return fig

    _(
        df
        .filter(
            (
                (pl.col("model_key") == "DINOv3 ViT-L/16")
                & (pl.col("data_key") == "IN1K/train")
                & pl.col("config/objective/n_prefixes").is_not_null()
            )
        )
        .with_columns(
            pl
            .when(pl.col("model_key") == "DINOv3 ViT-S/16")
            .then(pl.lit("5e195bbf"))
            .when(pl.col("model_key") == "DINOv3 ViT-B/16")
            .then(pl.lit("66a5d2c1"))
            .when(pl.col("model_key") == "DINOv3 ViT-L/16")
            .then(pl.lit("3802cb66"))
            .alias("ade20k_val_shards"),
        )
        .with_columns(
            pl
            .struct("id", "ade20k_val_shards")
            .map_elements(
                lambda cols: get_nmse(cols["id"], cols["ade20k_val_shards"]),
                return_dtype=pl.Float64,
            )
            .alias("ade20k_val_nmse"),
        )
        .filter(pl.col("ade20k_val_nmse") < 1)
    )
    return


@app.cell
def _(df, get_nmse, pl):
    (
        df
        .filter(
            (
                (pl.col("model_key") == "DINOv3 ViT-L/16")
                & (pl.col("config/train_data/layer") == 23)
                & (pl.col("data_key") == "IN1K/train")
            )
            | (
                (pl.col("model_key") == "DINOv3 ViT-B/16")
                & (pl.col("config/train_data/layer") == 11)
            )
            | (
                (pl.col("model_key") == "DINOv3 ViT-S/16")
                & (pl.col("config/train_data/layer") == 11)
            )
        )
        .with_columns(
            pl
            .when(pl.col("model_key") == "DINOv3 ViT-S/16")
            .then(pl.lit("5e195bbf"))
            .when(pl.col("model_key") == "DINOv3 ViT-B/16")
            .then(pl.lit("66a5d2c1"))
            .when(pl.col("model_key") == "DINOv3 ViT-L/16")
            .then(pl.lit("3802cb66"))
            .alias("ade20k_val_shards"),
            pl
            .when(pl.col("config/objective/n_prefixes").is_null())
            .then(pl.lit("vanilla"))
            .otherwise(pl.lit("matryoshka"))
            .alias("objective"),
        )
        .with_columns(
            pl
            .struct("id", "ade20k_val_shards")
            .map_elements(
                lambda cols: get_nmse(cols["id"], cols["ade20k_val_shards"]),
                return_dtype=pl.Float64,
            )
            .alias("val_nmse"),
        )
        .select("id", "model_key", "data_key", "ade20k_val_shards", "val_nmse")
    )
    return


@app.cell
def _():
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(beartype, df, json, pl, saev):
    import functools
    import pathlib

    @beartype.beartype
    def get_nmse(id: str, shards: str) -> float:
        nmse = 1.0
        try:
            run = saev.disk.Run(
                pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs") / id
            )

            path = run.inference / shards / "metrics.json"

            nmse = json.loads(path.read_text())["normalized_mse"]
        except FileNotFoundError:
            pass
        return nmse

    df.filter(pl.col("model_key") == "DINOv3 ViT-L/16").with_columns(
        pl
        .col("id")
        .map_elements(
            functools.partial(get_nmse, shards="614861a0"), return_dtype=pl.Float64
        )
        .alias("train_nmse"),
        pl
        .col("id")
        .map_elements(
            functools.partial(get_nmse, shards="3802cb66"), return_dtype=pl.Float64
        )
        .alias("val_nmse"),
    ).select("id", "train_nmse", "val_nmse").filter(pl.col("train_nmse") < 1)
    return functools, get_nmse


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df, mpl, np, pl, plt):
    def _(df: pl.DataFrame):
        # mse_col = "ade20k_val_nmse"
        mse_col = "summary/eval/mse"
        lr_col = "config/lr"
        layer_col = ("config/val_data/layer",)

        fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300, layout="constrained")

        point_alpha = 0.8

        layers = (13, 15, 17, 19, 21, 23)
        colormap = mpl.colormaps.get_cmap("plasma")
        colors = colormap(np.linspace(0, 1, len(layers)))[:, :3]

        for layer, color in zip(layers, colors):
            layer_df = df.filter(
                (
                    pl.col("config/train_data/layer") == layer
                    # & pl.col("is_pareto")
                )
            )

            ids = layer_df.get_column("id").to_list()
            xs = layer_df.get_column(lr_col).to_numpy()
            ys = layer_df.get_column(mse_col).to_numpy()

            line, *_ = ax.plot(
                xs,
                ys,
                color=color,
                marker="o",
                alpha=point_alpha,
                label=f"Layer {layer + 1}",
                clip_on=False,
            )

        ax.set_xlabel("L0 ($\\downarrow$)")
        ax.set_ylabel("MSE ($\\downarrow$)")
        ax.grid(True, linewidth=0.3, alpha=0.7)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="lower right")

        return fig

    _(df.filter((pl.col("model_key") == "DINOv3 ViT-L/16")))
    return


if __name__ == "__main__":
    app.run()
