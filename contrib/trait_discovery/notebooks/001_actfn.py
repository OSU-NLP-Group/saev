import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import collections
    import concurrent.futures
    import itertools
    import json
    import os.path
    import pathlib
    import pickle

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import wandb
    from jaxtyping import Float, jaxtyped

    import saev.colors
    import saev.data.datasets

    return (
        Float,
        base64,
        beartype,
        collections,
        concurrent,
        itertools,
        jaxtyped,
        json,
        mo,
        np,
        os,
        pathlib,
        pickle,
        pl,
        plt,
        saev,
        wandb,
    )


@app.cell
def _(pathlib):
    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "saev"
    WANDB_TAG = "arch-comparison-v0.2"

    runs_root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
    shards_root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")
    return WANDB_PROJECT, WANDB_TAG, WANDB_USERNAME, runs_root, shards_root


@app.cell
def _(
    WANDB_PROJECT,
    WANDB_TAG,
    WANDB_USERNAME,
    beartype,
    concurrent,
    get_baseline_ce,
    get_data_key,
    get_inference_probe_metric_fpaths,
    get_model_key,
    get_probe_split_label,
    json,
    load_freqs,
    load_mean_values,
    mo,
    mode,
    np,
    pathlib,
    pl,
    runs_root,
    saev,
    shards_root,
    wandb,
):
    @beartype.beartype
    def _row_from_run(wandb_run) -> dict[str, object] | None:
        saev_run = saev.disk.Run(runs_root / wandb_run.id)
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
            sae_cfg = config.pop("sae")
            obj_cfg = config.pop("objective")
        except KeyError as err:
            print(f"Run {wandb_run.id} missing config section: {err}.")
            return None

        row.update(**{
            f"config/train_data/{key}": value for key, value in train_data.items()
        })
        row.update(**{
            f"config/val_data/{key}": value for key, value in val_data.items()
        })
        row.update(**{f"config/sae/{key}": value for key, value in sae_cfg.items()})
        row.update(**{
            f"config/objective/{key}": value for key, value in obj_cfg.items()
        })

        row.update(**{f"config/{key}": value for key, value in config.items()})

        metadata = row.get("config/train_data/metadata")
        assert metadata is not None, f"Run {wandb_run.id} missing metadata"

        row["model_key"] = get_model_key(metadata)
        row["data_key"] = get_data_key(metadata)
        row["config/d_model"] = metadata["d_model"]

        split_map: dict[str, tuple[pathlib.Path, str, pathlib.Path]] = {}
        for metrics_fpath in get_inference_probe_metric_fpaths(saev_run.run_dir):
            shard_id = metrics_fpath.parent.name
            shards_dpath = shards_root / shard_id

            if not shards_dpath.exists():
                print(f"Skipping {wandb_run.id}: shards dir {shards_dpath} missing.")
                continue

            split_label = get_probe_split_label(shards_dpath)
            if split_label is None:
                print(
                    f"Skipping shards {shard_id}: unknown split (run {wandb_run.id})."
                )
                continue

            if split_label in split_map:
                print(f"Skipping {wandb_run.id}: duplicate {split_label} metrics.")
                split_map = {}
                break

            split_map[split_label] = (metrics_fpath, shard_id, shards_dpath)

        if not split_map:
            print(f"Skipping {wandb_run.id}: no splits.")
            return row

        if "train" not in split_map:
            print(f"Skipping {wandb_run.id}: missing train, got {split_map.keys()}.")
            return row

        if "val" not in split_map:
            print(f"Skipping {wandb_run.id}: missing val, got {split_map.keys()}.")
            return row

        train_probe_metrics_fpath, train_shards, train_shards_dpath = split_map["train"]
        val_probe_metrics_fpath, val_shards, val_shards_dpath = split_map["val"]

        with np.load(train_probe_metrics_fpath) as fd:
            train_loss = fd["loss"]
            w = fd["weights"]

        with np.load(val_probe_metrics_fpath) as fd:
            val_loss = fd["loss"]

        assert train_loss.ndim == 2
        assert val_loss.ndim == 2
        assert train_loss.shape == val_loss.shape

        n_latents, n_classes = train_loss.shape

        best_i = np.argmin(train_loss, axis=0)
        train_ce = train_loss[best_i, np.arange(n_classes)].mean().item()
        val_ce = val_loss[best_i, np.arange(n_classes)].mean().item()

        row["downstream/frac_w_neg"] = (w < 0).mean().item()
        row["downstream/frac_best_w_neg"] = (
            (w[best_i, np.arange(n_classes)] < 0).mean().item()
        )

        train_base_ce = get_baseline_ce(train_shards_dpath).mean().item()
        val_base_ce = get_baseline_ce(val_shards_dpath).mean().item()

        row["downstream/train/probe_ce"] = train_ce
        row["downstream/train/baseline_ce"] = train_base_ce
        row["downstream/train/probe_r"] = 1 - train_ce / train_base_ce

        row["downstream/val/probe_ce"] = val_ce
        row["downstream/val/baseline_ce"] = val_base_ce
        row["downstream/val/probe_r"] = 1 - val_ce / val_base_ce

        path = saev_run.inference / train_shards / "metrics.json"
        if path.is_file():
            nmse = json.loads(path.read_text())["normalized_mse"]
            row["downstream/train/normalized_mse"] = nmse

        path = saev_run.inference / val_shards / "metrics.json"
        if path.is_file():
            nmse = json.loads(path.read_text())["normalized_mse"]
            row["downstream/val/normalized_mse"] = nmse

        # k = 16
        path = (
            saev_run.inference
            / val_shards
            / f"probe1d_metrics__train-{train_shards}.npz"
        )
        if path.is_file():
            with np.load(path) as fd:
                ap_c = fd["ap"]
                prec_c = fd["precision"]
                recall_c = fd["recall"]
                f1_c = fd["f1"]
                top_labels_dk = fd["top_labels"]

            row["downstream/val/mean_ap"] = ap_c.mean().item()
            row["downstream/val/mean_prec"] = prec_c.mean().item()
            row["downstream/val/mean_recall"] = recall_c.mean().item()
            row["downstream/val/mean_f1"] = f1_c.mean().item()
            for tau in [0.3, 0.5, 0.7]:
                row[f"downstream/val/cov_at_{tau}".replace(".", "_")] = (
                    (ap_c > tau).mean().item()
                )

            for k in [16, 64, 256]:
                _, count = mode(top_labels_dk[best_i, :k], axis=1)
                row[f"downstream/va/mean_purity_at_{k}"] = (count / k).mean().item()

        return row

    @beartype.beartype
    def _finalize_df(rows: list[dict[str, object]]):
        df = pl.DataFrame(rows, infer_schema_length=None)

        activation_schema = pl.Struct([
            pl.Field("sparsity", pl.Struct([pl.Field("coeff", pl.Float64)])),
            pl.Field("top_k", pl.Int64),
        ])

        df = df.with_columns(
            pl.col("config/sae/activation").cast(
                activation_schema, strict=False
            )  # add missing fields as null
        )

        df = df.with_columns(
            pl
            .when(pl.col("config/sae/activation").struct.field("top_k").is_not_null())
            .then(pl.lit("topk"))
            .otherwise(pl.lit("relu"))
            .alias("config/sae/activation_kind"),
        )

        group_cols = (
            "model_key",
            "config/val_data/layer",
            "data_key",
            "config/sae/activation_kind",
            "config/sae/reinit_blend",
        )

        # Compute Pareto explicitly
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
    def make_df_parallel(n_workers: int = 16):
        filters = {}

        filters["config.tag"] = WANDB_TAG

        runs = list(
            wandb.Api().runs(path=f"{WANDB_USERNAME}/{WANDB_PROJECT}", filters=filters)
        )
        # runs = runs[:8]
        if not runs:
            raise ValueError("No runs found.")

        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            fut_to_run_id = {pool.submit(_row_from_run, run): run.id for run in runs}
            for fut in mo.status.progress_bar(
                concurrent.futures.as_completed(fut_to_run_id),
                total=len(fut_to_run_id),
                remove_on_exit=True,
            ):
                try:
                    row = fut.result()
                except Exception as err:
                    print(f"Run {fut_to_run_id[fut]} blew up: {err}")
                    continue
                if row is None:
                    continue
                rows.append(row)

        assert rows, "No valid runs."
        return _finalize_df(rows)

    df = make_df_parallel()
    return (df,)


@app.cell
def _(df, pl):
    df.filter(
        pl.col("is_pareto"), pl.col("downstream/frac_w_neg").is_not_null()
    ).select("id", "^downstream/.*$")
    return


@app.cell
def _(df, np, pl, plt):
    def _():
        fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300, layout="constrained")
        ks, ys, ids = (
            df
            .filter(pl.col("config/sae/activation_kind") == "topk")
            .group_by(pl.col("config/sae/activation").struct.field("top_k"))
            .agg(pl.col("summary/eval/l0"), pl.col("id"))
            .sort(by="top_k")
        )
        ks = ks.to_numpy()
        ys = ys.to_numpy()

        # ids = np.array(ids.to_list())
        # ax.boxplot(ys)

        ax.violinplot(ys)

        for i, y in enumerate(ys):
            ax.axhline(ks[i], color="tab:red", alpha=0.9, linewidth=0.1)
            print(
                f"k={ks[i]}:\trun {ids[i, y.argmin().item()]} -> {y.min().item():.4f}"
            )

        ax.set_xticks(np.arange(len(ks)) + 1, [f"$k$={k}" for k in ks])
        ax.set_yscale("log")
        ax.set_ylabel("Observed L$_0$")
        ax.spines[["top", "right"]].set_visible(False)

        return fig

    _()
    return


@app.cell
def _(df, pl):
    df.filter(
        True
        & pl.col("is_pareto")
        & (pl.col("config/sae/activation_kind") == "topk")
        & (pl.col("config/sae/activation").struct.field("top_k") == 8)
        & (pl.col("config/val_data/layer") == 23)
        & (pl.col("data_key") == "FishVista (Img)")
        & (pl.col("config/sae/reinit_blend") == 0.0)
    )
    #     .select(
    #     "id",
    #     pl.col("config/sae/activation").struct.field("top_k"),
    #     "summary/eval/l0",
    #     "summary/eval/normalized_mse",
    # ).sort("summary/eval/l0")
    return


@app.cell
def _(collections, df, itertools, mo, pl, plt, saev):
    def _(df: pl.DataFrame):
        # mse_col = "ade20k_val_nmse"
        x_col = "summary/eval/l0"
        y_col = "summary/eval/normalized_mse"

        layer_col = "config/val_data/layer"

        point_alpha = 0.5

        layers = [13, 15, 17, 19, 21, 23]
        data_keys = ["FishVista (Img)", "IN1K/train"]

        activation_fns = [
            ("relu", "ReLU", "-"),
            ("topk", "TopK", "--"),
        ]
        blends = [(0.0, "o"), (0.8, "^")]

        colors = [
            saev.colors.BLUE_RGB01,
            saev.colors.SEA_RGB01,
            saev.colors.ORANGE_RGB01,
            saev.colors.SCARLET_RGB01,
        ]

        fig, axes = plt.subplots(
            figsize=(8, 10),
            nrows=4,
            ncols=3,
            dpi=300,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes = axes.reshape(-1)

        pareto_ckpts = collections.defaultdict(list)

        for i, ((data_key, layer), ax) in enumerate(
            zip(itertools.product(data_keys, layers), axes)
        ):
            texts = []
            for ((kind, label, linestyle), (blend, marker)), color in zip(
                itertools.product(activation_fns, blends), colors
            ):
                group = df.filter(
                    (pl.col("config/sae/activation_kind") == kind)
                    & (pl.col("config/sae/reinit_blend") == blend)
                    & (pl.col("config/val_data/layer") == layer)
                    & (pl.col("data_key") == data_key)
                )
                group = group.sort(by=x_col)

                pareto = group.filter(
                    pl.col("is_pareto") & (pl.col(x_col) > 1) & (pl.col(x_col) < 1000)
                    # & (pl.col(x_col) < 800)
                )
                if pareto.height == 0:
                    continue

                ids = pareto.get_column("id").to_list()
                xs = pareto.get_column(x_col).to_numpy()
                ys = pareto.get_column(y_col).to_numpy()

                line, *_ = ax.plot(
                    xs,
                    ys,
                    alpha=0.5,
                    label=f"{label}/p={blend:.1f}",
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                )

                # if layer in (2:
                print(layer, data_key, blend, kind, len(ids))
                pareto_ckpts[(layer, data_key, blend, kind)].extend(ids)

                if kind == "topk":
                    print(xs, ys)

                # edge_mask = pl.col("is_lr_min") | pl.col("is_lr_max")
                # edge_df = pareto.filter(edge_mask)

                # if edge_df.height > 0:
                #     edge_xs = edge_df.get_column(x_col).to_numpy()
                #     edge_ys = edge_df.get_column(y_col).to_numpy()
                #     ax.scatter(
                #         edge_xs,
                #         edge_ys,
                #         facecolors="none",
                #         edgecolors="tab:red",
                #         marker=marker,
                #         s=60,
                #         linewidths=1.2,
                #         zorder=line.get_zorder() + 1,
                #     )

                # lr_min = pareto.get_column("is_lr_min").to_list()
                # lr_max = pareto.get_column("is_lr_max").to_list()

                # for x, y, rid, is_lr_min, is_lr_max in zip(xs, ys, ids, lr_min, lr_max):
                #     edge_parts = []
                #     if is_lr_min:
                #         edge_parts.append("LR min")
                #     if is_lr_max:
                #         edge_parts.append("LR max")

                #     label = rid if not edge_parts else f"{rid} ({', '.join(edge_parts)})"
                #     color_text = "tab:red" if edge_parts else "black"
                #     texts.append(
                #         ax.text(
                #             x,
                #             y,
                #             label,
                #             fontsize=8,
                #             color=color_text,
                #             ha="left",
                #             va="bottom",
                #         )
                #     )

            # adjust_text(texts)

            if i in (9, 10, 11):
                ax.set_xlabel("L$_0$ ($\\downarrow$)")

            if i in (0, 3, 6, 9):
                ax.set_ylabel("Normalized MSE ($\\downarrow$)")

            if i in (0,):
                ax.legend()

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.set_title(f"Layer {layer + 1}/24 ({data_key})")

        fig.savefig("contrib/trait_discovery/docs/assets/relu-topk-pareto.pdf")
        # return fig

        return mo.vstack([fig, dict(pareto_ckpts)])

    _(df)
    return


@app.cell
def _(df, pl):
    df.filter(pl.col("is_pareto")).sort(
        "model_key",
        "data_key",
        "config/sae/activation_kind",
        "config/val_data/layer",
        "summary/eval/l0",
    ).select(
        "id",
        "model_key",
        "config/val_data/layer",
        "data_key",
        "config/sae/activation_kind",
        "summary/eval/l0",
        "summary/eval/normalized_mse",
    ).filter(
        (pl.col("config/val_data/layer") == 13) & (pl.col("data_key") != "IN1K/train")
    )
    return


@app.cell
def _(collections, df, mo, pl, plt, saev):
    def _(df: pl.DataFrame):
        # mse_col = "ade20k_val_nmse"
        x_col = "config/lr"
        y_col = "summary/eval/normalized_mse"

        layer_col = "config/val_data/layer"

        point_alpha = 0.5

        layers = [13, 15, 17, 19, 21, 23]
        ks = [
            # (8, saev.colors.BLUE_RGB01, "o", "-"),
            (32, saev.colors.SEA_RGB01, "^", "--"),
            (128, saev.colors.CREAM_RGB01, "s", "-."),
            (512, saev.colors.ORANGE_RGB01, "x", ":"),
        ]

        fig, axes = plt.subplots(
            figsize=(8, 5),
            nrows=2,
            ncols=3,
            dpi=300,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes = axes.reshape(-1)

        pareto_ckpts = collections.defaultdict(list)
        texts = []

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            for k, color, marker, linestyle in ks:
                group = df.filter(
                    (pl.col("config/sae/activation_kind") == "topk")
                    & (pl.col("config/val_data/layer") == layer)
                    & (pl.col("config/sae/activation").struct.field("top_k") == k)
                ).sort(by=x_col)

                ids = group.get_column("id").to_list()
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()

                line, *_ = ax.plot(
                    xs,
                    ys,
                    alpha=0.8,
                    label=f"$k$={k}",
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                )

                # for x, y, rid in zip(xs, ys, ids):
                #     ax.text(x, y, rid, fontsize=12, color="black", ha="left", va="bottom")

            if i in (3, 4, 5):
                ax.set_xlabel("Learning Rate")

            if i in (0, 3):
                ax.set_ylabel("Normalized MSE ($\\downarrow$)")

            if i in (5,):
                ax.legend()

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.set_title(f"Layer {layer + 1}/24")

        fig.suptitle("FishVista (TopK)")

        fig.savefig("contrib/trait_discovery/docs/assets/topk-fishvista-lr.pdf")

        txt = mo.md(
            """
    # Learning Rate Sweeps

    How does learning rate affect performance? For a fixed $k$, what is the optimal learning rate?

    """.strip()
        )

        return mo.vstack([txt, fig])

    _(df)
    return


@app.cell
def _(collections, df, mo, pl, plt, saev):
    def _(df: pl.DataFrame):
        # mse_col = "ade20k_val_nmse"
        x_col = "config/lr"
        y_col = "summary/eval/normalized_mse"

        layer_col = "config/val_data/layer"

        point_alpha = 0.5

        layers = [13, 15, 17, 19, 21, 23]
        lambdas = [
            (1e-4, saev.colors.BLUE_RGB01, "o", "-"),
            (1e-3, saev.colors.SEA_RGB01, "^", "--"),
            (1e-2, saev.colors.CREAM_RGB01, "s", "-."),
            (1e-1, saev.colors.ORANGE_RGB01, "x", ":"),
        ]

        fig, axes = plt.subplots(
            figsize=(8, 5),
            nrows=2,
            ncols=3,
            dpi=300,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes = axes.reshape(-1)

        pareto_ckpts = collections.defaultdict(list)
        texts = []

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            for lam, color, marker, linestyle in lambdas:
                group = df.filter(
                    (pl.col("config/sae/activation_kind") == "relu")
                    & (pl.col("config/val_data/layer") == layer)
                    & (
                        pl
                        .col("config/sae/activation")
                        .struct.field("sparsity")
                        .struct.field("coeff")
                        == lam
                    )
                ).sort(by=x_col)

                ids = group.get_column("id").to_list()
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()

                line, *_ = ax.plot(
                    xs,
                    ys,
                    alpha=0.8,
                    label=f"$\\lambda$={lam:.1g}",
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                )

                # for x, y, rid in zip(xs, ys, ids):
                #     ax.text(x, y, rid, fontsize=12, color="black", ha="left", va="bottom")

            if i in (3, 4, 5):
                ax.set_xlabel("Learning Rate")

            if i in (0, 3):
                ax.set_ylabel("Normalized MSE ($\\downarrow$)")

            if i in (5,):
                ax.legend()

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.set_title(f"Layer {layer + 1}/24")

        fig.suptitle("FishVista (ReLU + L1)")

        fig.savefig("contrib/trait_discovery/docs/assets/relu-fishvista-lr.pdf")

        txt = mo.md(
            """
    # ReLU Learning Rate Sweeps

    How does learning rate affect performance? For a fixed $\\lambda$, what is the optimal learning rate?

    """.strip()
        )

        return mo.vstack([txt, fig])

    _(df)
    return


@app.cell
def _(collections, df, mo, pl, plt, saev):
    def _(df: pl.DataFrame):
        # mse_col = "ade20k_val_nmse"
        x_col = "config/lr"
        y_col = "summary/eval/l0"

        layer_col = "config/val_data/layer"

        point_alpha = 0.5

        layers = [13, 15, 17, 19, 21, 23]
        lambdas = [
            (1e-4, saev.colors.BLUE_RGB01, "o", "-"),
            (1e-3, saev.colors.SEA_RGB01, "^", "--"),
            (1e-2, saev.colors.CREAM_RGB01, "s", "-."),
            (1e-1, saev.colors.ORANGE_RGB01, "x", ":"),
        ]

        fig, axes = plt.subplots(
            figsize=(8, 5),
            nrows=2,
            ncols=3,
            dpi=300,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes = axes.reshape(-1)

        pareto_ckpts = collections.defaultdict(list)
        texts = []

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            for lam, color, marker, linestyle in lambdas:
                group = df.filter(
                    (pl.col("config/sae/activation_kind") == "relu")
                    & (pl.col("config/val_data/layer") == layer)
                    & (
                        pl
                        .col("config/sae/activation")
                        .struct.field("sparsity")
                        .struct.field("coeff")
                        == lam
                    )
                ).sort(by=x_col)

                ids = group.get_column("id").to_list()
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()

                line, *_ = ax.plot(
                    xs,
                    ys,
                    alpha=0.8,
                    label=f"$\\lambda$={lam:.1g}",
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                )

                # for x, y, rid in zip(xs, ys, ids):
                #     ax.text(x, y, rid, fontsize=12, color="black", ha="left", va="bottom")

            if i in (3, 4, 5):
                ax.set_xlabel("Learning Rate")

            if i in (0, 3):
                ax.set_ylabel("L$_0$ ($\\downarrow$)")

            if i in (5,):
                ax.legend()

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.set_title(f"Layer {layer + 1}/24")

        fig.suptitle("FishVista (ReLU + L1)")

        fig.savefig("contrib/trait_discovery/docs/assets/relu-fishvista-lr-l0.pdf")

        txt = mo.md(
            """
    # ReLU Learning Rate Sweeps

    How does learning rate affect performance? For a fixed $\\lambda$, what is the optimal learning rate?

    """.strip()
        )

        return mo.vstack([txt, fig])

    _(df)
    return


@app.cell
def _(df, pl):
    df.filter(
        pl.col("is_pareto")
        & pl.col("downstream/val/mean_ap").is_null()
        & (pl.col("config/val_data/layer") == 23)
    ).select("id", "^downstream.*$")
    return


@app.cell
def _(df, itertools, np, pl, plt, saev):
    def _(df: pl.DataFrame):
        # mse_col = "ade20k_val_nmse"
        x_col = "summary/eval/normalized_mse"
        # x_col = 'summary/eval/n_dead'
        y_col = "downstream/val/mean_ap"

        alpha = 0.5

        data_keys = ["FishVista (Img)", "IN1K/train"]

        activation_fns = [
            ("relu", "ReLU", "-"),
            ("topk", "TopK", "--"),
        ]
        blends = [(0.0, "o"), (0.8, "^")]

        colors = [
            saev.colors.BLUE_RGB01,
            saev.colors.SEA_RGB01,
            saev.colors.ORANGE_RGB01,
            saev.colors.SCARLET_RGB01,
        ]

        fig, axes = plt.subplots(
            figsize=(8, 4),
            nrows=1,
            ncols=2,
            dpi=300,
            sharex=False,
            sharey=True,
            layout="constrained",
        )
        axes = axes.reshape(-1)

        for i, (data_key, ax) in enumerate(zip(data_keys, axes)):
            for ((act_key, label, linestyle), (blend, marker)), color in zip(
                itertools.product(activation_fns, blends), colors
            ):
                group = df.filter(
                    (pl.col("config/sae/activation_kind") == act_key)
                    & pl.col(y_col).is_not_null()
                    & pl.col("is_pareto")
                    & (pl.col("config/val_data/layer") == 23)
                    & (pl.col("data_key") == data_key)
                    & (pl.col("config/sae/reinit_blend") == blend)
                ).sort(by=x_col)

                ids = group.get_column("id").to_list()
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()

                ax.scatter(
                    xs,
                    ys,
                    alpha=alpha,
                    label=f"{label}; $p$={blend}",
                    color=color,
                    marker=marker,
                )

                try:
                    line = np.polynomial.Polynomial.fit(np.log10(xs), ys, deg=2)
                    x_line, y_line = line.linspace()
                    ax.plot(
                        10**x_line,
                        y_line,
                        color=color,
                        linestyle=linestyle,
                        alpha=alpha,
                    )
                except ValueError:
                    pass

            if i in (0, 1):
                ax.set_xlabel(x_col)

            if i in (0,):
                ax.set_ylabel(y_col)

            if i in (1,):
                ax.legend()

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")

            ax.set_title(data_key)

        return fig

    _(df)
    return


@app.cell
def _(df, mo, pl):
    def _(df):
        col = "summary/eval/n_dead"
        # col = "summary/eval/n_almost_dead"

        dfs = []

        for data_key in ["FishVista (Img)", "IN1K/train"]:
            print(
                data_key,
                df.filter(
                    pl.col(col).is_not_null()
                    & pl.col("is_pareto")
                    & (pl.col("data_key") == data_key)
                    # & (pl.col("config/val_data/layer") != 23)
                ).with_columns(
                    (
                        pl.col("downstream/train/probe_r")
                        == pl.col("downstream/train/probe_r").max()
                    )
                    .over(
                        "config/sae/reinit_blend",
                        "config/val_data/layer",
                        "config/sae/activation_kind",
                    )
                    .alias("best_train_probe_r")
                ),  # .select('best_train_probe_r').unique(),
            )
            group = (
                df
                .filter(
                    pl.col(col).is_not_null()
                    & pl.col("is_pareto")
                    & (pl.col("data_key") == data_key)
                    & (pl.col("config/val_data/layer") != 23)
                )
                .with_columns(
                    (
                        pl.col("downstream/train/probe_r")
                        == pl.col("downstream/train/probe_r").max()
                    )
                    .over(
                        "config/sae/activation_kind",
                        "config/sae/reinit_blend",
                        "config/val_data/layer",
                    )
                    .alias("best_train_probe_r")
                )
                .filter(pl.col("best_train_probe_r"))
                .select(
                    "id",
                    "data_key",
                    "config/sae/activation_kind",
                    "config/sae/reinit_blend",
                    "config/val_data/layer",
                    "summary/eval/l0",
                    "summary/eval/normalized_mse",
                    "downstream/train/probe_r",
                    "downstream/val/probe_r",
                    "downstream/val/mean_ap",
                    "downstream/val/mean_f1",
                    "downstream/val/cov_at_0_3",
                    # "^downstream/val.*$",
                )
            )

            dfs.append(group)

        return mo.vstack([mo.md("# Probe Results"), *dfs])

    _(df)
    return


@app.cell
def _(df, mo, pl):
    def _(df):
        col = "summary/eval/n_dead"
        # col = "summary/eval/n_almost_dead"

        dfs = []

        for kind in ["relu", "topk"]:
            group = (
                df
                .filter(
                    (pl.col("config/sae/activation_kind") == kind)
                    & pl.col(col).is_not_null()
                    # & pl.col("is_pareto")
                    # & (pl.col("config/val_data/layer") == 23)
                )
                .group_by(
                    pl.col("config/sae/activation_kind").alias("act_key"),
                    pl.col("data_key"),
                    pl.col("config/sae/reinit_blend").alias("blend"),
                )
                .agg(
                    pl.count().alias("n_trials"),
                    (pl.col(col) / (1024 * 16) * 100).mean().alias("mean_pct"),
                    (pl.col(col) / (1024 * 16) * 100).std().alias("std_pct"),
                )
                .sort("blend", "data_key")
            )

            dfs.append(group)

        return mo.vstack([mo.md("# Dead Units"), *dfs])

    _(df)
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
                print(fpath)
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
                print(fpath)
                with open(fpath) as fd:
                    raw = json.load(fd)
                return np.array(raw["data"], dtype=float).reshape(-1)
        except Exception as err:
            raise RuntimeError(f"Wandb sucks: {err}") from err

        raise ValueError(f"mean_values not found in run '{run.id}'")

    return load_freqs, load_mean_values


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
def _(beartype, pathlib):
    @beartype.beartype
    def get_inference_probe_metric_fpaths(
        run_dpath: pathlib.Path,
    ) -> list[pathlib.Path]:
        if not run_dpath.is_dir():
            return []

        inference_dpath = run_dpath / "inference"
        if not inference_dpath.is_dir():
            return []

        probe_metric_fpaths: list[pathlib.Path] = []
        for shard_dpath in inference_dpath.iterdir():
            if not shard_dpath.is_dir():
                continue

            probe_metrics_fpath = shard_dpath / "probe1d_metrics.npz"
            if not probe_metrics_fpath.is_file():
                continue

            probe_metric_fpaths.append(probe_metrics_fpath)

        return probe_metric_fpaths

    return (get_inference_probe_metric_fpaths,)


@app.cell
def _(beartype, pathlib, saev):
    @beartype.beartype
    def get_probe_split_label(shards_dpath: pathlib.Path) -> str | None:
        try:
            metadata = saev.data.Metadata.load(shards_dpath)
        except Exception as err:
            print(f"Failed to load metadata for {shards_dpath}: {err}")
            return None

        data_cfg = metadata.make_data_cfg()
        split = getattr(data_cfg, "split", None)
        if split is None:
            return None

        split_name = str(split).lower()
        if split_name in {"train", "training"}:
            return "train"
        if split_name in {"val", "validation"}:
            return "val"
        return None

    return (get_probe_split_label,)


@app.cell
def _(beartype, mo, np, pathlib, saev):
    @mo.cache
    @beartype.beartype
    def get_baseline_ce(shards_dir: pathlib.Path):
        md = saev.data.Metadata.load(shards_dir)
        labels = np.memmap(
            shards_dir / "labels.bin",
            mode="r",
            dtype=np.uint8,
            shape=(md.n_examples, md.content_tokens_per_example),
        )

        n_samples = np.prod(labels.shape)

        n_classes = labels.max() + 1
        # Convert to one-hot encoding
        y = np.zeros((n_samples, n_classes), dtype=float)
        y[np.arange(n_samples), labels.reshape(n_samples)] = 1.0

        prob = y.mean(axis=0)
        return -(prob * np.log(prob) + (1 - prob) * np.log(1 - prob))

    return (get_baseline_ce,)


@app.cell
def _(np):
    def mode(a, axis=0):
        scores = np.unique(np.ravel(a))  # get ALL unique values
        testshape = list(a.shape)
        testshape[axis] = 1
        oldmostfreq = np.zeros(testshape)
        oldcounts = np.zeros(testshape)

        for score in scores:
            template = a == score
            counts = np.expand_dims(np.sum(template, axis), axis)
            mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
            oldcounts = np.maximum(counts, oldcounts)
            oldmostfreq = mostfrequent

        return mostfrequent, oldcounts

    return (mode,)


if __name__ == "__main__":
    app.run()
