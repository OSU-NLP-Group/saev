import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import collections
    import concurrent.futures
    import json
    import os.path
    import pathlib
    import pickle

    import beartype
    import cloudpickle
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import scipy.optimize
    import sklearn.metrics
    import wandb
    from jaxtyping import Float, jaxtyped

    import saev.colors
    import saev.data.datasets

    return (
        Float,
        base64,
        beartype,
        cloudpickle,
        collections,
        concurrent,
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
        scipy,
        sklearn,
        wandb,
    )


@app.cell
def _(np):
    def jitter(n, cat_width=0.3, data_width=0.0):
        """Compute random offsets for jitter plot.

        Args:
            n: Number of points
            cat_width: Jitter width in category units (x-axis, typically 0-6)
            data_width: Jitter width in data units (y-axis, typically 0-1)
        """
        cat = np.random.uniform(-cat_width / 2, cat_width / 2, n)
        data = np.random.uniform(-data_width / 2, data_width / 2, n)
        return cat, data

    return (jitter,)


@app.cell
def _(pathlib):
    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "saev"
    # Fetch runs from both tags (IN1K SAEs for ADE20K eval, FishVista SAEs)
    WANDB_TAGS = ["auxk-comparison-v0.3", "fishbase-v0.1"]

    runs_root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
    shards_root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")

    # Run IDs from contrib/trait_discovery/sweeps/006_proposal_audit/cls_train.py
    FISHVISTA_RUN_IDS = {
        13: ["vjyiz6qo", "4j4cpxpj", "ltpubtmx", "xcqixn3v", "ut004yhy"],
        15: ["3lcqylos", "du0p6063", "bldfz1qi", "qz686gdn", "hp93bxwi"],
        17: ["ihvb8175", "rx0y07bl", "ctftp72w", "qh9mnelt", "48op2zys"],
        19: ["jnl93dlg", "1gywxpjg", "cvjrkpo1", "qnze2wzc", "dwnwbjo9"],
        21: ["fpgvte58", "9ol8p6x7", "u6b884y1", "g2mkhipq", "nuekzgyn"],
        23: ["pdikj9bl", "hfpct5ae", "s465wgg4", "dc86xg8z", "bpz34d80"],
    }
    IN1K_RUN_IDS = {
        13: ["3ld8ilmo", "l03epvhu", "co7dpa0w", "kpadjov4", "2edpn91i", "1up044nl"],
        15: ["6r92o6t6", "e4w7u0np", "jsr327fs", "emz255bp", "ffqb9b3n", "3hzenf5e"],
        17: ["tkdd41tq", "4g4lbmgs", "h8nfg6ci", "2hsh4w50", "jjz6a7ja", "huzxe3hu"],
        19: ["0c4mlnn7", "6x4t5t76", "xk0a9w3g", "cdu13t6j", "hh7d7yop", "32zm1zcd"],
        21: ["rez38zbu", "jxxje744", "2k6kq9f2", "jttb6ijl", "s5srn2q7", "qurkdz1r"],
        23: ["a95jzikd", "elwq2g19", "ztnu4ml1", "flqkcam7", "s3pqewz1", "l8hooa3r"],
    }
    ALL_RUN_IDS = set()
    for ids in FISHVISTA_RUN_IDS.values():
        ALL_RUN_IDS.update(ids)
    for ids in IN1K_RUN_IDS.values():
        ALL_RUN_IDS.update(ids)
    return (
        ALL_RUN_IDS,
        WANDB_PROJECT,
        WANDB_TAGS,
        WANDB_USERNAME,
        runs_root,
        shards_root,
    )


@app.cell
def _(
    ALL_RUN_IDS,
    WANDB_PROJECT,
    WANDB_TAGS,
    WANDB_USERNAME,
    beartype,
    concurrent,
    get_cls_results,
    get_data_key,
    get_model_key,
    load_freqs,
    load_mean_values,
    mo,
    pl,
    runs_root,
    saev,
    wandb,
):
    @beartype.beartype
    def _row_from_run(
        wandb_run,
    ) -> tuple[dict[str, object], list[dict[str, object]]] | None:
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

        metadata = row.get("config/train_data/metadata")
        assert metadata is not None, f"Run {wandb_run.id} missing metadata"

        row["model_key"] = get_model_key(metadata)
        row["data_key"] = get_data_key(metadata)

        cls_results = get_cls_results(saev_run)

        return row, cls_results

    @beartype.beartype
    def _finalize_sae_df(rows: list[dict[str, object]]):
        df = pl.DataFrame(rows, infer_schema_length=None)

        group_cols = (
            "model_key",
            "config/val_data/layer",
            "data_key",
            "config/sae/activation/key",
            "config/sae/activation/aux/key",
            "config/sae/reinit_blend",
            "config/optim",
        )

        df = (
            df.unnest("config/sae", "config/train_data/metadata", separator="/")
            .unnest("config/sae/activation", separator="/")
            .unnest(
                "config/sae/activation/aux",
                "config/sae/activation/sparsity",
                separator="/",
            )
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
    def _finalize_clf_df(rows: list[dict[str, object]]):
        df = pl.DataFrame(rows, infer_schema_length=None)
        df = (
            df.unnest("config/sae", "config/train_data/metadata", separator="/")
            .unnest("config/sae/activation", separator="/")
            .unnest(
                "config/sae/activation/aux",
                "config/sae/activation/sparsity",
                separator="/",
            )
            .unnest("cls/cfg", separator="/")
            .unnest("cls/cfg/cls", separator="/")
        )
        return df

    @beartype.beartype
    def _fetch_wandb_runs():
        all_runs = []
        for tag in WANDB_TAGS:
            filters = {"config.tag": tag}
            runs = list(
                wandb.Api().runs(
                    path=f"{WANDB_USERNAME}/{WANDB_PROJECT}", filters=filters
                )
            )
            all_runs.extend(runs)
        # Filter to only runs in our proposal-audit set
        filtered = [r for r in all_runs if r.id in ALL_RUN_IDS]
        if not filtered:
            raise ValueError(f"No runs found matching {ALL_RUN_IDS}.")
        return filtered

    @beartype.beartype
    def make_dfs_parallel(n_workers: int = 16):
        runs = _fetch_wandb_runs()

        sae_rows = []
        clf_rows = []
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

                row, cls_results = result
                sae_rows.append(row)
                for cls_result in cls_results:
                    clf_rows.append(
                        row | {f"cls/{key}": val for key, val in cls_result.items()}
                    )

        assert sae_rows, "No valid runs."
        sae_df = _finalize_sae_df(sae_rows)
        clf_df = _finalize_clf_df(clf_rows) if clf_rows else pl.DataFrame()
        return sae_df, clf_df

    sae_df, clf_df = make_dfs_parallel()
    return clf_df, sae_df


@app.cell
def _(sae_df):
    sae_df
    return


@app.cell
def _(clf_df):
    clf_df
    return


@app.cell
def _(clf_df):
    clf_df.columns
    return


@app.cell
def _(clf_df):
    clf_df.select("cls/cfg/cls/key").unique()
    return


@app.cell
def _(clf_df, pl):
    # ADE20K hypothesis testing: Yield@10 and Yield@30 analysis
    ade_df = clf_df.filter(
        (pl.col("cls/cfg/patch_agg") == "max")
        & (pl.col("data_key") == "IN1K/train")
        & pl.col("cls/audit/yield_at_10").is_not_null()
    ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

    # Extract key columns for analysis
    analysis_df = ade_df.select(
        pl.col("config/val_data/layer").alias("layer"),
        pl.col("config/sae/activation/top_k").alias("top_k"),
        pl.col("cls/cfg/cls/key").alias("clf_type"),
        pl.col("cls/cfg/cls/C").alias("C"),
        pl.col("cls/cfg/cls/max_depth").alias("max_depth"),
        pl.col("cls/n_nonzero").alias("n_nonzero"),
        pl.col("cls/macro_f1").alias("macro_f1"),
        pl.col("cls/audit/yield_at_3").alias("y3"),
        pl.col("cls/audit/yield_at_10").alias("y10"),
        pl.col("cls/audit/yield_at_30").alias("y30"),
        pl.col("cls/audit/yield_at_100").alias("y100"),
        pl.col("cls/audit/auc_b").alias("auc_b"),
    )
    analysis_df
    return (analysis_df,)


@app.cell
def _(analysis_df, jitter, mo, np, pl, plt):
    def _(df: pl.DataFrame):
        # Hypothesis 1: Does layer affect Yield? (Jitter plot)
        layers = sorted(df.get_column("layer").unique().to_list())
        n_per_layer = (
            df.group_by("layer").len().sort("layer").get_column("len").to_list()
        )

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150, layout="constrained")
        colors = {"y3": "C0", "y10": "C1", "y30": "C2"}
        markers = {"y3": "^", "y10": "o", "y30": "s"}

        fit_info = []
        for j, (metric, label) in enumerate([
            ("y3", "Yield@3"),
            ("y10", "Yield@10"),
            ("y30", "Yield@30"),
        ]):
            all_xs, all_ys = [], []
            for i, layer in enumerate(layers):
                ys = df.filter(pl.col("layer") == layer).get_column(metric).to_numpy()
                all_xs.extend([layer] * len(ys))
                all_ys.extend(ys)
                j_cat, j_data = jitter(len(ys))
                ax.scatter(
                    i + j_cat + (j - 1) * 0.3,
                    ys + j_data,
                    alpha=0.33,
                    c=colors[metric],
                    marker=markers[metric],
                    label=label if i == 0 else None,
                    clip_on=False,
                )
            # Linear fit
            all_xs, all_ys = np.array(all_xs), np.array(all_ys)
            slope, intercept = np.polyfit(all_xs, all_ys, 1)
            r_sq = np.corrcoef(all_xs, all_ys)[0, 1] ** 2
            x_fit = np.array([layers[0], layers[-1]])
            y_fit = slope * x_fit + intercept
            # Plot on index scale
            ax.plot(
                [0 + (j - 1) * 0.3, len(layers) - 1 + (j - 1) * 0.3],
                y_fit,
                c=colors[metric],
                linestyle="--",
                alpha=0.8,
            )
            fit_info.append(
                f"**{label}:** y = {slope:.4f}x + {intercept:.3f}, R^2 = {r_sq:.3f}"
            )

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{layer}" for layer in layers])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Yield")
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_title("Hypothesis 1: Layer effect on Yield")

        sample_str = ", ".join(
            f"L{layer}: n={n}" for layer, n in zip(layers, n_per_layer)
        )
        explanation = mo.md(
            f"**Samples per layer:** {sample_str}\n\n" + "\n\n".join(fit_info)
        )
        return mo.vstack([fig, explanation])

    _(analysis_df)
    return


@app.cell
def _(analysis_df, jitter, mo, pl, plt):
    def _(df: pl.DataFrame):
        # Hypothesis 2: Does classifier type matter? (Jitter plot)
        clf_types = sorted(df.get_column("clf_type").unique().to_list())
        n_per_clf = [df.filter(pl.col("clf_type") == c).height for c in clf_types]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150, layout="constrained")
        metrics = [("y3", "Yield@3"), ("y10", "Yield@10"), ("y30", "Yield@30")]

        for ax, (metric, label) in zip(axes, metrics):
            for i, clf in enumerate(clf_types):
                ys = df.filter(pl.col("clf_type") == clf).get_column(metric).to_numpy()
                j_cat, j_data = jitter(len(ys))
                ax.scatter(i + j_cat, ys + j_data, alpha=0.5, s=15, clip_on=False)
            ax.set_xticks(range(len(clf_types)))
            ax.set_xticklabels(clf_types, rotation=15, ha="right")
            ax.set_ylabel(label)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3, axis="y")
            ax.spines[["right", "top"]].set_visible(False)

        fig.suptitle("Hypothesis 2: Classifier type effect")
        sample_str = ", ".join(f"{c}: n={n}" for c, n in zip(clf_types, n_per_clf))
        explanation = mo.md(f"**Samples per classifier:** {sample_str}")
        return mo.vstack([fig, explanation])

    _(analysis_df)
    return


@app.cell
def _(analysis_df, np, pl, plt):
    def _(df: pl.DataFrame):
        # Hypothesis 3: Scatter of n_nonzero vs Yield with correlation and best fit
        xs = df.get_column("n_nonzero").to_numpy()
        log_xs = np.log10(xs)
        y3 = df.get_column("y3").to_numpy()
        y10 = df.get_column("y10").to_numpy()
        y30 = df.get_column("y30").to_numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150, layout="constrained")
        x_fit = np.geomspace(xs.min(), xs.max(), 100)
        log_x_fit = np.log10(x_fit)

        for ax, ys, label in zip(
            axes, [y3, y10, y30], ["Yield@3", "Yield@10", "Yield@30"]
        ):
            r = np.corrcoef(xs, ys)[0, 1]
            ax.scatter(xs, ys, alpha=0.5, s=20, clip_on=False)
            # Linear fit on log(x) vs y
            slope, intercept = np.polyfit(log_xs, ys, 1)
            ax.plot(x_fit, slope * log_x_fit + intercept, "r--", alpha=0.7, label="fit")
            ax.set_xscale("log")
            ax.set_xlabel("# Non-zero Features")
            ax.set_ylabel(label)
            ax.set_ylim(-0.1, 1.1)
            ax.set_title(f"r = {r:.3f}")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        fig.suptitle("Hypothesis 3: n_nonzero vs Yield")
        return fig

    _(analysis_df)
    return


@app.cell
def _(analysis_df, pl, plt):
    def _(df: pl.DataFrame):
        # Hypothesis 4: Layer x Classifier interaction - line chart
        by_layer_clf = (
            df
            .group_by("layer", "clf_type")
            .agg(
                pl.col("y10").mean().alias("y10_mean"),
                pl.col("y30").mean().alias("y30_mean"),
            )
            .sort("layer", "clf_type")
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, layout="constrained")
        for clf_type in ["decision-tree", "sparse-linear"]:
            subset = by_layer_clf.filter(pl.col("clf_type") == clf_type)
            layers = subset.get_column("layer").to_numpy()
            y10 = subset.get_column("y10_mean").to_numpy()
            y30 = subset.get_column("y30_mean").to_numpy()
            marker = "^" if clf_type == "decision-tree" else "o"
            axes[0].plot(layers, y10, marker=marker, label=clf_type)
            axes[1].plot(layers, y30, marker=marker, label=clf_type)
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Yield@10")
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].spines[["right", "top"]].set_visible(False)
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Yield@30")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].spines[["right", "top"]].set_visible(False)
        fig.suptitle("Hypothesis 4: Layer x Classifier interaction")
        return fig

    _(analysis_df)
    return


@app.cell
def _(analysis_df, jitter, mo, np, pl, plt):
    def _(df: pl.DataFrame):
        # Hypothesis 5: top_k effect by classifier type (Jitter plot)
        clf_types = ["decision-tree", "sparse-linear"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, layout="constrained")
        sample_info = []
        fit_info = []
        colors = {"y3": "C0", "y10": "C1", "y30": "C2"}
        markers = {"y3": "d", "y10": "o", "y30": "s"}

        for ax, clf_type in zip(axes, clf_types):
            clf_df = df.filter(pl.col("clf_type") == clf_type)
            topk_vals = sorted(clf_df.get_column("top_k").unique().to_list())
            n_per_k = [clf_df.filter(pl.col("top_k") == k).height for k in topk_vals]
            sample_info.append(
                f"**{clf_type}:** "
                + ", ".join(f"k={k}: n={n}" for k, n in zip(topk_vals, n_per_k))
            )

            clf_fit_info = []
            for j, (metric, label) in enumerate([
                ("y3", "Yield@3"),
                ("y10", "Yield@10"),
                ("y30", "Yield@30"),
            ]):
                all_xs, all_ys = [], []
                for i, k in enumerate(topk_vals):
                    ys = (
                        clf_df
                        .filter(pl.col("top_k") == k)
                        .get_column(metric)
                        .to_numpy()
                    )
                    all_xs.extend([k] * len(ys))
                    all_ys.extend(ys)
                    j_cat, j_data = jitter(len(ys))
                    ax.scatter(
                        i + j_cat + (j - 1) * 0.3,
                        ys + j_data,
                        alpha=0.6,
                        s=15,
                        c=colors[metric],
                        marker=markers[metric],
                        label=label if i == 0 else None,
                        clip_on=False,
                    )
                # Linear fit
                all_xs, all_ys = np.array(all_xs), np.array(all_ys)
                slope, intercept = np.polyfit(all_xs, all_ys, 1)
                r_sq = np.corrcoef(all_xs, all_ys)[0, 1] ** 2
                x_fit = np.array([topk_vals[0], topk_vals[-1]])
                y_fit = slope * x_fit + intercept
                ax.plot(
                    [0 + (j - 1) * 0.3, len(topk_vals) - 1 + (j - 1) * 0.3],
                    y_fit,
                    c=colors[metric],
                    linestyle="--",
                    alpha=0.8,
                )
                clf_fit_info.append(
                    f"{label}: y = {slope:.5f}x + {intercept:.3f}, R^2 = {r_sq:.3f}"
                )
            fit_info.append(f"**{clf_type}:** " + "; ".join(clf_fit_info))

            ax.set_xticks(range(len(topk_vals)))
            ax.set_xticklabels([str(k) for k in topk_vals])
            ax.set_xlabel("top_k")
            ax.set_ylabel("Yield")
            ax.set_ylim(-0.1, 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_title(clf_type)

        fig.suptitle("Hypothesis 5: top_k effect on Yield by Classifier Type")

        explanation = mo.md(f"""**Samples per top_k:**

    {sample_info[0]}

    {sample_info[1]}

    **Linear fits:**

    {fit_info[0]}

    {fit_info[1]}
    """)
        return mo.vstack([fig, explanation])

    _(analysis_df)
    return


@app.cell
def _(analysis_df, mo, np, pl, plt):
    def _(df: pl.DataFrame):
        # Yield vs Classification Accuracy (macro F1)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150, layout="constrained")

        f1 = df.get_column("macro_f1").to_numpy()
        clf_types = df.get_column("clf_type").to_numpy()
        colors = {"decision-tree": "C0", "sparse-linear": "C1"}
        stats = []  # Collect stats for text below

        for ax, (metric, label) in zip(
            axes, [("y3", "Yield@3"), ("y10", "Yield@10"), ("y30", "Yield@30")]
        ):
            ys = df.get_column(metric).to_numpy()
            metric_stats = []
            for clf_type in ["decision-tree", "sparse-linear"]:
                mask = clf_types == clf_type
                f1_sub, ys_sub = f1[mask], ys[mask]
                ax.scatter(
                    f1_sub,
                    ys_sub,
                    alpha=0.4,
                    s=20,
                    c=colors[clf_type],
                    label=clf_type,
                    clip_on=False,
                )
                # Per-classifier correlation and fit
                r = np.corrcoef(f1_sub, ys_sub)[0, 1]
                slope, intercept = np.polyfit(f1_sub, ys_sub, 1)
                x_fit = np.array([f1_sub.min(), f1_sub.max()])
                ax.plot(
                    x_fit,
                    slope * x_fit + intercept,
                    c=colors[clf_type],
                    linestyle="--",
                    alpha=0.7,
                )
                metric_stats.append((clf_type, slope, intercept, r))
            stats.append((label, metric_stats))
            ax.set_xlabel("Classification Macro F1")
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        fig.suptitle("ADE20K: Yield vs Classification Accuracy")

        # Build stats text
        lines = [
            "**Question:** Do classifiers that achieve higher F1 also select more grounded features?",
            "",
        ]
        for label, metric_stats in stats:
            lines.append(f"**{label}:**")
            for clf_type, slope, intercept, r in metric_stats:
                sign = "+" if intercept >= 0 else "-"
                lines.append(
                    f"- {clf_type}: y = {slope:.3f}x {sign} {abs(intercept):.3f}, R^2 = {r**2:.3f}"
                )
            lines.append("")
        lines.append(
            "A positive correlation suggests **classification and groundedness are aligned**. A weak/negative correlation suggests a **tradeoff**."
        )
        explanation = mo.md("\n".join(lines))
        return mo.vstack([fig, explanation])

    _(analysis_df)
    return


@app.cell
def _(clf_df, pl):
    # FishVista hypothesis testing: Yield@10 and Yield@30 analysis
    fv_df = clf_df.filter(
        (pl.col("cls/cfg/patch_agg") == "max")
        & (pl.col("data_key") == "FishVista (Img)")
        & pl.col("cls/audit/yield_at_10").is_not_null()
    ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

    # Extract key columns for analysis
    fv_analysis_df = fv_df.select(
        pl.col("config/val_data/layer").alias("layer"),
        pl.col("config/sae/activation/top_k").alias("top_k"),
        pl.col("cls/cfg/cls/key").alias("clf_type"),
        pl.col("cls/cfg/cls/C").alias("C"),
        pl.col("cls/cfg/cls/max_depth").alias("max_depth"),
        pl.col("cls/n_nonzero").alias("n_nonzero"),
        pl.col("cls/audit/yield_at_3").alias("y3"),
        pl.col("cls/audit/yield_at_10").alias("y10"),
        pl.col("cls/audit/yield_at_30").alias("y30"),
        pl.col("cls/audit/yield_at_100").alias("y100"),
        pl.col("cls/audit/auc_b").alias("auc_b"),
    )
    fv_analysis_df
    return (fv_analysis_df,)


@app.cell
def _(fv_analysis_df, jitter, mo, np, pl, plt):
    def _(df: pl.DataFrame):
        # FishVista Hypothesis 1: Does layer affect Yield? (Jitter plot)
        layers = sorted(df.get_column("layer").unique().to_list())
        n_per_layer = (
            df.group_by("layer").len().sort("layer").get_column("len").to_list()
        )

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150, layout="constrained")
        colors = {"y3": "C0", "y10": "C1", "y30": "C2"}
        markers = {"y3": "^", "y10": "o", "y30": "s"}

        fit_info = []
        for j, (metric, label) in enumerate([
            ("y3", "Yield@3"),
            ("y10", "Yield@10"),
            ("y30", "Yield@30"),
        ]):
            all_xs, all_ys = [], []
            for i, layer in enumerate(layers):
                ys = df.filter(pl.col("layer") == layer).get_column(metric).to_numpy()
                all_xs.extend([layer] * len(ys))
                all_ys.extend(ys)
                j_cat, j_data = jitter(len(ys))
                ax.scatter(
                    i + j_cat + (j - 1) * 0.3,
                    ys + j_data,
                    alpha=0.33,
                    c=colors[metric],
                    marker=markers[metric],
                    label=label if i == 0 else None,
                    clip_on=False,
                )
            # Linear fit
            all_xs, all_ys = np.array(all_xs), np.array(all_ys)
            slope, intercept = np.polyfit(all_xs, all_ys, 1)
            r_sq = np.corrcoef(all_xs, all_ys)[0, 1] ** 2
            x_fit = np.array([layers[0], layers[-1]])
            y_fit = slope * x_fit + intercept
            ax.plot(
                [0 + (j - 1) * 0.3, len(layers) - 1 + (j - 1) * 0.3],
                y_fit,
                c=colors[metric],
                linestyle="--",
                alpha=0.8,
            )
            fit_info.append(
                f"**{label}:** y = {slope:.4f}x + {intercept:.3f}, R^2 = {r_sq:.3f}"
            )

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{layer}" for layer in layers])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Yield")
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_title("FishVista: Layer effect on Yield")

        sample_str = ", ".join(
            f"L{layer}: n={n}" for layer, n in zip(layers, n_per_layer)
        )
        explanation = mo.md(
            f"**Samples per layer:** {sample_str}\n\n" + "\n\n".join(fit_info)
        )
        return mo.vstack([fig, explanation])

    _(fv_analysis_df)
    return


@app.cell
def _(fv_analysis_df, jitter, mo, np, pl, plt):
    def _(df: pl.DataFrame):
        # FishVista Hypothesis 5: top_k effect by classifier type (Jitter plot)
        clf_types = ["decision-tree", "sparse-linear"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, layout="constrained")
        sample_info = []
        fit_info = []
        colors = {"y3": "C0", "y10": "C1", "y30": "C2"}
        markers = {"y3": "d", "y10": "o", "y30": "s"}

        for ax, clf_type in zip(axes, clf_types):
            clf_df = df.filter(pl.col("clf_type") == clf_type)
            topk_vals = sorted(clf_df.get_column("top_k").unique().to_list())
            n_per_k = [clf_df.filter(pl.col("top_k") == k).height for k in topk_vals]
            sample_info.append(
                f"**{clf_type}:** "
                + ", ".join(f"k={k}: n={n}" for k, n in zip(topk_vals, n_per_k))
            )

            clf_fit_info = []
            for j, (metric, label) in enumerate([
                ("y3", "Yield@3"),
                ("y10", "Yield@10"),
                ("y30", "Yield@30"),
            ]):
                all_xs, all_ys = [], []
                for i, k in enumerate(topk_vals):
                    ys = (
                        clf_df
                        .filter(pl.col("top_k") == k)
                        .get_column(metric)
                        .to_numpy()
                    )
                    all_xs.extend([k] * len(ys))
                    all_ys.extend(ys)
                    j_cat, j_data = jitter(len(ys))
                    ax.scatter(
                        i + j_cat + (j - 1) * 0.3,
                        ys + j_data,
                        alpha=0.6,
                        s=15,
                        c=colors[metric],
                        marker=markers[metric],
                        label=label if i == 0 else None,
                        clip_on=False,
                    )
                # Linear fit
                all_xs, all_ys = np.array(all_xs), np.array(all_ys)
                slope, intercept = np.polyfit(all_xs, all_ys, 1)
                r_sq = np.corrcoef(all_xs, all_ys)[0, 1] ** 2
                x_fit = np.array([topk_vals[0], topk_vals[-1]])
                y_fit = slope * x_fit + intercept
                ax.plot(
                    [0 + (j - 1) * 0.3, len(topk_vals) - 1 + (j - 1) * 0.3],
                    y_fit,
                    c=colors[metric],
                    linestyle="--",
                    alpha=0.8,
                )
                clf_fit_info.append(
                    f"{label}: y = {slope:.5f}x + {intercept:.3f}, R^2 = {r_sq:.3f}"
                )
            fit_info.append(f"**{clf_type}:** " + "; ".join(clf_fit_info))

            ax.set_xticks(range(len(topk_vals)))
            ax.set_xticklabels([str(k) for k in topk_vals])
            ax.set_xlabel("top_k")
            ax.set_ylabel("Yield")
            ax.set_ylim(-0.1, 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_title(clf_type)

        fig.suptitle("FishVista: top_k effect on Yield by Classifier Type")

        explanation = mo.md(f"""**Samples per top_k:**

    {sample_info[0]}

    {sample_info[1]}

    **Linear fits:**

    {fit_info[0]}

    {fit_info[1]}
    """)
        return mo.vstack([fig, explanation])

    _(fv_analysis_df)
    return


@app.cell
def _(analysis_df, pl, plt):
    def _(df: pl.DataFrame):
        # Hypothesis 6: SparseLinear C effect - line chart by layer, colored by C
        sparse_linear = df.filter(pl.col("clf_type") == "sparse-linear")
        # Aggregate by (C, layer) across all top_k values
        by_c_layer = (
            sparse_linear
            .group_by("C", "layer")
            .agg(
                pl.col("y10").mean().alias("y10_mean"),
                pl.col("y10").max().alias("y10_max"),
                pl.col("y30").mean().alias("y30_mean"),
                pl.col("y30").max().alias("y30_max"),
            )
            .sort("layer", "C")
        )
        c_values = sorted(by_c_layer.get_column("C").unique().to_list())
        cmap = plt.cm.viridis

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, layout="constrained")
        for i, c in enumerate(c_values):
            subset = by_c_layer.filter(pl.col("C") == c)
            layers = subset.get_column("layer").to_numpy()
            color = cmap(i / (len(c_values) - 1)) if len(c_values) > 1 else cmap(0.5)
            # Mean (solid line)
            axes[0].plot(
                layers,
                subset.get_column("y10_mean").to_numpy(),
                marker="o",
                color=color,
                label=f"C={c} mean",
            )
            axes[1].plot(
                layers,
                subset.get_column("y30_mean").to_numpy(),
                marker="o",
                color=color,
                label=f"C={c} mean",
            )
            # Max (dashed line)
            axes[0].plot(
                layers,
                subset.get_column("y10_max").to_numpy(),
                marker="s",
                linestyle="--",
                color=color,
                label=f"C={c} max",
            )
            axes[1].plot(
                layers,
                subset.get_column("y30_max").to_numpy(),
                marker="s",
                linestyle="--",
                color=color,
                label=f"C={c} max",
            )
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Yield@10")
        axes[0].set_ylim(0, 1)
        axes[0].legend(fontsize=6, ncol=2)
        axes[0].grid(True, alpha=0.3)
        axes[0].spines[["right", "top"]].set_visible(False)
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Yield@30")
        axes[1].set_ylim(0, 1)
        axes[1].legend(fontsize=6, ncol=2)
        axes[1].grid(True, alpha=0.3)
        axes[1].spines[["right", "top"]].set_visible(False)
        fig.suptitle("Hypothesis 6: SparseLinear C x Layer (mean & max across top_k)")
        return fig

    _(analysis_df)
    return


@app.cell
def _(analysis_df, np, pl, plt):
    def _(df: pl.DataFrame):
        # Hypothesis 6b: top_k x C interaction for SparseLinear
        sparse_linear = df.filter(pl.col("clf_type") == "sparse-linear")
        # Aggregate by (top_k, C) across all layers
        by_topk_c = (
            sparse_linear
            .group_by("top_k", "C")
            .agg(
                pl.col("y10").mean().alias("y10_mean"),
                pl.col("y10").max().alias("y10_max"),
                pl.col("y30").mean().alias("y30_mean"),
                pl.col("y30").max().alias("y30_max"),
            )
            .sort("top_k", "C")
        )
        c_values = sorted(by_topk_c.get_column("C").unique().to_list())
        cmap = plt.cm.viridis

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, layout="constrained")
        for i, c in enumerate(c_values):
            subset = by_topk_c.filter(pl.col("C") == c)
            top_ks = subset.get_column("top_k").to_numpy()
            color = cmap(i / (len(c_values) - 1)) if len(c_values) > 1 else cmap(0.5)
            # Mean (solid) and max (dashed)
            axes[0].plot(
                top_ks,
                subset.get_column("y10_mean").to_numpy(),
                marker="o",
                color=color,
                label=f"C={c} mean",
            )
            axes[0].plot(
                top_ks,
                subset.get_column("y10_max").to_numpy(),
                marker="s",
                linestyle="--",
                color=color,
                label=f"C={c} max",
            )
            axes[1].plot(
                top_ks,
                subset.get_column("y30_mean").to_numpy(),
                marker="o",
                color=color,
                label=f"C={c} mean",
            )
            axes[1].plot(
                top_ks,
                subset.get_column("y30_max").to_numpy(),
                marker="s",
                linestyle="--",
                color=color,
                label=f"C={c} max",
            )
        for ax, ylabel in zip(axes, ["Yield@10", "Yield@30"]):
            ax.set_xlabel("top_k (SAE sparsity)")
            ax.set_ylabel(ylabel)
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)
        fig.suptitle(
            "Hypothesis 6b: top_k x C interaction (SparseLinear, across layers)"
        )
        return fig

    _(analysis_df)
    return


@app.cell
def _(analysis_df, pl, plt):
    def _(df: pl.DataFrame):
        # Hypothesis 7: DecisionTree depth effect - line chart by layer, colored by depth
        decision_tree = df.filter(pl.col("clf_type") == "decision-tree")
        by_depth_layer = (
            decision_tree
            .group_by("max_depth", "layer")
            .agg(
                pl.col("y10").mean().alias("y10_mean"),
                pl.col("y30").mean().alias("y30_mean"),
            )
            .sort("layer", "max_depth")
        )
        depths = sorted(by_depth_layer.get_column("max_depth").unique().to_list())
        cmap = plt.cm.coolwarm

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, layout="constrained")
        for i, d in enumerate(depths):
            subset = by_depth_layer.filter(pl.col("max_depth") == d)
            layers = subset.get_column("layer").to_numpy()
            y10 = subset.get_column("y10_mean").to_numpy()
            y30 = subset.get_column("y30_mean").to_numpy()
            color = cmap(i / (len(depths) - 1)) if len(depths) > 1 else cmap(0.5)
            label = "unlimited" if d == -1 else str(d)
            axes[0].plot(layers, y10, marker="^", color=color, label=f"d={label}")
            axes[1].plot(layers, y30, marker="^", color=color, label=f"d={label}")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Yield@10")
        axes[0].set_ylim(0, 1)
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)
        axes[0].spines[["right", "top"]].set_visible(False)
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Yield@30")
        axes[1].set_ylim(0, 1)
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)
        axes[1].spines[["right", "top"]].set_visible(False)
        fig.suptitle("Hypothesis 7: DecisionTree depth x Layer")
        return fig

    _(analysis_df)
    return


@app.cell
def _(analysis_df, mo):
    # Hypothesis 8: Best configs - what maximizes y10?
    best_y10 = (
        analysis_df
        .sort("y10", descending=True)
        .head(20)
        .select(
            "layer",
            "top_k",
            "clf_type",
            "C",
            "max_depth",
            "n_nonzero",
            "y10",
            "y30",
            "auc_b",
        )
    )
    mo.vstack([mo.md("## Top 20 configs by Yield@10"), best_y10])
    return


@app.cell
def _(analysis_df, np, pl, plt):
    def _(df: pl.DataFrame):
        # Hypothesis 9: Correlation heatmap
        corr_data = df.select("n_nonzero", "y10", "y30", "auc_b").to_numpy()
        corr_matrix = np.corrcoef(corr_data.T)
        labels = ["n_nonzero", "y10", "y30", "auc_b"]

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150, layout="constrained")
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        fig.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Hypothesis 9: Correlation matrix")
        return fig

    _(analysis_df)
    return


@app.cell
def _(clf_df, mo, np, pl, plt, saev, scipy):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/macro_f1"
        k_col = "config/sae/activation/top_k"

        layers = [13, 15, 17, 19, 21, 23]

        # Sigmoid function on log-x scale: y = L / (1 + exp(-k*(log(x) - x0))) + b
        def sigmoid(x, L, k, x0, b):
            return L / (1 + np.exp(-k * (np.log(x) - x0))) + b

        # Filter to ADE20K scene_top50 task only
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "IN1K/train")  # IN1K SAEs evaluated on ADE20K
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        filtered = filtered.filter(pl.col("task_name") == "scene_top50")

        # Get n_classes for random chance baseline
        n_classes = filtered.get_column("cls/n_classes").unique().item()
        random_chance = 1 / n_classes

        # Table of individual points
        table = (
            filtered.select(
                "config/val_data/layer",
                "cls/cfg/cls/key",
                k_col,
                "cls/n_nonzero",
                "cls/macro_f1",
                "cls/test_acc",
                "cls/balanced_acc",
            )
            .sort("config/val_data/layer", "cls/cfg/cls/key", "cls/n_nonzero")
            .with_columns(pl.lit(random_chance).alias("random_chance"))
        )

        # Colormap based on log2(top_k)
        all_k = filtered.get_column(k_col).to_numpy()
        log2_k = np.log2(all_k)
        k_min, k_max = log2_k.min(), log2_k.max()
        cmap = plt.cm.plasma

        # Define classifier markers
        clf_markers = {"sparse-linear": "o", "decision-tree": "^"}

        # Global x range across all layers for sigmoid fit line
        global_x_min = filtered.get_column(x_col).min()
        global_x_max = filtered.get_column(x_col).max()
        x_fit = np.geomspace(global_x_min, global_x_max, 100)

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

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            # Get all points for this layer (both classifiers) for sigmoid fit
            layer_group = filtered.filter(pl.col("config/val_data/layer") == layer)
            all_xs = layer_group.get_column(x_col).to_numpy()
            all_ys = layer_group.get_column(y_col).to_numpy()

            for clf_key, marker in clf_markers.items():
                group = filtered.filter(
                    (pl.col("config/val_data/layer") == layer)
                    & (pl.col("cls/cfg/cls/key") == clf_key)
                )

                if group.height == 0:
                    continue

                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ks = group.get_column(k_col).to_numpy()
                colors = cmap((np.log2(ks) - k_min) / (k_max - k_min))

                ax.scatter(xs, ys, alpha=0.7, c=colors, marker=marker)

            # Fit sigmoid to all points (both classifiers)
            if len(all_xs) >= 4:
                try:
                    y_range = all_ys.max() - all_ys.min()
                    p0 = [y_range, 1.0, np.log(np.median(all_xs)), all_ys.min()]
                    bounds = ([0, 0.01, -10, 0], [1, 10, 20, 1])
                    popt, _ = scipy.optimize.curve_fit(
                        sigmoid, all_xs, all_ys, p0=p0, bounds=bounds, maxfev=5000
                    )
                    y_fit = sigmoid(x_fit, *popt)
                    ax.plot(
                        x_fit,
                        y_fit,
                        color=saev.colors.BLACK_RGB01,
                        linestyle="--",
                        alpha=0.5,
                    )
                except Exception:
                    pass  # Skip if fit fails

            ax.axhline(
                random_chance,
                color="red",
                linestyle="--",
                alpha=0.5,
                label="Random",
            )

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.set_title(f"Layer {layer + 1}/24")

            if i in (3, 4, 5):
                ax.set_xlabel("# Non-zero Features")

            if i in (0, 3):
                ax.set_ylabel("Macro F1")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(k_min, k_max))
        cbar = fig.colorbar(sm, ax=axes)
        cbar.set_label("log2(k)")

        # Add legend for marker shapes
        axes[2].scatter([], [], marker="o", c="gray", alpha=0.5, label="SparseLinear")
        axes[2].scatter([], [], marker="^", c="gray", alpha=0.5, label="DecisionTree")
        axes[2].legend(loc="upper left")

        fig.suptitle("ADE20K Scene Classification (Top 50)")
        return mo.vstack([table, fig])

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, np, pl, plt):
    def _(df: pl.DataFrame):
        """ADE20K: AUC_B by layer (overall scatter plot)"""
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/auc_b"
        k_col = "config/sae/activation/top_k"

        layers = [13, 15, 17, 19, 21, 23]

        # Filter to ADE20K with audit results
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "IN1K/train")
            & pl.col(y_col).is_not_null()
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No ADE20K audit results available yet.")

        # Table of individual points
        table = filtered.select(
            "config/val_data/layer",
            "task_name",
            "cls/cfg/cls/key",
            k_col,
            x_col,
            y_col,
            "cls/audit/yield_at_3",
            "cls/audit/yield_at_10",
            "cls/audit/yield_at_100",
        ).sort("config/val_data/layer", "task_name", "cls/cfg/cls/key", x_col)

        # Colormap based on log2(top_k)
        all_k = filtered.get_column(k_col).to_numpy()
        log2_k = np.log2(all_k)
        k_min, k_max = log2_k.min(), log2_k.max()
        cmap = plt.cm.plasma

        clf_markers = {"sparse-linear": "o", "decision-tree": "^"}

        global_y_max = filtered.get_column(y_col).max()
        y_lim_max = np.ceil(global_y_max * 11) / 10

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

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            for clf_key, marker in clf_markers.items():
                group = filtered.filter(
                    (pl.col("config/val_data/layer") == layer)
                    & (pl.col("cls/cfg/cls/key") == clf_key)
                )
                if group.height == 0:
                    continue

                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ks = group.get_column(k_col).to_numpy()
                colors = cmap((np.log2(ks) - k_min) / (k_max - k_min))
                ax.scatter(xs, ys, alpha=0.7, c=colors, marker=marker)

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_ylim(0, y_lim_max)
            ax.set_title(f"Layer {layer + 1}/24")

            if i in (3, 4, 5):
                ax.set_xlabel("# Non-zero Features")
            if i in (0, 3):
                ax.set_ylabel("AUC_B")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(k_min, k_max))
        cbar = fig.colorbar(sm, ax=axes)
        cbar.set_label("log2(k)")

        axes[2].scatter([], [], marker="o", c="gray", alpha=0.5, label="SparseLinear")
        axes[2].scatter([], [], marker="^", c="gray", alpha=0.5, label="DecisionTree")
        axes[2].legend(loc="upper left")

        fig.suptitle("ADE20K: Feature Grounding (AUC over Yield@B)")
        return mo.vstack([table, fig])

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        """ADE20K: Yield@3 by layer"""
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_3"

        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "IN1K/train")
            & pl.col(y_col).is_not_null()
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No ADE20K audit results available yet.")

        layers = [13, 15, 17, 19, 21, 23]

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for clf_key in ["sparse-linear", "decision-tree"]:
                group = layer_df.filter(pl.col("cls/cfg/cls/key") == clf_key)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                marker = "o" if clf_key == "sparse-linear" else "^"
                ax.scatter(
                    xs, ys, alpha=0.6, marker=marker, label=clf_key if i == 0 else None
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@3")
        fig.suptitle("ADE20K: Yield@3 by Layer")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        """ADE20K: Yield@10 by layer"""
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_10"

        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "IN1K/train")
            & pl.col(y_col).is_not_null()
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No ADE20K audit results available yet.")

        layers = [13, 15, 17, 19, 21, 23]

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for clf_key in ["sparse-linear", "decision-tree"]:
                group = layer_df.filter(pl.col("cls/cfg/cls/key") == clf_key)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                marker = "o" if clf_key == "sparse-linear" else "^"
                ax.scatter(
                    xs, ys, alpha=0.6, marker=marker, label=clf_key if i == 0 else None
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@10")
        fig.suptitle("ADE20K: Yield@10 by Layer")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        """ADE20K SparseLinear: Yield@3 by Regularization (C)"""
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_3"

        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "IN1K/train")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "sparse-linear")
        )

        if filtered.height == 0:
            return mo.md("No ADE20K sparse-linear audit results available yet.")

        c_values = sorted(filtered.get_column("cls/cfg/cls/C").unique().to_list())
        cmap = plt.cm.viridis

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        c_colors = {c: cmap(i / (len(c_values) - 1)) for i, c in enumerate(c_values)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for c in c_values:
                group = layer_df.filter(pl.col("cls/cfg/cls/C") == c)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[c_colors[c]],
                    label=f"C={c}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@3")
        fig.suptitle("ADE20K SparseLinear: Yield@3 by Regularization (C)")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        """ADE20K DecisionTree: Yield@3 by Max Depth"""
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_3"

        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "IN1K/train")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "decision-tree")
        )

        if filtered.height == 0:
            return mo.md("No ADE20K decision-tree audit results available yet.")

        depths = sorted(filtered.get_column("cls/cfg/cls/max_depth").unique().to_list())
        cmap = plt.cm.coolwarm

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        depth_colors = {d: cmap(i / (len(depths) - 1)) for i, d in enumerate(depths)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for d in depths:
                group = layer_df.filter(pl.col("cls/cfg/cls/max_depth") == d)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                label_d = "unlimited" if d == -1 else str(d)
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[depth_colors[d]],
                    label=f"d={label_d}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=7)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@3")
        fig.suptitle("ADE20K DecisionTree: Yield@3 by Max Depth")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        """ADE20K SparseLinear: Yield@10 by Regularization (C)"""
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_10"

        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "IN1K/train")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "sparse-linear")
        )

        if filtered.height == 0:
            return mo.md("No ADE20K sparse-linear audit results available yet.")

        c_values = sorted(filtered.get_column("cls/cfg/cls/C").unique().to_list())
        cmap = plt.cm.viridis

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        c_colors = {c: cmap(i / (len(c_values) - 1)) for i, c in enumerate(c_values)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for c in c_values:
                group = layer_df.filter(pl.col("cls/cfg/cls/C") == c)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[c_colors[c]],
                    label=f"C={c}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@10")
        fig.suptitle("ADE20K SparseLinear: Yield@10 by Regularization (C)")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        """ADE20K DecisionTree: Yield@10 by Max Depth"""
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_10"

        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "IN1K/train")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "decision-tree")
        )

        if filtered.height == 0:
            return mo.md("No ADE20K decision-tree audit results available yet.")

        depths = sorted(filtered.get_column("cls/cfg/cls/max_depth").unique().to_list())
        cmap = plt.cm.coolwarm

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        depth_colors = {d: cmap(i / (len(depths) - 1)) for i, d in enumerate(depths)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for d in depths:
                group = layer_df.filter(pl.col("cls/cfg/cls/max_depth") == d)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                label_d = "unlimited" if d == -1 else str(d)
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[depth_colors[d]],
                    label=f"d={label_d}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=7)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@10")
        fig.suptitle("ADE20K DecisionTree: Yield@10 by Max Depth")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        """ADE20K SparseLinear: AUC_B by Regularization (C)"""
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/auc_b"

        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "IN1K/train")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "sparse-linear")
        )

        if filtered.height == 0:
            return mo.md("No ADE20K sparse-linear audit results available yet.")

        c_values = sorted(filtered.get_column("cls/cfg/cls/C").unique().to_list())
        cmap = plt.cm.viridis

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        c_colors = {c: cmap(i / (len(c_values) - 1)) for i, c in enumerate(c_values)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for c in c_values:
                group = layer_df.filter(pl.col("cls/cfg/cls/C") == c)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[c_colors[c]],
                    label=f"C={c}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("AUC_B")
        fig.suptitle("ADE20K SparseLinear: AUC_B by Regularization (C)")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        """ADE20K DecisionTree: AUC_B by Max Depth"""
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/auc_b"

        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "IN1K/train")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "decision-tree")
        )

        if filtered.height == 0:
            return mo.md("No ADE20K decision-tree audit results available yet.")

        depths = sorted(filtered.get_column("cls/cfg/cls/max_depth").unique().to_list())
        cmap = plt.cm.coolwarm

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        depth_colors = {d: cmap(i / (len(depths) - 1)) for i, d in enumerate(depths)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for d in depths:
                group = layer_df.filter(pl.col("cls/cfg/cls/max_depth") == d)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                label_d = "unlimited" if d == -1 else str(d)
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[depth_colors[d]],
                    label=f"d={label_d}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=7)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("AUC_B")
        fig.suptitle("ADE20K DecisionTree: AUC_B by Max Depth")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, np, pl, plt, saev, scipy):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/macro_f1"
        k_col = "config/sae/activation/top_k"

        layers = [13, 15, 17, 19, 21, 23]

        # Sigmoid function on log-x scale: y = L / (1 + exp(-k*(log(x) - x0))) + b
        def sigmoid(x, L, k, x0, b):
            return L / (1 + np.exp(-k * (np.log(x) - x0))) + b

        # Filter to FishVista habitat task only
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        filtered = filtered.filter(pl.col("task_name") == "habitat")

        # Get n_classes for random chance baseline
        n_classes = filtered.get_column("cls/n_classes").unique().item()
        random_chance = 1 / n_classes

        # Table of individual points
        table = (
            filtered.select(
                "config/val_data/layer",
                "cls/cfg/cls/key",
                k_col,
                "cls/n_nonzero",
                "cls/macro_f1",
                "cls/test_acc",
                "cls/balanced_acc",
            )
            .sort("config/val_data/layer", "cls/cfg/cls/key", "cls/n_nonzero")
            .with_columns(pl.lit(random_chance).alias("random_chance"))
        )

        # Colormap based on log2(top_k)
        all_k = filtered.get_column(k_col).to_numpy()
        log2_k = np.log2(all_k)
        k_min, k_max = log2_k.min(), log2_k.max()
        cmap = plt.cm.plasma

        # Define classifier markers
        clf_markers = {"sparse-linear": "o", "decision-tree": "^"}

        # Global x range across all layers for sigmoid fit line
        global_x_min = filtered.get_column(x_col).min()
        global_x_max = filtered.get_column(x_col).max()
        x_fit = np.geomspace(global_x_min, global_x_max, 100)

        # Y limit: round up to next 0.1
        global_y_max = filtered.get_column(y_col).max()
        y_lim_max = np.ceil(global_y_max * 10) / 10

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

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            # Get all points for this layer (both classifiers) for sigmoid fit
            layer_group = filtered.filter(pl.col("config/val_data/layer") == layer)
            all_xs = layer_group.get_column(x_col).to_numpy()
            all_ys = layer_group.get_column(y_col).to_numpy()

            for clf_key, marker in clf_markers.items():
                group = filtered.filter(
                    (pl.col("config/val_data/layer") == layer)
                    & (pl.col("cls/cfg/cls/key") == clf_key)
                )

                if group.height == 0:
                    continue

                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ks = group.get_column(k_col).to_numpy()
                colors = cmap((np.log2(ks) - k_min) / (k_max - k_min))

                ax.scatter(xs, ys, alpha=0.7, c=colors, marker=marker)

            # Fit sigmoid to all points (both classifiers)
            if len(all_xs) >= 4:
                try:
                    y_range = all_ys.max() - all_ys.min()
                    p0 = [y_range, 1.0, np.log(np.median(all_xs)), all_ys.min()]
                    bounds = ([0, 0.01, -10, 0], [1, 10, 20, 1])
                    popt, _ = scipy.optimize.curve_fit(
                        sigmoid, all_xs, all_ys, p0=p0, bounds=bounds, maxfev=5000
                    )
                    y_fit = sigmoid(x_fit, *popt)
                    ax.plot(
                        x_fit,
                        y_fit,
                        color=saev.colors.BLACK_RGB01,
                        linestyle="--",
                        alpha=0.5,
                    )
                except Exception:
                    pass  # Skip if fit fails

            ax.axhline(
                random_chance,
                color="red",
                linestyle="--",
                alpha=0.5,
                label="Random",
            )

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_ylim(0, y_lim_max)
            ax.set_title(f"Layer {layer + 1}/24")

            if i in (3, 4, 5):
                ax.set_xlabel("# Non-zero Features")

            if i in (0, 3):
                ax.set_ylabel("Macro F1")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(k_min, k_max))
        cbar = fig.colorbar(sm, ax=axes)
        cbar.set_label("log2(k)")

        # Add legend for marker shapes
        axes[2].scatter([], [], marker="o", c="gray", alpha=0.5, label="SparseLinear")
        axes[2].scatter([], [], marker="^", c="gray", alpha=0.5, label="DecisionTree")
        axes[2].legend(loc="upper left")

        fig.suptitle("FishVista Habitat")
        return mo.vstack([table, fig])

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/auc_b"

        # Filter to FishVista with audit results
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No audit results available yet.")

        # Normalize auc_b to [0,1] by dividing by 6

        tasks = sorted(filtered.get_column("task_name").unique().to_list())
        cmap = plt.cm.tab10

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        # Plot by task (rows) x classifier type (columns would be too many, so color by clf type)
        for i, task in enumerate(tasks):
            ax = axes[i]
            task_df = filtered.filter(pl.col("task_name") == task)

            for j, clf_key in enumerate(["sparse-linear", "decision-tree"]):
                group = task_df.filter(pl.col("cls/cfg/cls/key") == clf_key)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                marker = "o" if clf_key == "sparse-linear" else "^"
                ax.scatter(
                    xs, ys, alpha=0.6, marker=marker, label=clf_key if i == 0 else None
                )

            ax.set_xscale("log")
            ax.set_title(task.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        # Hide unused axes
        for i in range(len(tasks), len(axes)):
            axes[i].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("AUC_B")
        fig.suptitle("FishVista: AUC_B by Task")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/auc_b"
        k_col = "config/sae/activation/top_k"

        # Filter to FishVista with audit results
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No audit results available yet.")

        # Normalize auc_b

        top_ks = sorted(filtered.get_column(k_col).unique().to_list())
        cmap = plt.cm.plasma

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        # Plot by top_k, color by layer
        layers = sorted(filtered.get_column("config/val_data/layer").unique().to_list())
        layer_colors = {l: cmap(i / (len(layers) - 1)) for i, l in enumerate(layers)}

        for i, k in enumerate(top_ks):
            ax = axes[i]
            k_df = filtered.filter(pl.col(k_col) == k)

            for layer in layers:
                group = k_df.filter(pl.col("config/val_data/layer") == layer)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.6,
                    c=[layer_colors[layer]],
                    label=f"L{layer + 1}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_title(f"top_k = {k}")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        for i in range(len(top_ks), len(axes)):
            axes[i].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=7, ncol=2)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("AUC_B")
        fig.suptitle("FishVista: AUC_B by Top-K (colored by layer)")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/auc_b"

        # Filter to FishVista with audit results, sparse-linear only
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "sparse-linear")
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No sparse-linear audit results available yet.")

        # Get C values
        c_values = sorted(filtered.get_column("cls/cfg/cls/C").unique().to_list())
        cmap = plt.cm.viridis

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        c_colors = {c: cmap(i / (len(c_values) - 1)) for i, c in enumerate(c_values)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for c in c_values:
                group = layer_df.filter(pl.col("cls/cfg/cls/C") == c)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[c_colors[c]],
                    label=f"C={c}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("AUC_B")
        fig.suptitle("SparseLinear: AUC_B by Regularization (C)")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/auc_b"

        # Filter to FishVista with audit results, decision-tree only
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "decision-tree")
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No decision-tree audit results available yet.")

        # Get max_depth values
        depths = sorted(filtered.get_column("cls/cfg/cls/max_depth").unique().to_list())
        cmap = plt.cm.coolwarm

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        depth_colors = {d: cmap(i / (len(depths) - 1)) for i, d in enumerate(depths)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for d in depths:
                group = layer_df.filter(pl.col("cls/cfg/cls/max_depth") == d)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                label_d = "unlimited" if d == -1 else str(d)
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[depth_colors[d]],
                    label=f"d={label_d}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=7)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("AUC_B")
        fig.suptitle("DecisionTree: AUC_B by Max Depth")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, np, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/auc_b"
        k_col = "config/sae/activation/top_k"

        layers = [13, 15, 17, 19, 21, 23]

        # Filter to FishVista with audit results
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No audit results available yet.")

        # Table of individual points
        table = filtered.select(
            "config/val_data/layer",
            "task_name",
            "cls/cfg/cls/key",
            k_col,
            x_col,
            y_col,
            "cls/audit/yield_at_3",
            "cls/audit/yield_at_10",
            "cls/audit/yield_at_100",
        ).sort("config/val_data/layer", "task_name", "cls/cfg/cls/key", x_col)

        # Colormap based on log2(top_k)
        all_k = filtered.get_column(k_col).to_numpy()
        log2_k = np.log2(all_k)
        k_min, k_max = log2_k.min(), log2_k.max()
        cmap = plt.cm.plasma

        # Define classifier markers
        clf_markers = {"sparse-linear": "o", "decision-tree": "^"}

        # Y limit: round up to next 0.5
        global_y_max = filtered.get_column(y_col).max()
        y_lim_max = min(np.ceil(global_y_max * 2) / 2, 6.0)

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

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            for clf_key, marker in clf_markers.items():
                group = filtered.filter(
                    (pl.col("config/val_data/layer") == layer)
                    & (pl.col("cls/cfg/cls/key") == clf_key)
                )

                if group.height == 0:
                    continue

                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ks = group.get_column(k_col).to_numpy()
                colors = cmap((np.log2(ks) - k_min) / (k_max - k_min))

                ax.scatter(xs, ys, alpha=0.7, c=colors, marker=marker)

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_ylim(0, y_lim_max)
            ax.set_title(f"Layer {layer + 1}/24")

            if i in (3, 4, 5):
                ax.set_xlabel("# Non-zero Features")

            if i in (0, 3):
                ax.set_ylabel("AUC_B")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(k_min, k_max))
        cbar = fig.colorbar(sm, ax=axes)
        cbar.set_label("log2(k)")

        # Add legend for marker shapes
        axes[2].scatter([], [], marker="o", c="gray", alpha=0.5, label="SparseLinear")
        axes[2].scatter([], [], marker="^", c="gray", alpha=0.5, label="DecisionTree")
        axes[2].legend(loc="upper left")

        fig.suptitle("FishVista: Feature Grounding (AUC over Yield@B)")
        return mo.vstack([table, fig])

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_3"

        # Filter to FishVista with audit results
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No audit results available yet.")

        tasks = sorted(filtered.get_column("task_name").unique().to_list())

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        for i, task in enumerate(tasks):
            ax = axes[i]
            task_df = filtered.filter(pl.col("task_name") == task)

            for clf_key in ["sparse-linear", "decision-tree"]:
                group = task_df.filter(pl.col("cls/cfg/cls/key") == clf_key)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                marker = "o" if clf_key == "sparse-linear" else "^"
                ax.scatter(
                    xs, ys, alpha=0.6, marker=marker, label=clf_key if i == 0 else None
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(task.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        for i in range(len(tasks), len(axes)):
            axes[i].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@3")
        fig.suptitle("FishVista: Yield@3 by Task")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_3"
        k_col = "config/sae/activation/top_k"

        # Filter to FishVista with audit results
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No audit results available yet.")

        top_ks = sorted(filtered.get_column(k_col).unique().to_list())
        cmap = plt.cm.plasma

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = sorted(filtered.get_column("config/val_data/layer").unique().to_list())
        layer_colors = {l: cmap(i / (len(layers) - 1)) for i, l in enumerate(layers)}

        for i, k in enumerate(top_ks):
            ax = axes[i]
            k_df = filtered.filter(pl.col(k_col) == k)

            for layer in layers:
                group = k_df.filter(pl.col("config/val_data/layer") == layer)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.6,
                    c=[layer_colors[layer]],
                    label=f"L{layer + 1}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"top_k = {k}")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        for i in range(len(top_ks), len(axes)):
            axes[i].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=7, ncol=2)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@3")
        fig.suptitle("FishVista: Yield@3 by Top-K (colored by layer)")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_3"

        # Filter to FishVista with audit results, sparse-linear only
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "sparse-linear")
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No sparse-linear audit results available yet.")

        c_values = sorted(filtered.get_column("cls/cfg/cls/C").unique().to_list())
        cmap = plt.cm.viridis

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        c_colors = {c: cmap(i / (len(c_values) - 1)) for i, c in enumerate(c_values)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for c in c_values:
                group = layer_df.filter(pl.col("cls/cfg/cls/C") == c)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[c_colors[c]],
                    label=f"C={c}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@3")
        fig.suptitle("SparseLinear: Yield@3 by Regularization (C)")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_3"

        # Filter to FishVista with audit results, decision-tree only
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "decision-tree")
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No decision-tree audit results available yet.")

        depths = sorted(filtered.get_column("cls/cfg/cls/max_depth").unique().to_list())
        cmap = plt.cm.coolwarm

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        depth_colors = {d: cmap(i / (len(depths) - 1)) for i, d in enumerate(depths)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for d in depths:
                group = layer_df.filter(pl.col("cls/cfg/cls/max_depth") == d)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                label_d = "unlimited" if d == -1 else str(d)
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[depth_colors[d]],
                    label=f"d={label_d}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=7)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@3")
        fig.suptitle("DecisionTree: Yield@3 by Max Depth")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_10"

        # Filter to FishVista with audit results
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No audit results available yet.")

        tasks = sorted(filtered.get_column("task_name").unique().to_list())

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        for i, task in enumerate(tasks):
            ax = axes[i]
            task_df = filtered.filter(pl.col("task_name") == task)

            for clf_key in ["sparse-linear", "decision-tree"]:
                group = task_df.filter(pl.col("cls/cfg/cls/key") == clf_key)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                marker = "o" if clf_key == "sparse-linear" else "^"
                ax.scatter(
                    xs, ys, alpha=0.6, marker=marker, label=clf_key if i == 0 else None
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(task.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        for i in range(len(tasks), len(axes)):
            axes[i].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@10")
        fig.suptitle("FishVista: Yield@10 by Task")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_10"

        # Filter to FishVista with audit results, sparse-linear only
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "sparse-linear")
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No sparse-linear audit results available yet.")

        c_values = sorted(filtered.get_column("cls/cfg/cls/C").unique().to_list())
        cmap = plt.cm.viridis

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        c_colors = {c: cmap(i / (len(c_values) - 1)) for i, c in enumerate(c_values)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for c in c_values:
                group = layer_df.filter(pl.col("cls/cfg/cls/C") == c)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[c_colors[c]],
                    label=f"C={c}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=8)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@10")
        fig.suptitle("SparseLinear: Yield@10 by Regularization (C)")
        return fig

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, pl, plt):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/audit/yield_at_10"

        # Filter to FishVista with audit results, decision-tree only
        filtered = df.filter(
            (pl.col("cls/cfg/patch_agg") == "max")
            & (pl.col("data_key") == "FishVista (Img)")
            & pl.col(y_col).is_not_null()
            & (pl.col("cls/cfg/cls/key") == "decision-tree")
        ).with_columns(pl.col("cls/cfg/task").struct.field("name").alias("task_name"))

        if filtered.height == 0:
            return mo.md("No decision-tree audit results available yet.")

        depths = sorted(filtered.get_column("cls/cfg/cls/max_depth").unique().to_list())
        cmap = plt.cm.coolwarm

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(10, 6),
            dpi=150,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        layers = [13, 15, 17, 19, 21, 23]
        depth_colors = {d: cmap(i / (len(depths) - 1)) for i, d in enumerate(depths)}

        for i, layer in enumerate(layers):
            ax = axes[i]
            layer_df = filtered.filter(pl.col("config/val_data/layer") == layer)

            for d in depths:
                group = layer_df.filter(pl.col("cls/cfg/cls/max_depth") == d)
                if group.height == 0:
                    continue
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()
                label_d = "unlimited" if d == -1 else str(d)
                ax.scatter(
                    xs,
                    ys,
                    alpha=0.7,
                    c=[depth_colors[d]],
                    label=f"d={label_d}" if i == 0 else None,
                )

            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Layer {layer + 1}/24")
            ax.grid(True, alpha=0.3)
            ax.spines[["right", "top"]].set_visible(False)

        axes[0].legend(loc="upper left", fontsize=7)
        fig.supxlabel("# Non-zero Features")
        fig.supylabel("Yield@10")
        fig.suptitle("DecisionTree: Yield@10 by Max Depth")
        return fig

    _(clf_df)
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

        if isinstance(
            data_cfg, saev.data.datasets.ImgFolder
        ) and "butterflies-imgfolder" in str(data_cfg.root):
            return "Butterflies (Img)"

        print(f"Unknown data: {data_cfg}")
        return None

    return get_data_key, get_model_key


@app.cell
def _(
    beartype,
    cloudpickle,
    collections,
    json,
    np,
    pathlib,
    saev,
    shards_root,
    sklearn,
):
    @beartype.beartype
    def get_shards_split_label(shards_dpath: pathlib.Path) -> str | None:
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

    @beartype.beartype
    def get_cls_results_fpaths(run: saev.disk.Run) -> dict[str, list[pathlib.Path]]:
        if not run.inference.is_dir():
            return {}

        globbed = list(run.inference.glob("**/cls_*.pkl"))
        if not globbed:
            return {}

        cls_results_fpaths = collections.defaultdict(list)
        for results_fpath in globbed:
            shard_id = results_fpath.parent.name
            shards_dpath = shards_root / shard_id

            if not shards_dpath.exists():
                print(f"Skipping {run.run_id}: shards dir {shards_dpath} missing.")
                continue

            split_label = get_shards_split_label(shards_dpath)
            if split_label is None:
                # For some datasets we use the same shards for train/test
                split_label = "all"

            cls_results_fpaths[split_label].append(results_fpath)

        return cls_results_fpaths

    @beartype.beartype
    def get_audit_results(run: saev.disk.Run) -> dict[str, dict[str, object]]:
        """Load audit results and return a dict mapping checkpoint filename to audit metrics."""
        audit_by_ckpt = {}
        for audit_fpath in run.inference.glob("**/audit_results.json"):
            try:
                with open(audit_fpath, "r") as fd:
                    audit_data = json.load(fd)
                for clf_result in audit_data.get("classifiers", []):
                    ckpt_path = clf_result.get("cls_checkpoint", "")
                    ckpt_fname = pathlib.Path(ckpt_path).name
                    audit_by_ckpt[ckpt_fname] = {
                        "audit/cls_type": clf_result.get("cls_type"),
                        "audit/n_nonzero_importance": clf_result.get(
                            "n_nonzero_importance"
                        ),
                        "audit/tau": clf_result.get("tau"),
                        "audit/auc_b": clf_result.get("auc_b"),
                        **{
                            f"audit/yield_at_{k}": v
                            for k, v in clf_result.get("yield_at_b", {}).items()
                        },
                    }
            except Exception as err:
                print(f"Failed to load audit results from {audit_fpath}: {err}")
        return audit_by_ckpt

    @beartype.beartype
    def get_cls_results(run: saev.disk.Run) -> list[dict[str, object]]:
        audit_by_ckpt = get_audit_results(run)

        results = []
        for split, results_fpaths in get_cls_results_fpaths(run).items():
            for fpath in results_fpaths:
                try:
                    with open(fpath, "rb") as fd:
                        header_line = fd.readline()
                        header = json.loads(header_line.decode("utf8"))
                        ckpt = cloudpickle.load(fd)

                    # Compute additional metrics from test_pred and test_y
                    test_pred = ckpt["test_pred"]
                    test_y = ckpt["test_y"]

                    balanced_acc = sklearn.metrics.balanced_accuracy_score(
                        test_y, test_pred
                    )
                    macro_f1 = sklearn.metrics.f1_score(
                        test_y, test_pred, average="macro"
                    )

                    # Sparsity diagnostics
                    clf = ckpt["classifier"]
                    if hasattr(clf, "coef_"):
                        # SparseLinear: count non-zero coefficients
                        n_nonzero = int(np.count_nonzero(clf.coef_))
                    elif hasattr(clf, "tree_"):
                        # DecisionTree: count features used
                        n_nonzero = int((clf.feature_importances_ > 0).sum())
                    else:
                        n_nonzero = None

                    result = header | {
                        "classifier": clf,
                        "balanced_acc": balanced_acc,
                        "macro_f1": macro_f1,
                        "n_nonzero": n_nonzero,
                    }

                    # Merge audit results if available
                    ckpt_fname = fpath.name
                    if ckpt_fname in audit_by_ckpt:
                        result |= audit_by_ckpt[ckpt_fname]

                    results.append(result)

                except Exception as err:
                    print(f"Failed to load {fpath}: {err}")
                    continue

        return results

    return (get_cls_results,)


if __name__ == "__main__":
    app.run()
