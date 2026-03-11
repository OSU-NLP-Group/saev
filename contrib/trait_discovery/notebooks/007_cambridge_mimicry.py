import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import pathlib

    import beartype
    import cloudpickle
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import sklearn.metrics
    from tdiscovery.classification import (
        LabelGrouping,
        apply_grouping,
        extract_feature_ranking,
        load_image_labels,
    )

    return (
        LabelGrouping,
        apply_grouping,
        beartype,
        cloudpickle,
        extract_feature_ranking,
        json,
        load_image_labels,
        mo,
        np,
        pathlib,
        pl,
        plt,
        sklearn,
    )


@app.cell
def _(pathlib):
    RUNS_ROOT_DPATH = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")

    # Append another dict when new shard versions are ready.
    DATASET_SPECS = [
        {
            "version": "v1.6",
            "shards_by_n_patches": {
                640: "/fs/scratch/PAS2136/samuelstevens/saev/shards/79239bdd"
            },
            "run_ids_by_layer": {
                21: ["zhul9opa", "gz2dikb3", "3rqci2h1", "r27w7pmf", "x4n29kua"],
                23: ["pnsi8yhe", "onqqe859", "rd8wc24d", "vends70d", "pa5cu0mf"],
            },
        }
    ]
    ACTIVE_VERSIONS = {"v1.6"}

    MIMIC_PAIRS = [
        ("lativitta", "malleti"),
        ("cyrbia", "cythera"),
        ("notabilis", "plesseni"),
        ("hydara", "melpomene"),
        ("venus", "vulcanus"),
        ("demophoon", "rosina"),
        ("phyllis", "nanna"),
        ("erato", "thelxiopeia"),
    ]
    VIEWS = ["dorsal", "ventral"]
    C_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    MIN_SAMPLES_PER_CLASS = 50
    TOP_K = 20
    return (
        ACTIVE_VERSIONS,
        C_VALUES,
        DATASET_SPECS,
        MIMIC_PAIRS,
        MIN_SAMPLES_PER_CLASS,
        RUNS_ROOT_DPATH,
        TOP_K,
        VIEWS,
    )


@app.cell
def _(mo):
    mo.md("""
    # Cambridge Mimic Pair Feature Discrimination

    This notebook analyzes `cls::train` outputs for the Cambridge mimic-pair sweep.
    It computes balanced accuracy from saved predictions, summarizes sparsity vs accuracy tradeoffs, ranks selected SAE features, and builds a cross-pair difficulty table.

    Note: this is feature discovery, not generalization. The sweep uses the same shards for train and test.
    """)
    return


@app.cell
def _(beartype, pathlib):
    @beartype.beartype
    def get_task_name(erato_ssp: str, melp_ssp: str, view: str) -> str:
        return f"{erato_ssp}_{view}_vs_{melp_ssp}_{view}"

    @beartype.beartype
    def get_pretty_task_name(task_name: str) -> str:
        return task_name.replace("_vs_", " vs ").replace("_", " ")

    @beartype.beartype
    def parse_run_id_from_ckpt_fpath(ckpt_fpath: pathlib.Path) -> str:
        parts = ckpt_fpath.parts
        msg = f"'inference' not in checkpoint path: {ckpt_fpath}"
        assert "inference" in parts, msg
        i_inference = parts.index("inference")
        msg = f"Unexpected checkpoint path, cannot parse run id: {ckpt_fpath}"
        assert i_inference > 0, msg
        return parts[i_inference - 1]

    return get_pretty_task_name, get_task_name, parse_run_id_from_ckpt_fpath


@app.cell
def _(DATASET_SPECS, beartype):
    @beartype.beartype
    def make_run_to_layer_map(dataset_specs: list[dict]) -> dict[str, int]:
        run_to_layer = {}
        for spec in dataset_specs:
            run_ids_by_layer = spec["run_ids_by_layer"]
            for layer, run_ids in run_ids_by_layer.items():
                for run_id in run_ids:
                    run_to_layer[str(run_id)] = int(layer)
        return run_to_layer

    run_id_to_layer = make_run_to_layer_map(DATASET_SPECS)
    return (run_id_to_layer,)


@app.cell
def _(
    ACTIVE_VERSIONS,
    C_VALUES,
    DATASET_SPECS,
    LabelGrouping,
    MIMIC_PAIRS,
    MIN_SAMPLES_PER_CLASS,
    RUNS_ROOT_DPATH,
    VIEWS,
    apply_grouping,
    beartype,
    cloudpickle,
    get_task_name,
    json,
    load_image_labels,
    np,
    parse_run_id_from_ckpt_fpath,
    pathlib,
    pl,
    run_id_to_layer,
    sklearn,
):
    @beartype.beartype
    def get_pair_counts_df() -> pl.DataFrame:
        rows = []
        for spec in DATASET_SPECS:
            version = str(spec["version"])
            if version not in ACTIVE_VERSIONS:
                continue

            for n_patches, shards_dpath_raw in spec["shards_by_n_patches"].items():
                shards_dpath = pathlib.Path(str(shards_dpath_raw))
                if not shards_dpath.exists():
                    print(f"Skipping missing shards directory: {shards_dpath}")
                    continue

                _, labels_by_col = load_image_labels(shards_dpath)
                msg = f"Expected 'subspecies_view' labels in {shards_dpath}"
                assert "subspecies_view" in labels_by_col, msg
                ssp_view_labels = labels_by_col["subspecies_view"]

                for erato_ssp, melp_ssp in MIMIC_PAIRS:
                    for view in VIEWS:
                        task_name = get_task_name(erato_ssp, melp_ssp, view)
                        task = LabelGrouping(
                            name=task_name,
                            source_col="subspecies_view",
                            groups={
                                "erato": [f"{erato_ssp}_{view}"],
                                "melpomene": [f"{melp_ssp}_{view}"],
                            },
                        )
                        targets_n, class_names, include_n = apply_grouping(
                            ssp_view_labels, task
                        )
                        class_to_i = {name: i for i, name in enumerate(class_names)}
                        msg = f"Missing expected classes for {task_name}: {class_names}"
                        assert {"erato", "melpomene"} <= set(class_to_i), msg

                        filtered_targets_n = targets_n[include_n]
                        n_erato = int((filtered_targets_n == class_to_i["erato"]).sum())
                        n_melp = int(
                            (filtered_targets_n == class_to_i["melpomene"]).sum()
                        )
                        n_total = int(filtered_targets_n.shape[0])
                        assert n_total == n_erato + n_melp

                        majority_acc = (
                            None if n_total == 0 else max(n_erato, n_melp) / n_total
                        )
                        insufficient_data = min(n_erato, n_melp) < MIN_SAMPLES_PER_CLASS

                        rows.append({
                            "version": version,
                            "n_patches": int(n_patches),
                            "task": task_name,
                            "n_erato": n_erato,
                            "n_melpomene": n_melp,
                            "n_total": n_total,
                            "majority_acc": majority_acc,
                            "insufficient_data": insufficient_data,
                        })

        return pl.DataFrame(rows, infer_schema_length=None)

    @beartype.beartype
    def get_results_df(pair_task_name_set: set[str]) -> pl.DataFrame:
        rows = []
        for spec in DATASET_SPECS:
            version = str(spec["version"])
            if version not in ACTIVE_VERSIONS:
                continue

            run_ids_by_layer = spec["run_ids_by_layer"]
            for n_patches, shards_dpath_raw in spec["shards_by_n_patches"].items():
                shard_id = pathlib.Path(str(shards_dpath_raw)).name
                for run_ids in run_ids_by_layer.values():
                    for run_id in run_ids:
                        ckpts_dpath = RUNS_ROOT_DPATH / run_id / "inference" / shard_id
                        if not ckpts_dpath.is_dir():
                            continue

                        for ckpt_fpath in sorted(ckpts_dpath.glob("cls_*.pkl")):
                            try:
                                with open(ckpt_fpath, "rb") as fd:
                                    header_line = fd.readline()
                                    header = json.loads(header_line.decode("utf8"))
                                    payload = cloudpickle.load(fd)
                            except Exception as err:
                                print(f"Failed loading {ckpt_fpath}: {err}")
                                continue

                            cfg = header["cfg"]
                            task_name = cfg["task"]["name"]
                            if task_name not in pair_task_name_set:
                                continue
                            if cfg["patch_agg"] != "max":
                                continue

                            cls_cfg = cfg["cls"]
                            if cls_cfg["key"] != "sparse-linear":
                                continue

                            C = float(cls_cfg["C"])
                            if C not in C_VALUES:
                                continue
                            test_y_n = np.asarray(payload["test_y"])
                            test_pred_n = np.asarray(payload["test_pred"])
                            if test_y_n.size == 0:
                                continue
                            msg = f"Shape mismatch in {ckpt_fpath}"
                            assert test_y_n.shape == test_pred_n.shape, msg

                            classifier = payload["classifier"]
                            coef_cd = np.asarray(classifier.coef_)
                            nonzero_mask_d = np.any(coef_cd != 0, axis=0)
                            n_nonzero = int(nonzero_mask_d.sum())
                            features = np.where(nonzero_mask_d)[0].tolist()
                            # Binary: coef shape is (1, d), positive => melpomene
                            weights = (
                                coef_cd[0, nonzero_mask_d].tolist()
                                if coef_cd.shape[0] == 1
                                else []
                            )

                            n_classes = int(header["n_classes"])
                            class_names = [str(name) for name in header["class_names"]]
                            class_counts_c = np.bincount(test_y_n, minlength=n_classes)
                            n_examples = int(class_counts_c.sum())
                            if n_examples == 0:
                                continue

                            class_to_i = {name: i for i, name in enumerate(class_names)}
                            n_erato = (
                                int(class_counts_c[class_to_i["erato"]])
                                if "erato" in class_to_i
                                else None
                            )
                            n_melp = (
                                int(class_counts_c[class_to_i["melpomene"]])
                                if "melpomene" in class_to_i
                                else None
                            )
                            majority_acc = float(class_counts_c.max() / n_examples)
                            balanced_acc = float(
                                sklearn.metrics.balanced_accuracy_score(
                                    test_y_n, test_pred_n
                                )
                            )

                            run_id_from_path = parse_run_id_from_ckpt_fpath(ckpt_fpath)
                            if run_id_from_path not in run_id_to_layer:
                                continue

                            rows.append({
                                "version": version,
                                "n_patches": int(n_patches),
                                "shard_id": shard_id,
                                "run_id": run_id_from_path,
                                "layer": run_id_to_layer[run_id_from_path],
                                "task": task_name,
                                "C": C,
                                "test_acc": float(header["test_acc"]),
                                "balanced_acc": balanced_acc,
                                "majority_acc": majority_acc,
                                "n_nonzero": n_nonzero,
                                "features": features,
                                "weights": weights,
                                "n_examples": n_examples,
                                "n_erato_test": n_erato,
                                "n_melpomene_test": n_melp,
                                "ckpt_fpath": str(ckpt_fpath),
                            })

        return pl.DataFrame(rows, infer_schema_length=None)

    pair_counts_df = get_pair_counts_df()
    results_df = get_results_df(
        set(get_task_name(*pair, view=view) for pair in MIMIC_PAIRS for view in VIEWS)
    )
    return pair_counts_df, results_df


@app.cell
def _(mo, results_df):
    mo.md(f"""
    Loaded `{results_df.height}` classifier checkpoints.

    Pair count metadata (exact-match grouping on `subspecies_view`, hybrids excluded):
    """)
    return


@app.cell
def _(pair_counts_df):
    pair_counts_df.sort("task")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Checkpoint Explorer
    """)
    return


@app.cell
def _(pl, results_df):
    results_df.select(
        "run_id",
        "shard_id",
        "ckpt_fpath",
        "task",
        "layer",
        "C",
        "n_nonzero",
        "balanced_acc",
        pl.col("features").list.len().alias("n_features"),
        "features",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Feature Browser

    One row per feature per checkpoint. Positive weight favors melpomene, negative favors erato.
    """)
    return


@app.cell
def _(pl, results_df):
    feature_df = (
        results_df
        .select(
            "task",
            "layer",
            "C",
            "n_nonzero",
            "balanced_acc",
            "run_id",
            "shard_id",
            "ckpt_fpath",
            "features",
            "weights",
        )
        .explode("features", "weights")
        .rename({"features": "feature", "weights": "weight"})
        .with_columns(
            pl
            .when(pl.col("weight") > 0)
            .then(pl.lit("melpomene"))
            .otherwise(pl.lit("erato"))
            .alias("favors"),
        )
        .sort(
            "task",
            "n_nonzero",
            "balanced_acc",
            "feature",
            descending=[False, False, True, False],
        )
    )
    feature_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part A: Sparsity-Accuracy Tradeoff
    """)
    return


@app.cell
def _(
    C_VALUES,
    MIMIC_PAIRS,
    VIEWS,
    get_pretty_task_name,
    get_task_name,
    pair_counts_df,
    pl,
    plt,
    results_df,
):
    def _():
        pair_counts_by_task = {
            row["task"]: row
            for row in pair_counts_df.sort("version", "n_patches").iter_rows(named=True)
        }

        n_pairs = len(MIMIC_PAIRS)
        n_views = len(VIEWS)
        fig_a, axes = plt.subplots(
            n_pairs,
            n_views,
            figsize=(10, 28),
            dpi=120,
            sharex=True,
            layout="constrained",
        )
        c_to_color = {
            0.00001: "#9467bd",
            0.0001: "#d62728",
            0.001: "#1f77b4",
            0.01: "#ff7f0e",
            0.1: "#2ca02c",
        }

        for i, (erato_ssp, melp_ssp) in enumerate(MIMIC_PAIRS):
            for j, view in enumerate(VIEWS):
                ax = axes[i, j]
                task_name = get_task_name(erato_ssp, melp_ssp, view)
                pair_df = results_df.filter(pl.col("task") == task_name)
                pair_counts = pair_counts_by_task.get(task_name, {})
                n_erato = int(pair_counts.get("n_erato", 0))
                n_melp = int(pair_counts.get("n_melpomene", 0))
                majority_acc = pair_counts.get("majority_acc", None)

                title = (
                    f"{get_pretty_task_name(task_name)} ({n_erato} vs {n_melp} samples)"
                )
                ax.set_title(title)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_xlabel("# Non-Zero Features")
                ax.set_ylabel("Balanced Accuracy")
                ax.set_ylim(0, 1.02)
                ax.set_xscale("log")
                ax.grid(alpha=0.2)

                if majority_acc is not None:
                    ax.axhline(
                        float(majority_acc),
                        color="gray",
                        linestyle=":",
                        linewidth=2,
                        alpha=0.5,
                        label="majority baseline",
                    )

                if pair_df.height == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No checkpoints",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                for C in C_VALUES:
                    c_df = pair_df.filter(pl.col("C") == C)
                    if c_df.height == 0:
                        continue
                    ax.scatter(
                        c_df.get_column("n_nonzero").to_numpy(),
                        c_df.get_column("balanced_acc").to_numpy(),
                        color=c_to_color[C],
                        alpha=0.5,
                        label=f"C={C}",
                    )

                agg_df = (
                    pair_df
                    .group_by("C")
                    .agg(
                        pl.col("n_nonzero").mean().alias("n_nonzero_mean"),
                        pl.col("balanced_acc").mean().alias("balanced_acc_mean"),
                        pl.col("balanced_acc").std().alias("balanced_acc_std"),
                        pl.col("test_acc").mean().alias("test_acc_mean"),
                    )
                    .sort("C")
                )
                yerr = agg_df.get_column("balanced_acc_std").fill_null(0.0).to_numpy()
                ax.errorbar(
                    agg_df.get_column("n_nonzero_mean").to_numpy(),
                    agg_df.get_column("balanced_acc_mean").to_numpy(),
                    yerr=yerr,
                    color="black",
                    alpha=0.5,
                    marker="o",
                    markersize=4,
                    label="balanced acc mean+/-std",
                )
                ax.plot(
                    agg_df.get_column("n_nonzero_mean").to_numpy(),
                    agg_df.get_column("test_acc_mean").to_numpy(),
                    color="black",
                    alpha=0.5,
                    linestyle="--",
                    marker="s",
                    markersize=3.5,
                    label="raw acc mean",
                )

        import matplotlib.ticker as mticker

        ticks = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
        axes[0, 0].xaxis.set_major_locator(mticker.FixedLocator(ticks))
        axes[0, 0].xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x)}")
        )
        axes[0, 0].xaxis.set_minor_locator(mticker.NullLocator())
        for ax_row in axes:
            for ax in ax_row:
                ax.tick_params(labelbottom=True)

        axes[0, 0].legend()
        fig_a.suptitle("Mimic pair sparsity-accuracy tradeoff per wing view")

        return fig_a

    _()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part B: Feature Ranking Table

    For each pair, this table takes the best L1 model by balanced accuracy and lists its top nonzero features.
    Sign convention for `weight`: positive favors class 1 (`melpomene`), negative favors class 0 (`erato`).
    """)
    return


@app.cell
def _(
    MIMIC_PAIRS,
    TOP_K,
    VIEWS,
    cloudpickle,
    extract_feature_ranking,
    get_task_name,
    json,
    np,
    pathlib,
    pl,
    results_df,
):
    def _():
        ranking_rows = []
        for erato_ssp, melp_ssp in MIMIC_PAIRS:
            for view in VIEWS:
                task_name = get_task_name(erato_ssp, melp_ssp, view)
                task_df = results_df.filter(pl.col("task") == task_name)
                if task_df.height == 0:
                    continue

                best_row = task_df.sort(
                    "balanced_acc", "n_nonzero", descending=[True, False]
                ).row(0, named=True)
                best_ckpt_fpath = pathlib.Path(best_row["ckpt_fpath"])
                with open(best_ckpt_fpath, "rb") as fd:
                    header_line = fd.readline()
                    header = json.loads(header_line.decode("utf8"))
                    payload = cloudpickle.load(fd)

                class_names = [str(name) for name in header["class_names"]]
                if class_names != ["erato", "melpomene"]:
                    print(
                        f"Unexpected class order for {best_ckpt_fpath}: {class_names}"
                    )

                classifier = payload["classifier"]
                ranked_i, importance_d = extract_feature_ranking(
                    classifier, "sparse-linear"
                )
                coef_cd = np.asarray(classifier.coef_)
                if coef_cd.ndim != 2 or coef_cd.shape[0] != 1:
                    print(
                        f"Skipping non-binary coefficient shape {coef_cd.shape}: {best_ckpt_fpath}"
                    )
                    continue
                weights_d = coef_cd[0]

                n_added = 0
                for feature_i in ranked_i:
                    weight = float(weights_d[feature_i])
                    if weight == 0.0:
                        continue
                    n_added += 1
                    ranking_rows.append({
                        "task": task_name,
                        "rank": n_added,
                        "feature_i": int(feature_i),
                        "weight": weight,
                        "abs_weight": float(abs(weight)),
                        "importance": float(importance_d[feature_i]),
                        "favored_class": "melpomene" if weight > 0 else "erato",
                        "run_id": best_row["run_id"],
                        "layer": int(best_row["layer"]),
                        "C": float(best_row["C"]),
                        "best_balanced_acc": float(best_row["balanced_acc"]),
                    })
                    if n_added >= TOP_K:
                        break

        ranking_df = pl.DataFrame(ranking_rows, infer_schema_length=None)
        if ranking_df.height > 0:
            ranking_df = ranking_df.sort("task", "rank")
        return ranking_df

    ranking_df = _()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part C: Cross-Pair Comparison
    """)
    return


@app.cell
def _(
    MIMIC_PAIRS,
    VIEWS,
    beartype,
    cloudpickle,
    extract_feature_ranking,
    get_task_name,
    json,
    np,
    pair_counts_df,
    pathlib,
    pl,
    results_df,
):
    pair_counts_by_task = {
        row["task"]: row
        for row in pair_counts_df.sort("version", "n_patches").iter_rows(named=True)
    }

    @beartype.beartype
    def get_best_nonzero_feature_i(ckpt_fpath: pathlib.Path) -> int | None:
        with open(ckpt_fpath, "rb") as fd:
            header_line = fd.readline()
            _ = json.loads(header_line.decode("utf8"))
            payload = cloudpickle.load(fd)

        classifier = payload["classifier"]
        coef_cd = np.asarray(classifier.coef_)
        if coef_cd.ndim != 2 or coef_cd.shape[0] != 1:
            return None
        weights_d = coef_cd[0]
        ranked_i, _ = extract_feature_ranking(classifier, "sparse-linear")
        for feature_i in ranked_i:
            if weights_d[feature_i] != 0.0:
                return int(feature_i)
        return None

    summary_rows = []
    for pair in MIMIC_PAIRS:
        for view in VIEWS:
            task_name = get_task_name(*pair, view=view)
            pair_df = results_df.filter(pl.col("task") == task_name)
            pair_counts = pair_counts_by_task.get(task_name)

            n_erato = None if pair_counts is None else int(pair_counts["n_erato"])
            n_melp = None if pair_counts is None else int(pair_counts["n_melpomene"])
            majority_acc = None if pair_counts is None else pair_counts["majority_acc"]

            if pair_df.height == 0:
                summary_rows.append({
                    "task": task_name,
                    "best_balanced_acc": None,
                    "n_features_for_90_bal_acc": None,
                    "best_single_feature_i": None,
                    "best_single_feature_balanced_acc": None,
                    "majority_baseline_acc": majority_acc,
                    "n_erato": n_erato,
                    "n_melpomene": n_melp,
                    "n_ckpts": 0,
                })
                continue

            best_balanced_acc = float(pair_df.get_column("balanced_acc").max())
            over_90_df = pair_df.filter(pl.col("balanced_acc") >= 0.90)
            n_features_for_90 = (
                None
                if over_90_df.height == 0
                else int(over_90_df.get_column("n_nonzero").min())
            )

            single_df = pair_df.filter(pl.col("n_nonzero") == 1).sort(
                "balanced_acc", descending=True
            )
            if single_df.height == 0:
                best_single_feature_i = None
                best_single_balanced_acc = None
            else:
                single_best_row = single_df.row(0, named=True)
                best_single_feature_i = get_best_nonzero_feature_i(
                    pathlib.Path(single_best_row["ckpt_fpath"])
                )
                best_single_balanced_acc = float(single_best_row["balanced_acc"])

            summary_rows.append({
                "task": task_name,
                "best_balanced_acc": best_balanced_acc,
                "n_features_for_90_bal_acc": n_features_for_90,
                "best_single_feature_i": best_single_feature_i,
                "best_single_feature_balanced_acc": best_single_balanced_acc,
                "majority_baseline_acc": majority_acc,
                "n_erato": n_erato,
                "n_melpomene": n_melp,
                "n_ckpts": int(pair_df.height),
            })

    summary_df = pl.DataFrame(summary_rows, infer_schema_length=None)
    if summary_df.height > 0:
        summary_df = (
            summary_df
            .with_columns(
                pl
                .when(pl.col("best_balanced_acc").is_null())
                .then(-1.0)
                .otherwise(pl.col("best_balanced_acc"))
                .alias("_difficulty_bal"),
                pl
                .when(pl.col("n_features_for_90_bal_acc").is_null())
                .then(10**9)
                .otherwise(pl.col("n_features_for_90_bal_acc"))
                .alias("_difficulty_feat"),
            )
            .sort("_difficulty_bal", "_difficulty_feat", descending=[False, True])
            .with_row_index("difficulty_rank", offset=1)
            .drop("_difficulty_bal", "_difficulty_feat")
        )
    summary_df
    return


if __name__ == "__main__":
    app.run()
