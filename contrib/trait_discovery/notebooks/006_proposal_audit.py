import marimo

__generated_with = "0.18.4"
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
            df
            .unnest("config/sae", "config/train_data/metadata", separator="/")
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
            df
            .unnest("config/sae", "config/train_data/metadata", separator="/")
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
def _(clf_df, mo, np, pl, plt, saev, scipy):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/macro_f1"

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
            filtered
            .select(
                "config/val_data/layer",
                "cls/cfg/cls/key",
                "cls/n_nonzero",
                "cls/macro_f1",
                "cls/test_acc",
                "cls/balanced_acc",
            )
            .sort("config/val_data/layer", "cls/cfg/cls/key", "cls/n_nonzero")
            .with_columns(pl.lit(random_chance).alias("random_chance"))
        )

        # Define classifier styles
        clf_styles = {
            "sparse-linear": (saev.colors.BLUE_RGB01, "o", "SparseLinear"),
            "decision-tree": (saev.colors.ORANGE_RGB01, "^", "DecisionTree"),
        }

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

            for clf_key, (color, marker, label) in clf_styles.items():
                group = filtered.filter(
                    (pl.col("config/val_data/layer") == layer)
                    & (pl.col("cls/cfg/cls/key") == clf_key)
                )

                if group.height == 0:
                    continue

                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()

                ax.scatter(
                    xs,
                    ys,
                    alpha=0.5,
                    color=color,
                    marker=marker,
                    label=label,
                )

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

            if i == 2:
                ax.legend()

        fig.suptitle("ADE20K Scene Classification (Top 50)")
        return mo.vstack([table, fig])

    _(clf_df)
    return


@app.cell
def _(clf_df, mo, np, pl, plt, saev, scipy):
    def _(df: pl.DataFrame):
        x_col = "cls/n_nonzero"
        y_col = "cls/macro_f1"

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
            filtered
            .select(
                "config/val_data/layer",
                "cls/cfg/cls/key",
                "cls/n_nonzero",
                "cls/macro_f1",
                "cls/test_acc",
                "cls/balanced_acc",
            )
            .sort("config/val_data/layer", "cls/cfg/cls/key", "cls/n_nonzero")
            .with_columns(pl.lit(random_chance).alias("random_chance"))
        )

        # Define classifier styles
        clf_styles = {
            "sparse-linear": (saev.colors.BLUE_RGB01, "o", "SparseLinear"),
            "decision-tree": (saev.colors.ORANGE_RGB01, "^", "DecisionTree"),
        }

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

            for clf_key, (color, marker, label) in clf_styles.items():
                group = filtered.filter(
                    (pl.col("config/val_data/layer") == layer)
                    & (pl.col("cls/cfg/cls/key") == clf_key)
                )

                if group.height == 0:
                    continue

                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()

                ax.scatter(
                    xs,
                    ys,
                    alpha=0.5,
                    color=color,
                    marker=marker,
                    label=label,
                )

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

            if i == 2:
                ax.legend()

        fig.suptitle("FishVista Habitat")
        return mo.vstack([table, fig])

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
    def get_cls_results(run: saev.disk.Run) -> list[dict[str, object]]:
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
                    results.append(result)

                except Exception as err:
                    print(f"Failed to load {fpath}: {err}")
                    continue

        return results
    return (get_cls_results,)


if __name__ == "__main__":
    app.run()
