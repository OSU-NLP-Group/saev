import marimo

__generated_with = "0.17.2"
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
        wandb,
    )


@app.cell
def _(pathlib):
    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "saev"
    WANDB_TAG = "fishbase-v0.1"

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

        cls_results = get_cls_results(saev_run)
        if not cls_results:
            print(f"Run {wandb_run.id} missing cls results.")
            return None

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

        return row, cls_results

    @beartype.beartype
    def _finalize_df(rows: list[dict[str, object]]):
        df = pl.DataFrame(rows, infer_schema_length=None)

        # activation_schema = pl.Struct([
        #     pl.Field("sparsity", pl.Struct([pl.Field("coeff", pl.Float64)])),
        #     pl.Field("top_k", pl.Int64),
        # ])

        # df = df.with_columns(
        #     pl.col("config/sae/activation").cast(
        #         activation_schema, strict=False
        #     )  # add missing fields as null
        # )

        # df = df.with_columns(
        #     pl.when(pl.col("config/sae/activation").struct.field("top_k").is_not_null())
        #     .then(pl.lit("topk"))
        #     .otherwise(pl.lit("relu"))
        #     .alias("config/sae/activation_kind"),
        # )

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
            .unnest("cls/cfg", separator="/")
            .unnest("cls/cfg/cls", separator="/")
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

        # runs = [wandb.Api().run(path=f"{WANDB_USERNAME}/{WANDB_PROJECT}/pdikj9bl")]
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
                    result = fut.result()
                except Exception as err:
                    print(f"Run {fut_to_run_id[fut]} blew up: {err}")
                    continue
                if result is None:
                    continue

                run_row, cls_rows = result
                for cls_row in cls_rows:
                    rows.append(
                        run_row | {f"cls/{key}": val for key, val in cls_row.items()}
                    )

        assert rows, "No valid runs."
        return _finalize_df(rows)

    df = make_df_parallel()
    return (df,)


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

        print(f"Unknown data: {data_cfg}")
        return None

    return get_data_key, get_model_key


@app.cell
def _(beartype, pathlib, saev):
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

    return (get_shards_split_label,)


@app.cell
def _(
    beartype,
    cloudpickle,
    collections,
    get_shards_split_label,
    json,
    pathlib,
    saev,
    shards_root,
):
    @beartype.beartype
    def get_cls_results_fpaths(run: saev.disk.Run) -> dict[str, list[pathlib.Path]]:
        if not run.inference.is_dir():
            return {}

        globbed = list(run.inference.glob("**/cls_*.pkl"))
        if not globbed:
            print(f"Skipping {run.run_id}: no cls_*.pkl files.")
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
                print(f"Skipping shards {shard_id}: unknown split (run {run.run_id}).")
                continue

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

                    results.append(header | ckpt)

                except Exception as err:
                    print(f"Failed to load {fpath}: {err}")
                    continue

        return results

    return (get_cls_results,)


@app.cell
def _(df, mo, pl):
    def _(df):
        df = df.filter(
            True
            & (pl.col("cls/cfg/target_col") == "habitat")
            & (pl.col("cls/cfg/patch_agg") == "max")
            # & (pl.col("cls/cfg/cls/C") == 0.001)
            # & (pl.col("cls/cfg/cls/C").is_not_null())
            & (pl.col("cls/cfg/cls/max_depth").is_not_null())
            & (pl.col("cls/cfg/cls/max_depth") == 7)
        )

        clf = (
            df.filter((pl.col("id") == "hfpct5ae")).get_column("cls/classifier").item()
        )

        return df.select("^cls/.*$").sort("cls/test_acc", descending=True), clf

    disp, clf = _(df)

    # top_latents = np.abs(clf.coef_).argsort(axis=1)[:, -8:]

    mo.vstack([mo.md("# Sparse Classifiers"), disp, clf])
    return (clf,)


@app.cell
def _(clf, np):
    def print_tree(tree):
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        print(" ".join(str(f) for f in sorted(set(feature.tolist())) if f >= 0))
        threshold = tree.threshold
        values = tree.value

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        print(
            "The binary tree structure has {n} nodes and has "
            "the following tree structure:\n".format(n=n_nodes)
        )
        for i in range(n_nodes):
            if is_leaves[i]:
                print(
                    "{space}node={node} is a leaf node with value={value}.".format(
                        space=node_depth[i] * "\t",
                        node=i,
                        value=np.around(values[i], 3),
                    )
                )
            else:
                print(
                    "{space}node={node} is a split node with value={value}: "
                    "go to node {left} if X[:, {feature}] <= {threshold} "
                    "else to node {right}.".format(
                        space=node_depth[i] * "\t",
                        node=i,
                        left=children_left[i],
                        feature=feature[i],
                        threshold=threshold[i],
                        right=children_right[i],
                        value=np.around(values[i], 3),
                    )
                )

    print_tree(clf.tree_)
    return


@app.cell
def _(base64, df, pickle, saev, shards_root):
    def get_idx_to_label(df, col):
        """Get {int: string} mapping for a column."""
        shards = df.get_column("cls/cfg/train_shards").unique().item()

        md = saev.data.Metadata.load(shards_root / shards)

        b64 = md.data.encode("utf8")

        data_cfg = pickle.loads(base64.b64decode(b64))

        assert isinstance(data_cfg, saev.data.datasets.ImgSegFolder), data_cfg
        ds = saev.data.datasets.get_dataset(data_cfg)
        mapping = {}
        for sample in ds.samples:
            mapping[sample.targets[col]] = sample.labels[col]
        return mapping

    habitats = get_idx_to_label(df, "habitat")
    return (habitats,)


@app.cell
def _(clf, habitats, top_latents):
    def _():
        # TODO: pretty sure this order of habitats does NOT match the order used in classification.
        for h, habitat in sorted(habitats.items()):
            print(f"{habitat} ({h}) =", end="\n")
            for l, c in reversed(
                list(
                    zip(top_latents[h].tolist(), clf.coef_[h, top_latents[h]].tolist())
                )
            ):
                if c == 0:
                    break
                if c < 0:
                    continue
                print(f"\tLatent {l} x {c:.2g} +", end="\n")

            print(f"\t{clf.intercept_[h]:.2g}")

    _()
    return


@app.cell
def _(top_latents):
    print(" ".join([str(i) for i in sorted(set(top_latents.ravel()))]))
    return


@app.cell
def _(collections, df, mo, np, pl, plt, saev):
    def _(df: pl.DataFrame):
        # mse_col = "ade20k_val_nmse"
        x_col = "cls/cfg/cls/C"
        y_col = "cls/test_acc"

        topks = [8, 16, 32, 64, 128, 256]
        layers = [13, 15, 17, 19, 21, 23]

        point_alpha = 0.5

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

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            for k, color in zip(topks, saev.colors.ALL_RGB01[1:]):
                texts = []

                group = df.filter(
                    (pl.col("config/sae/activation/key") == "top-k")
                    & (pl.col("config/sae/reinit_blend") == 0.8)
                    & (pl.col("config/val_data/layer") == layer)
                    & (pl.col("config/sae/activation/aux/key") == "auxk")
                    & (pl.col("config/sae/activation/top_k") == k)
                    & (pl.col("data_key") == "FishVista (Img)")
                    & (pl.col("cls/cfg/patch_agg") == "max")
                    & (pl.col("cls/cfg/target_col") == "habitat")
                    & (pl.col(x_col).is_not_null())
                )
                group = group.sort(by=x_col).with_columns(
                    pl
                    .col("cls/classifier")
                    .map_elements(
                        lambda clf: len(np.nonzero(clf.coef_)[0]), return_dtype=pl.Int32
                    )
                    .alias("cls/n_nonzero")
                )

                pareto = group.filter(pl.col("is_pareto"))
                if pareto.height == 0:
                    continue

                ids = pareto.get_column("id").to_list()
                xs = pareto.get_column("cls/n_nonzero").to_numpy()
                ys = pareto.get_column(y_col).to_numpy()

                line, *_ = ax.plot(
                    xs,
                    ys,
                    alpha=0.5,
                    label=f"TopK=${k}$",
                    color=color,
                    marker="o",
                )

                if i in (3, 4, 5):
                    ax.set_xlabel("L$_0$ ($\\downarrow$)")

                if i in (0, 3, 6, 9):
                    ax.set_ylabel("Test Acc. ($\\uparrow$)")

                if i in (2,):
                    ax.legend()

                ax.grid(True, linewidth=0.3, alpha=0.7)
                ax.spines[["right", "top"]].set_visible(False)
                ax.set_xscale("log")
                # ax.set_ylim(0.5, 0.8)
                ax.set_title(f"Layer {layer + 1}/24")

        fig.savefig("contrib/trait_discovery/docs/assets/fishbase-pareto-mse.pdf")
        # return fig

        return mo.vstack([fig, dict(pareto_ckpts)])

    _(df)
    return


if __name__ == "__main__":
    app.run()
