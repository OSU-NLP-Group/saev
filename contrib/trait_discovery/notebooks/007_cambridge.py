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
    WANDB_TAGS = [
        "cambridge-butterflies-256p",
        "cambridge-butterflies-384p",
        "cambridge-butterflies-512p",
        "cambridge-butterflies-640p",
    ]

    runs_root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
    shards_root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")
    return WANDB_PROJECT, WANDB_TAGS, WANDB_USERNAME, runs_root, shards_root


@app.cell
def _(
    WANDB_PROJECT,
    WANDB_TAGS,
    WANDB_USERNAME,
    beartype,
    concurrent,
    get_baseline_ce,
    get_cls_results,
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
    def _row_from_run(
        wandb_run,
    ) -> tuple[dict[str, object], list[dict[str, object]]] | None:
        saev_run = saev.disk.Run(runs_root / wandb_run.id)
        row = {"id": wandb_run.id}

        row.update(**{f"summary/{key}": value for key, value in wandb_run.summary.items()})

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

        row.update(
            **{f"config/train_data/{key}": value for key, value in train_data.items()}
        )
        row.update(**{f"config/val_data/{key}": value for key, value in val_data.items()})
        row.update(**{f"config/{key}": value for key, value in config.items()})

        metadata = row.get("config/train_data/metadata")
        assert metadata is not None, f"Run {wandb_run.id} missing metadata"

        row["model_key"] = get_model_key(metadata)
        row["data_key"] = get_data_key(metadata)

        split_map: dict[str, tuple[pathlib.Path, str, pathlib.Path]] = {}
        for metrics_fpath in get_inference_probe_metric_fpaths(saev_run.run_dir):
            shard_id = metrics_fpath.parent.name
            shards_dpath = shards_root / shard_id

            if not shards_dpath.exists():
                print(f"Skipping {wandb_run.id}: shards dir {shards_dpath} missing.")
                continue

            split_label = get_probe_split_label(shards_dpath)
            if split_label is None:
                print(f"Skipping shards {shard_id}: unknown split (run {wandb_run.id}).")
                continue

            if split_label in split_map:
                print(f"Skipping {wandb_run.id}: duplicate {split_label} metrics.")
                split_map = {}
                break

            split_map[split_label] = (metrics_fpath, shard_id, shards_dpath)

        cls_results = get_cls_results(saev_run)

        if not split_map:
            print(f"Skipping {wandb_run.id}: no splits.")
            return row, cls_results

        if "train" not in split_map:
            print(f"Skipping {wandb_run.id}: missing train, got {split_map.keys()}.")
            return row, cls_results

        if "val" not in split_map:
            print(f"Skipping {wandb_run.id}: missing val, got {split_map.keys()}.")
            return row, cls_results

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
            saev_run.inference / val_shards / f"probe1d_metrics__train-{train_shards}.npz"
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
            "config/train_data/metadata/content_tokens_per_example",
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
        seen_ids = set()
        for tag in WANDB_TAGS:
            # Use native W&B tags (set via wandb.init(tags=...))
            runs = list(
                wandb.Api().runs(path=f"{WANDB_USERNAME}/{WANDB_PROJECT}", filters={"tags": tag})
            )
            for run in runs:
                if run.id not in seen_ids:
                    all_runs.append(run)
                    seen_ids.add(run.id)
        if not all_runs:
            raise ValueError("No runs found.")
        return all_runs


    @beartype.beartype
    def make_sae_df_parallel(n_workers: int = 16):
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
                row, _ = result
                rows.append(row)

        assert rows, "No valid runs."
        return _finalize_sae_df(rows)


    @beartype.beartype
    def make_clf_df_parallel(n_workers: int = 16):
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
                row, cls_results = result
                for cls_result in cls_results:
                    rows.append(
                        row | {f"cls/{key}": val for key, val in cls_result.items()}
                    )

        if not rows:
            return pl.DataFrame()
        return _finalize_clf_df(rows)


    sae_df = make_sae_df_parallel()
    clf_df = make_clf_df_parallel()
    return (sae_df,)


@app.cell
def _(sae_df):
    sae_df
    return


@app.cell
def _(collections, mo, np, pl, plt, sae_df, saev):
    def _(df: pl.DataFrame):
        x_col = "summary/eval/l0"
        y_col = "summary/eval/normalized_mse"
        k_col = "config/sae/activation/top_k"

        layers = [21, 23]
        n_patches_list = [256, 384, 512, 640]
        k_values = [16, 32, 64, 128, 256]
        data_key = "Butterflies (ImgSeg)"

        fig, axes = plt.subplots(
            figsize=(8, 6),
            nrows=2,
            ncols=2,
            dpi=300,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes = axes.flatten()

        colors = list(saev.colors.ALL_RGB01)
        rng = np.random.default_rng(seed=1)
        rng.shuffle(colors)

        pareto_ckpts = collections.defaultdict(list)

        for ax, n_patches in zip(axes, n_patches_list):
            for i, (layer, color, marker) in enumerate(
                zip(layers, colors, ("o", "^", "s", "x", "+", "o"))
            ):
                group = df.filter(
                    (pl.col("config/sae/activation/key") == "top-k")
                    & (pl.col("config/sae/reinit_blend") == 0.8)
                    & (pl.col("config/val_data/layer") == layer)
                    & (pl.col("config/sae/activation/aux/key") == "auxk")
                    & (pl.col("data_key") == data_key)
                    & (
                        pl.col("config/train_data/metadata/content_tokens_per_example")
                        == n_patches
                    )
                )
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
                        label=f"Layer {layer + 1}",
                        color=color,
                        marker=marker,
                        linestyle="-",
                    )
                    pareto_ckpts[(n_patches, layer, data_key)].extend(ids)
                    pareto_k_values = set(pareto.get_column(k_col).to_list())
                else:
                    pareto_k_values = set()

                # For k values without pareto points, plot best (lowest NMSE) as faded point
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

            ax.set_title(f"{n_patches} patches")

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_yscale("log")

        axes[0].set_ylabel("Normalized MSE")
        axes[2].set_ylabel("Normalized MSE")
        axes[2].set_xlabel("L$_0$")
        axes[3].set_xlabel("L$_0$")
        axes[1].legend()

        return mo.vstack([fig, dict(pareto_ckpts)])


    _(sae_df)
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
            metadata[key] for key in ("vit_ckpt", "model_ckpt", "ckpt") if key in metadata
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

        if isinstance(
            data_cfg, saev.data.datasets.ImgSegFolder
        ) and "cambridge-segfolder" in str(data_cfg.root):
            return "Butterflies (ImgSeg)"

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
def _(beartype, cloudpickle, collections, json, pathlib, saev, shards_root):
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
                # For butterflies we use the same shards for train/test, so no split label
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

                    results.append(header | ckpt)

                except Exception as err:
                    print(f"Failed to load {fpath}: {err}")
                    continue

        return results
    return (get_cls_results,)


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
