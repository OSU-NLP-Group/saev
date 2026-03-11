import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import json
    import pathlib
    import pickle

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandera.polars as pa
    import polars as pl
    import wandb

    import saev.colors
    import saev.data
    import saev.disk

    return (
        base64,
        beartype,
        json,
        mo,
        np,
        pa,
        pathlib,
        pickle,
        pl,
        plt,
        saev,
        wandb,
    )


@app.cell
def _(pathlib):
    runs_root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/tdiscovery/saev/runs")
    shards_root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")

    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "tdiscovery"
    return WANDB_USERNAME, runs_root, shards_root


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

            # probe_metrics_fpath =
            for fpath in shard_dpath.glob("**/probe1d_metrics.npz"):
                probe_metric_fpaths.append(fpath)

        return probe_metric_fpaths

    return (get_inference_probe_metric_fpaths,)


@app.cell
def _(
    beartype,
    get_baseline_ce,
    get_inference_probe_metric_fpaths,
    get_probe_split_label,
    get_wandb_run,
    json,
    mo,
    mode,
    np,
    pa,
    pathlib,
    pl,
    runs_root,
    saev,
    shards_root,
):
    class BaselineProbeResultsSchema(pa.DataFrameModel):
        run_id: str

        model: str = pa.Field()
        layer: int = pa.Field(ge=0)
        method: str = pa.Field(isin=["k-means", "pca", "semi-nmf"])

        # From baselines::train
        fit_data: str = pa.Field()
        fit_val_mse: float = pa.Field(ge=0)
        fit_val_l0: float = pa.Field(ge=0)

        # From inference.py; normalized MSE on ADE20K
        task_train_nmse: float = pa.Field()
        task_val_nmse: float = pa.Field()

        # From probe1d.py
        frac_w_neg: float = pa.Field(ge=0)
        frac_best_w_neg: float = pa.Field(ge=0)
        # train
        train_probe_shards: str = pa.Field()
        train_probe_ce: float = pa.Field(ge=0)
        train_baseline_ce: float = pa.Field(ge=0)
        train_probe_r: float = pa.Field()
        # val
        val_probe_shards: str = pa.Field()
        val_probe_ce: float = pa.Field(ge=0)
        val_baseline_ce: float = pa.Field(ge=0)
        val_probe_r: float = pa.Field()

        # metrics
        val_mean_ap: float = pa.Field()
        val_mean_prec: float = pa.Field()
        val_mean_recall: float = pa.Field()
        val_mean_f1: float = pa.Field()
        val_mean_purity_16: float = pa.Field()
        cov_at_0_3: float = pa.Field(ge=0)
        cov_at_0_5: float = pa.Field(ge=0)
        cov_at_0_7: float = pa.Field(ge=0)

    @beartype.beartype
    @pa.check_types
    def load_baseline_probe_results_df() -> pa.typing.DataFrame[
        BaselineProbeResultsSchema
    ]:
        rows = []

        run_probe_fpaths: list[tuple[saev.disk.Run, list[pathlib.Path]]] = []
        n_probe_metrics = 0
        for run_dpath in runs_root.iterdir():
            if not run_dpath.is_dir():
                print(f"Skipping {run_dpath}: not a directory.")
                continue

            probe_fpaths = get_inference_probe_metric_fpaths(run_dpath)
            if not probe_fpaths:
                continue

            run = saev.disk.Run(run_dpath)
            run_probe_fpaths.append((run, probe_fpaths))
            n_probe_metrics += len(probe_fpaths)

        print(f"Found {n_probe_metrics} probe1d_metrics.npz files.")
        for run, probe_fpaths in mo.status.progress_bar(run_probe_fpaths):
            wandb_run = get_wandb_run(run.run_id, project="tdiscovery")
            if not wandb_run:
                print(f"Skipping {run.run_id}: no wandb run.")
                continue

            split_map: dict[str, tuple[pathlib.Path, str, pathlib.Path]] = {}
            for metrics_fpath in probe_fpaths:
                shard_id = metrics_fpath.parent.name
                shards_dpath = shards_root / shard_id

                if not shards_dpath.exists():
                    print(
                        f"Skipping run {run.run_id}: shards directory {shards_dpath} missing."
                    )
                    continue

                split_label = get_probe_split_label(shards_dpath)
                if split_label is None:
                    print(
                        f"Skipping shards {shard_id}: unknown split (run {run.run_id})."
                    )
                    continue

                if split_label in split_map:
                    print(
                        f"Skipping run {run.run_id}: duplicate {split_label} probe metrics discovered."
                    )
                    split_map = {}
                    break

                split_map[split_label] = (metrics_fpath, shard_id, shards_dpath)

            if not split_map:
                print(f"Skipping run {run.run_id}: no splits.")
                continue

            if "train" not in split_map or "val" not in split_map:
                print(
                    f"Skipping run {run.run_id}: expected train and val probe metrics, got {sorted(split_map.keys())}."
                )
                continue

            train_probe_metrics_fpath, train_shards, train_shards_dpath = split_map[
                "train"
            ]
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

            train_base_ce = get_baseline_ce(train_shards_dpath).mean().item()
            val_base_ce = get_baseline_ce(val_shards_dpath).mean().item()

            # Normalized MSE
            train_nmse = 1.0
            path = run.inference / train_shards / "metrics.json"
            if path.is_file():
                train_nmse = json.loads(path.read_text())["normalized_mse"]

            val_nmse = 1.0
            path = run.inference / val_shards / "metrics.json"
            if path.is_file():
                val_nmse = json.loads(path.read_text())["normalized_mse"]

            # Get mAP from /fs/ess/PAS2136/samuelstevens/saev/runs/knz7yndg/inference/66a5d2c1/avgprec__train-fd5781e8.npz
            taus = [0.3, 0.5, 0.7]
            mean_ap, mean_prec, mean_recall, mean_f1 = 0.0, 0.0, 0.0, 0.0
            cov = {f"cov_at_{tau}".replace(".", "_"): 0.0 for tau in taus}
            purity = 0.0

            k = 16
            path = (
                run.inference
                / val_shards
                / f"probe1d_metrics__train-{train_shards}.npz"
            )
            if path.is_file():
                with np.load(path) as fd:
                    ap_c = fd["ap"]
                    prec_c = fd["precision"]
                    recall_c = fd["recall"]
                    f1_c = fd["f1"]

                    if "top_labels" not in fd:
                        print(f"{run.run_id} missing 'top_labels'.")
                        continue

                    top_labels_dk = fd["top_labels"]

                mean_ap = ap_c.mean().item()
                mean_prec = prec_c.mean().item()
                mean_recall = recall_c.mean().item()
                mean_f1 = f1_c.mean().item()
                cov = {
                    f"cov_at_{tau}".replace(".", "_"): (ap_c > tau).mean().item()
                    for tau in taus
                }

                _, count = mode(top_labels_dk[best_i, :16], axis=1)
                purity = (count / k).mean().item()

            method_raw = wandb_run.get("config/method")
            method = None
            if method_raw == "kmeans":
                method = "k-means"
            elif method_raw in {"pca", "semi-nmf"}:
                method = method_raw

            if method is None:
                if "summary/eval/inertia" in wandb_run:
                    method = "k-means"
                elif "summary/eval/recon_mse" in wandb_run:
                    method = "pca"
                else:
                    raise ValueError(wandb_run)

            if method == "k-means":
                fit_val_mse = wandb_run["summary/eval/inertia"]
                fit_val_l0 = 1.0
            elif method in {"pca", "semi-nmf"}:
                fit_val_mse = wandb_run["summary/eval/recon_mse"]
                fit_val_l0 = wandb_run["config/k"]
            else:
                raise ValueError(method)

            rows.append({
                "run_id": run.run_id,
                "model": wandb_run["model_key"],
                "layer": wandb_run["config/val_data/layer"],
                "method": method,
                "fit_data": wandb_run["data_key"],
                "fit_val_mse": fit_val_mse,
                "fit_val_l0": fit_val_l0,
                "task_train_nmse": train_nmse,
                "task_val_nmse": val_nmse,
                "frac_w_neg": (w < 0).mean().item(),
                "frac_best_w_neg": (w[best_i, np.arange(n_classes)] < 0).mean().item(),
                "train_probe_shards": train_shards,
                "train_probe_ce": train_ce,
                "train_baseline_ce": train_base_ce,
                "train_probe_r": 1 - train_ce / train_base_ce,
                "val_probe_shards": val_shards,
                "val_probe_ce": val_ce,
                "val_baseline_ce": val_base_ce,
                "val_probe_r": 1 - val_ce / val_base_ce,
                "val_mean_ap": mean_ap,
                "val_mean_prec": mean_prec,
                "val_mean_recall": mean_recall,
                "val_mean_f1": mean_f1,
                f"val_mean_purity_{k}": purity,
                **cov,
            })

        return pl.DataFrame(rows)

    df = load_baseline_probe_results_df()
    df
    return (df,)


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("model") == "DINOv3 ViT-L/16")
        & (pl.col("val_probe_shards") == "3802cb66")
        & (pl.col("method") == "pca")
    ).sort(by="train_probe_r", descending=True).select(
        "run_id",
        "model",
        "layer",
        "train_probe_shards",
        "method",
        "fit_val_mse",
        "fit_val_l0",
        "train_probe_r",
        "val_probe_r",
        "task_val_nmse",
        "val_mean_ap",
        "val_mean_purity_16",
        "cov_at_0_3",
        "cov_at_0_5",
        "cov_at_0_7",
    )
    return


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("model") == "DINOv3 ViT-L/16")
        & (pl.col("val_probe_shards") == "3802cb66")
        & (pl.col("method") == "k-means")
    ).sort(by="train_probe_r", descending=True).select(
        "run_id",
        "model",
        "layer",
        "train_probe_shards",
        "method",
        "fit_val_mse",
        "fit_val_l0",
        "train_probe_r",
        "val_probe_r",
        "task_val_nmse",
        "val_mean_ap",
        "val_mean_purity_16",
        "cov_at_0_3",
        "cov_at_0_5",
        "cov_at_0_7",
    )
    return


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("model") == "DINOv3 ViT-L/16")
        & (pl.col("val_probe_shards") == "3802cb66")
        & (pl.col("method") == "semi-nmf")
    ).sort(by="train_probe_r", descending=True).select(
        "run_id",
        "model",
        "layer",
        "train_probe_shards",
        "method",
        "fit_val_mse",
        "fit_val_l0",
        "train_probe_r",
        "val_probe_r",
        "task_val_nmse",
        "val_mean_ap",
        "val_mean_purity_16",
        "cov_at_0_3",
        "cov_at_0_5",
        "cov_at_0_7",
    )
    return


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("model") == "DINOv3 ViT-L/16")
        & (pl.col("val_probe_shards") == "3802cb66")
        & (pl.col("method") == "pca")
        & (pl.col("layer") == 23)
    ).sort(by="train_probe_r", descending=True).select(
        "run_id",
        "fit_val_mse",
        "fit_val_l0",
        "train_probe_r",
        "val_probe_r",
        "task_val_nmse",
        "val_mean_ap",
        "val_mean_purity_16",
        "cov_at_0_3",
    )
    return


@app.cell
def _(
    get_baseline_ce,
    get_class_names,
    mode,
    np,
    runs_root,
    saev,
    shards_root,
):
    # Also need to generate some visuals for PCA & k-means. So for each of the ADE20K classes, we need to generate some examples.abs
    def _():
        # k-means on DINOv3 on ADE20K
        train_shards = "614861a0"
        val_shards = "3802cb66"
        # PCA
        run = saev.disk.Run(runs_root / "unu6dbfb")
        # k-means
        # run = saev.disk.Run(runs_root / "myy5btgw")

        with np.load(run.inference / train_shards / "probe1d_metrics.npz") as fd:
            train_loss = fd["loss"]

        with np.load(run.inference / val_shards / "probe1d_metrics.npz") as fd:
            val_loss = fd["loss"]

        assert train_loss.ndim == 2
        assert val_loss.ndim == 2
        assert train_loss.shape == val_loss.shape

        n_latents, n_classes = train_loss.shape

        best_i = np.argmin(train_loss, axis=0)
        train_ce = train_loss[best_i, np.arange(n_classes)]
        val_ce = val_loss[best_i, np.arange(n_classes)]

        train_base_ce = get_baseline_ce(shards_root / train_shards)
        val_base_ce = get_baseline_ce(shards_root / val_shards)

        path = run.inference / val_shards / f"probe1d_metrics__train-{train_shards}.npz"
        assert path.exists()

        k = 16

        with np.load(path) as fd:
            ap_c = fd["ap"]
            prec_c = fd["precision"]
            recall_c = fd["recall"]
            f1_c = fd["f1"]
            top_labels_dk = fd["top_labels"]

        _, counts = mode(top_labels_dk[best_i, :k], axis=1)
        print(f"Purity@{k}: {(counts / k).mean().item()}")

        class_names = get_class_names()
        for i, (latent, name) in enumerate(zip(best_i, class_names)):
            print(f"latent {latent} scores {ap_c[i].item():.3f} on class '{name}'")

        print("...")

        best_classes_i = np.argsort(ap_c)[::-1]
        for rank, class_i in enumerate(best_classes_i[:30]):
            latent = best_i[class_i]
            name = class_names[class_i]
            ap = ap_c[class_i].item()
            top_class, count = mode(top_labels_dk[latent, :k])
            print(
                f"#{rank + 1}: latent {latent} scores {ap:.3f} on class '{name}' with purity@{k} {count.item() / k:.3f} (for class {class_names[int(top_class.item())]})."
            )

    _()
    return


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
def _(WANDB_USERNAME, base64, beartype, json, os, pickle, saev, wandb):
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
    def get_wandb_run(run_id: str, *, project: str):
        run = wandb.Api().run(f"{WANDB_USERNAME}/{project}/{run_id}")

        row = {}
        row["id"] = run.id

        try:
            row.update(**{
                f"summary/{key}": value for key, value in run.summary.items()
            })
        except AttributeError as err:
            print(f"Run {run.id} has a problem in run.summary._json_dict: {err}")
            return {}

        # config
        row.update(**{
            f"config/train_data/{key}": value
            for key, value in run.config.pop("train_data").items()
        })
        row.update(**{
            f"config/val_data/{key}": value
            for key, value in run.config.pop("val_data").items()
        })
        if "sae" in run.config:
            row.update(**{
                f"config/sae/{key}": value
                for key, value in run.config.pop("sae").items()
            })

        if "objective" in run.config:
            row.update(**{
                f"config/objective/{key}": value
                for key, value in run.config.pop("objective").items()
            })

        row.update(**{f"config/{key}": value for key, value in run.config.items()})

        try:
            # Check if metadata is available in WandB config
            wandb_metadata = row.get("config/train_data/metadata")
            metadata = find_metadata(row["config/train_data/shards"], wandb_metadata)
        except MetadataAccessError as err:
            print(f"Bad run {run.id}: {err}")
            return {}

        row["model_key"] = get_model_key(metadata)

        data_key = get_data_key(metadata)
        if data_key is None:
            print(f"Bad run {run.id}: unknown data.")
            return {}

        row["data_key"] = data_key
        row["config/d_model"] = metadata["d_model"]

        return row

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

    return (get_wandb_run,)


@app.cell
def _(pathlib):
    def get_class_names():
        class_names = ["none"]
        path = pathlib.Path(
            "/fs/ess/PAS2136/samuelstevens/datasets/ADEChallengeData2016/objectInfo150.txt"
        )
        for line in path.read_text().split("\n")[1:]:
            if not line:
                continue
            i, _, _, _, name = line.split("\t")
            assert int(i) == len(class_names)
            class_names.append(name)

        return class_names

    get_class_names()
    return (get_class_names,)


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


@app.cell
def _(plt, saev):
    def _():
        fig, ax = plt.subplots(figsize=(3, 2), dpi=300, layout="constrained")
        # ax.barh([0, 1, 2, 3], [0.1, 0.15, 0.25, 0.5], color=saev.colors.CYAN_RGB01)
        # ax.barh([0, 1, 2, 3], [0.05, 0.05, 0.2, 0.7], color=saev.colors.BLUE_RGB01)
        ax.barh([0, 1, 2, 3], [0.15, 0.25, 0.25, 0.35], color=saev.colors.SEA_RGB01)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_xticks([])
        ax.set_yticks(
            [0, 1, 2, 3],
            ["N. nocturnus", "N. atherinoides", "N. texanus", "G. affinis"],
        )
        for label in ax.get_yticklabels():
            label.set_style("italic")
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        fig.patch.set_alpha(0.5)
        ax.patch.set_alpha(0.5)
        return fig

    _()
    return


if __name__ == "__main__":
    app.run()
