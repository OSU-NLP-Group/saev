import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import pathlib
    import pickle

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandera.polars as pa
    import polars as pl
    import wandb

    import saev.data
    import saev.disk
    return base64, beartype, mo, np, pa, pathlib, pickle, pl, plt, saev, wandb


@app.cell
def _(pathlib):
    runs_root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
    shards_root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")

    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "saev"
    return WANDB_PROJECT, WANDB_USERNAME, runs_root, shards_root


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
def _(
    beartype,
    get_baseline_ce,
    get_inference_probe_metric_fpaths,
    get_wandb_run,
    mo,
    np,
    pa,
    pathlib,
    pl,
    runs_root,
    saev,
    shards_root,
):
    class ProbeResultsSchema(pa.DataFrameModel):
        run_id: str

        model: str = pa.Field()
        layer: int = pa.Field(ge=0)

        # From SAE training
        sae_data: str = pa.Field()
        sae_val_mse: float = pa.Field(ge=0)
        sae_val_l0: float = pa.Field(ge=0)
        sae_val_l1: float = pa.Field(ge=0)

        # From probe1d
        probe_shards: str = pa.Field()
        probe_ce: float = pa.Field(ge=0)
        baseline_ce: float = pa.Field(ge=0)
        probe_r: float = pa.Field()


    @beartype.beartype
    @pa.check_types
    def load_probe_results_df() -> pa.typing.DataFrame[ProbeResultsSchema]:
        rows = []

        run_and_probe_pairs: list[tuple[saev.disk.Run, pathlib.Path]] = []
        for run_dpath in runs_root.iterdir():
            if not run_dpath.is_dir():
                continue

            probe_metric_fpaths = get_inference_probe_metric_fpaths(run_dpath)
            if not probe_metric_fpaths:
                continue

            run = saev.disk.Run(run_dpath)
            for probe_metrics_fpath in probe_metric_fpaths:
                run_and_probe_pairs.append((run, probe_metrics_fpath))

        n_probe_metrics = len(run_and_probe_pairs)
        print(f"Found {n_probe_metrics} probe1d_metrics.npz files.")
        for run, probe_metrics_fpath in mo.status.progress_bar(run_and_probe_pairs):
            shards_id = probe_metrics_fpath.parent.name

            if not (shards_root / shards_id).exists():
                continue

            with np.load(probe_metrics_fpath) as fd:
                loss = fd["loss"]

            probe_ce = loss.min(axis=0).mean().item()
            baseline_ce = get_baseline_ce(shards_root / shards_id).mean().item()

            wandb_run = get_wandb_run(run.run_id)

            rows.append(
                {
                    "run_id": run.run_id,
                    "model": wandb_run["model_key"],
                    "layer": wandb_run["config/val_data/layer"],
                    "sae_data": wandb_run["data_key"],
                    "sae_val_mse": wandb_run["summary/eval/mse"],
                    "sae_val_l0": wandb_run["summary/eval/l0"],
                    "sae_val_l1": wandb_run["summary/eval/l1"],
                    "probe_shards": shards_id,
                    "probe_ce": probe_ce,
                    "baseline_ce": baseline_ce,
                    "probe_r": 1 - probe_ce / baseline_ce,
                }
            )

        return pl.DataFrame(rows)


    df = load_probe_results_df()
    df
    return (df,)


@app.cell
def _(df, pl, plt):
    def _():
        filtered = df.filter(
            (pl.col("model") == "DINOv3 ViT-S/16") & (pl.col("probe_shards") == "781f8739")
        )

        fig, axes = plt.subplots(nrows=4, dpi=200, figsize=(6, 8), layout="constrained")
        for field, ax in zip(["sae_val_mse", "sae_val_l0", "sae_val_l1", "layer"], axes):
            xs = filtered.get_column(field).to_numpy()
            ys = filtered.get_column("probe_r").to_numpy()
            ax.scatter(xs, ys)

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            # ax.set_xscale("log")
            ax.set_xlabel(field)
            ax.set_ylabel("Probe R ($\\uparrow$)")

        return fig


    _()
    return


@app.cell
def _(df, pl, plt):
    def _():
        train = df.filter(
            (pl.col("model") == "DINOv3 ViT-S/16") & (pl.col("probe_shards") == "781f8739")
        )
        val = df.filter(
            (pl.col("model") == "DINOv3 ViT-S/16") & (pl.col("probe_shards") == "5e195bbf")
        )

        fig, (ax_ce, ax_r) = plt.subplots(
            nrows=2, dpi=200, figsize=(6, 8), layout="constrained"
        )

        # Plot cross entropy
        xs = train.get_column("probe_ce")
        ys = val.get_column("probe_ce")
        min_ce = min(xs.min(), ys.min())
        max_ce = max(xs.max(), ys.max())

        ax_ce.plot([min_ce, max_ce], [min_ce, max_ce], color="tab:red", alpha=0.3)
        ax_ce.fill_between(
            [min_ce, max_ce],
            [max_ce, max_ce],
            [min_ce, max_ce],
            alpha=0.3,
            color="tab:red",
            linewidth=0,
            label="Overfitting",
        )
        ax_ce.scatter(xs, ys, label="Probe CE")

        ax_ce.grid(True, linewidth=0.3, alpha=0.7)
        ax_ce.spines[["right", "top"]].set_visible(False)
        # ax.set_xscale("log")
        ax_ce.set_xlabel("Train CE ($\\downarrow$)")
        ax_ce.set_ylabel("Test CE ($\\downarrow$)")

        ax_ce.legend()

        xs = train.get_column("probe_r")
        ys = val.get_column("probe_r")
        min_r = min(xs.min(), ys.min())
        max_r = max(xs.max(), ys.max())

        ax_r.plot([min_r, max_r], [min_r, max_r], color="tab:red", alpha=0.2)
        ax_r.fill_between(
            [min_r, max_r],
            [min_r, min_r],
            [min_r, max_r],
            alpha=0.3,
            color="tab:red",
            linewidth=0,
            label="Overfitting",
        )
        ax_r.scatter(xs, ys, label="Probe R")

        ax_r.grid(True, linewidth=0.3, alpha=0.7)
        ax_r.spines[["right", "top"]].set_visible(False)
        # ax.set_xscale("log")
        ax_r.set_xlabel("Train R ($\\uparrow$)")
        ax_r.set_ylabel("Test R ($\\uparrow$)")

        ax_r.legend()

        fig.suptitle("Measuring Overfitting")

        return fig


    _()
    return


@app.cell
def _(df, np, pl, plt):
    def _():
        filtered = df.filter(
            (pl.col("model") == "DINOv3 ViT-S/16") & (pl.col("probe_shards") == "5e195bbf")
        ).sort(by="layer")

        fig, ax = plt.subplots(dpi=200, figsize=(6, 3), layout="constrained")

        xs = filtered.get_column("layer").to_numpy()
        ys = filtered.get_column("probe_r").to_numpy()
        line = np.polynomial.Polynomial.fit(xs, ys, deg=1)

        ax.scatter(xs, ys, label="Observed")
        m, b = line.coef

        ax.plot(
            *line.linspace(),
            color="tab:red",
            alpha=0.3,
            label=f"Best Fit (${m:.2f} x + {b:.2f}$)",
        )

        ax.grid(True, linewidth=0.3, alpha=0.7)
        ax.spines[["right", "top"]].set_visible(False)
        # ax.set_xscale("log")
        ax.set_xlabel("ViT-S/16 Layer")
        ax.set_ylabel("Probe R ($\\uparrow$)")
        ax.legend()

        return fig


    _()
    return


@app.cell
def _(mo):
    mo.md("""For a given layer, how does the MSE/L0 tradeoff correlate with probe $R$?""")
    return


@app.cell
def _(df, np, pl, plt):
    def _():
        fig, axes = plt.subplots(
            nrows=2, ncols=3, dpi=200, figsize=(8, 5), layout="constrained"
        )
        axes = axes.reshape(-1)

        layers = [6, 7, 8, 9, 10, 11]

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            filtered = df.filter(
                (pl.col("model") == "DINOv3 ViT-S/16")
                & (pl.col("probe_shards") == "5e195bbf")
                & (pl.col("layer") == layer)
            ).sort(by="layer")

            xs = filtered.get_column("sae_val_l0").to_numpy()
            ys = filtered.get_column("probe_r").to_numpy()

            ax.set_title(f"Layer {layer + 1}/12")
            ax.scatter(xs, ys)

            line = np.polynomial.Polynomial.fit(xs, ys, deg=2)
            ax.plot(*line.linspace(), color="tab:orange", alpha=0.3, label="$y=mx+b$")

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            # ax.set_xscale("log")

            if i in (3, 4, 5):
                ax.set_xlabel("L0")
            if i in (0, 3):
                ax.set_ylabel("Probe R ($\\uparrow$)")
            if i in (0,):
                ax.legend()

        return fig


    _()
    return


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
def _(
    WANDB_PROJECT,
    WANDB_USERNAME,
    base64,
    beartype,
    json,
    os,
    pickle,
    saev,
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
    def get_wandb_run(run_id: str):
        run = wandb.Api().run(f"{WANDB_USERNAME}/{WANDB_PROJECT}/{run_id}")

        row = {}
        row["id"] = run.id

        try:
            row.update(**{f"summary/{key}": value for key, value in run.summary.items()})
        except AttributeError as err:
            print(f"Run {run.id} has a problem in run.summary._json_dict: {err}")
            return {}

        # config
        row.update(
            **{
                f"config/train_data/{key}": value
                for key, value in run.config.pop("train_data").items()
            }
        )
        row.update(
            **{
                f"config/val_data/{key}": value
                for key, value in run.config.pop("val_data").items()
            }
        )
        row.update(
            **{f"config/sae/{key}": value for key, value in run.config.pop("sae").items()}
        )

        if "objective" in run.config:
            row.update(
                **{
                    f"config/objective/{key}": value
                    for key, value in run.config.pop("objective").items()
                }
            )

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

        print(f"Unknown data: {data_cfg}")
        return None
    return (get_wandb_run,)


@app.cell
def _(baseline_ce, np, plt, runs_root, saev):
    def _():
        runs = [
            "3hr3d3w0",
        ]

        fig, (ax1, ax2, ax3) = plt.subplots(
            dpi=200, layout="constrained", nrows=3, figsize=(12, 12)
        )

        for run in runs:
            run = saev.disk.Run(runs_root / run)

            probe_metrics = run.inference / "614861a0" / "probe1d_metrics.npz"
            if not probe_metrics.exists():
                continue

            with np.load(run.inference / "614861a0" / "probe1d_metrics.npz") as fd:
                loss = fd["loss"]
                biases = fd["biases"]
                weights = fd["weights"]
                # tp = fd["tp"]
                # tn = fd["tn"]
                # fp = fd["fp"]
                # fn = fd["fn"]

            best_i = np.argmin(loss, axis=0)

            ax1.hist(
                1 - loss[best_i, np.arange(151)] / baseline_ce,
                bins=np.linspace(0, 1, 101),
                label=f"{run.run_id} ({(1 - loss[best_i, np.arange(151)] / baseline_ce).mean().item():.3f})",
                alpha=0.3,
            )
            ax2.hist(
                biases[best_i, np.arange(151)],
                alpha=0.3,
                label=run.run_id,
                # bins=30,
                # bins=np.linspace(-100, 100, 101),
                bins=np.linspace(-20, 0, 51),
            )
            ax3.hist(
                weights[best_i, np.arange(151)],
                alpha=0.3,
                label=run.run_id,
                # bins=30,
                # bins=np.linspace(-3000, 3000, 101),
                bins=np.linspace(-0.5, 2.5, 51),
            )

        ax1.set_xlim(0, 1)
        ax1.set_xlabel("Variance Explained")
        ax1.set_ylabel("Count of 151 Classes")
        ax1.legend()
        ax1.spines[["top", "right"]].set_visible(False)

        ax2.set_title("Bias Term Distribution")
        # ax2.set_yscale("log")
        ax2.legend()
        ax2.spines[["top", "right"]].set_visible(False)

        ax3.set_title("Weight Term Distribution")
        # ax3.set_yscale("log")
        ax3.legend()
        ax3.spines[["top", "right"]].set_visible(False)

        return fig


    _()
    return


if __name__ == "__main__":
    app.run()
