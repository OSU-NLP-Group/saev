import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import pathlib
    import pickle

    import beartype
    import matplotlib.pyplot as plt
    import numpy as np
    import pandera.polars as pa
    import polars as pl
    import marimo as mo
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
def _(
    beartype,
    get_baseline_ce,
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
        # From SAE training
        sae_val_mse: float = pa.Field(ge=0)
        sae_val_l0: float = pa.Field(ge=0)
        sae_val_l1: float = pa.Field(ge=0)

        probe_train_ce: float = pa.Field(ge=0)
        baseline_ce: float = pa.Field(ge=0)
        probe_r: float = pa.Field()


    @beartype.beartype
    @pa.check_types
    def load_probe_results_df() -> pa.typing.DataFrame[ProbeResultsSchema]:
        rows = []

        for probe_npz in mo.status.progress_bar(
            list(runs_root.glob("**/probe1d_metrics.npz"))
        ):
            *run_parts, _, shards_id, _ = probe_npz.parts

            if not (shards_root / shards_id).exists():
                continue

            run = saev.disk.Run(pathlib.Path(*run_parts))

            with np.load(probe_npz) as fd:
                loss = fd["loss"]

            probe_train_ce = loss.min(axis=0).mean().item()
            baseline_ce = get_baseline_ce(shards_root / shards_id).mean().item()

            wandb_run = get_wandb_run(run.run_id)

            rows.append(
                {
                    "run_id": run.run_id,
                    "sae_val_mse": wandb_run["summary/eval/mse"],
                    "sae_val_l0": wandb_run["summary/eval/l0"],
                    "sae_val_l1": wandb_run["summary/eval/l1"],
                    "probe_train_ce": probe_train_ce,
                    "baseline_ce": baseline_ce,
                    "probe_r": 1 - probe_train_ce / baseline_ce,
                }
            )

        return pl.DataFrame(rows)


    df = load_probe_results_df()
    df
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
