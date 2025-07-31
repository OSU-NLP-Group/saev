import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import json
    import os

    import altair as alt
    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import wandb
    from jaxtyping import Float, jaxtyped

    return Float, alt, beartype, jaxtyped, json, mo, np, os, pl, plt, wandb


@app.cell
def __(mo):
    mo.md(
        """
        # SAE Metrics Explorer

        This notebook helps you analyze and compare SAE training runs from WandB.

        ## Setup Instructions

        1. Edit the configuration cell at the top to set your WandB username and project
        2. Make sure you have access to the original ViT activation shards
        3. Use the filters to narrow down which models to compare

        ## Troubleshooting

        - **Missing data error**: This notebook needs access to the original ViT activation shards
        - **No runs found**: Check your WandB username, project name, and tag filter
        """
    )
    return


@app.cell
def __():
    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "saev"
    return WANDB_PROJECT, WANDB_USERNAME


@app.cell
def __(mo):
    tag_input = mo.ui.text(value="classification-v1.0", label="Sweep Tag:")
    return (tag_input,)


@app.cell
def __(WANDB_PROJECT, WANDB_USERNAME, mo, tag_input):
    mo.vstack([
        mo.md(
            f"Look at [{WANDB_USERNAME}/{WANDB_PROJECT} on WandB](https://wandb.ai/{WANDB_USERNAME}/{WANDB_PROJECT}/table) to pick your tag."
        ),
        tag_input,
    ])
    return


@app.cell
def __(alt, df, mo, pl):
    chart = mo.ui.altair_chart(
        alt.Chart(
            df.filter(
                (pl.col("summary/_timestamp") > 1750809600)
                & (pl.col("summary/_runtime") > 300)
            ).select(
                "summary/eval/l0",
                "summary/eval/mse",
                "id",
                "config/objective/sparsity_coeff",
                "config/lr",
                "config/sae/d_sae",
                "model_key",
            )
        )
        .mark_point()
        .encode(
            x=alt.X("summary/eval/l0"),
            y=alt.Y("summary/eval/mse"),
            tooltip=["id", "config/lr"],
            color="config/lr:Q",
            # shape="config/objective/sparsity_coeff:N",
            shape="config/sae/d_sae:N",
            # shape="model_key",
        )
    )
    chart
    return (chart,)


@app.cell
def __(chart, df, mo, np, plot_dist, plt):
    mo.stop(
        len(chart.value) < 2,
        mo.md(
            "Select two or more points. Exactly one point is not supported because of a [Polars bug](https://github.com/pola-rs/polars/issues/19855)."
        ),
    )

    sub_df = (
        df.select(
            "id",
            "summary/eval/freqs",
            "summary/eval/mean_values",
            "summary/eval/l0",
        )
        .join(chart.value.select("id"), on="id", how="inner")
        .sort(by="summary/eval/l0")
        .head(4)
    )

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
    return (
        bins,
        freq_hist_ax,
        freqs,
        hist_axes,
        hist_fig,
        id,
        scatter_ax,
        scatter_axes,
        scatter_fig,
        sub_df,
        values,
        values_hist_ax,
    )


@app.cell
def __(scatter_fig):
    scatter_fig
    return


@app.cell
def __(hist_fig):
    hist_fig
    return


@app.cell
def __(chart, df, pl):
    df.join(chart.value.select("id"), on="id", how="inner").sort(
        by="summary/eval/l0"
    ).select("id", pl.selectors.starts_with("config/"))
    return


@app.cell
def __(Float, beartype, jaxtyped, np):
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
def __(
    beartype,
    get_data_key,
    get_model_key,
    json,
    load_freqs,
    load_mean_values,
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
    def find_metadata(shard_root: str):
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
        filters = {}
        if tag:
            filters["config.tag"] = tag
        runs = wandb.Api().runs(path="samuelstevens/saev", filters=filters)

        rows = []
        for run in mo.status.progress_bar(
            runs,
            remove_on_exit=True,
            title="Loading",
            subtitle="Parsing runs from WandB",
        ):
            row = {}
            row["id"] = run.id

            row.update(**{
                f"summary/{key}": value for key, value in run.summary.items()
            })
            try:
                row["summary/eval/freqs"] = load_freqs(run)
            except ValueError:
                print(f"Run {run.id} did not log eval/freqs.")
                continue
            except RuntimeError:
                print(f"Wandb blew up on run {run.id}.")
                continue
            try:
                row["summary/eval/mean_values"] = load_mean_values(run)
            except ValueError:
                print(f"Run {run.id} did not log eval/mean_values.")
                continue
            except RuntimeError:
                print(f"Wandb blew up on run {run.id}.")
                continue

            # config
            row.update(**{
                f"config/data/{key}": value
                for key, value in run.config.pop("data").items()
            })
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
                metadata = find_metadata(row["config/data/shard_root"])
            except MetadataAccessError as err:
                print(f"Bad run {run.id}: {err}")
                continue

            row["model_key"] = get_model_key(metadata)

            data_key = get_data_key(metadata)
            if data_key is None:
                print(f"Bad run {run.id}: unknown data.")
                continue
            row["data_key"] = data_key

            row["config/d_vit"] = metadata["d_vit"]
            rows.append(row)

        if not rows:
            raise ValueError("No runs found.")

        df = pl.DataFrame(rows).with_columns(
            (pl.col("config/sae/d_vit") * pl.col("config/sae/exp_factor")).alias(
                "config/sae/d_sae"
            )
        )
        return df

    df = make_df(tag_input.value)
    return MetadataAccessError, df, find_metadata, make_df


@app.cell
def __(beartype):
    @beartype.beartype
    def get_model_key(metadata: dict[str, object]) -> str:
        family = next(
            metadata[key] for key in ("vit_family", "model_family") if key in metadata
        )

        ckpt = next(
            metadata[key] for key in ("vit_ckpt", "model_ckpt") if key in metadata
        )

        if family == "dinov2" and ckpt == "dinov2_vitb14_reg":
            return "DINOv2 ViT-B/14 (reg)"
        if family == "clip" and ckpt == "ViT-B-16/openai":
            return "CLIP ViT-B/16"
        if family == "clip" and ckpt == "hf-hub:imageomics/bioclip":
            return "BioCLIP ViT-B/16"
        if family == "siglip" and ckpt == "hf-hub:timm/ViT-L-16-SigLIP2-256":
            return "SigLIP2 ViT-L/16"

        print(f"Unknown model: {(family, ckpt)}")
        return ckpt

    @beartype.beartype
    def get_data_key(metadata: dict[str, object]) -> str | None:
        if (
            "train_mini" in metadata["data"]["root"]
            and metadata["data"]["__cls__"] == "ImageFolder"
        ):
            return "iNat21"

        if "train" in metadata["data"] and "Imagenet" in metadata["data"]:
            return "ImageNet-1K"

        print(f"Unknown data: {metadata['data']}")
        return None

    return get_data_key, get_model_key


@app.cell
def __(Float, json, np, os):
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
def __(df):
    df.drop(
        "config/log_every",
        "config/slurm_acct",
        "config/device",
        "config/wandb_project",
        "config/track",
        "config/log_to",
        "config/ckpt_path",
    )
    return


if __name__ == "__main__":
    app.run()
