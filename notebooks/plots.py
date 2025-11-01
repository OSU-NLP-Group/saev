import marimo

__generated_with = "0.9.14"
app = marimo.App(
    width="full",
    css_file="/home/stevens.994/.config/marimo/custom.css",
)


@app.cell
def __():
    import json
    import os

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import torch
    import wandb
    from jaxtyping import Bool, Float, jaxtyped

    return (
        Bool,
        Float,
        beartype,
        jaxtyped,
        json,
        mo,
        np,
        os,
        pl,
        plt,
        torch,
        wandb,
    )


@app.cell
def __():
    # From https://colorbrewer2.org/#type=qualitative&scheme=Set2&n=8
    colors = [
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
    ]
    return (colors,)


@app.cell
def __(beartype):
    @beartype.beartype
    def get_model_key(metadata: dict[str, object]) -> str | None:
        if (
            metadata["model_org"] == "dinov2"
            and metadata["model_ckpt"] == "dinov2_vitb14_reg"
        ):
            return "DINOv2 ViT-B/14"
        if (
            metadata["model_org"] == "open-clip"
            and metadata["model_ckpt"] == "ViT-B-16/openai"
        ):
            return "CLIP ViT-B/16"

        print(f"Unknown model: {(metadata['model_org'], metadata['model_ckpt'])}")
        return None

    @beartype.beartype
    def get_data_key(metadata: dict[str, object]) -> str | None:
        if "train_mini" in metadata["data"] and "Inat21Dataset" in metadata["data"]:
            return "iNat21"

        if "train" in metadata["data"] and "Imagenet" in metadata["data"]:
            return "ImageNet-1K"

        print(f"Unknown data: {metadata['data']}")
        return None

    return get_data_key, get_model_key


@app.cell
def __(beartype):
    @beartype.beartype
    def to_human(num: float | int) -> str:
        prefix = "-" if num < 0 else ""

        num = abs(num)

        for i, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
            if num > i:
                return f"{prefix}{num / i:.1f}{suffix}"

        if num < 1:
            return f"{prefix}{num:.3g}"

        return f"{prefix}{num}"

    return (to_human,)


@app.cell
def __(get_data_key, get_model_key, json, os, pl, wandb):
    def make_df(runs):
        rows = []
        for run in runs:
            row = {}
            row["id"] = run.id
            row.update(**{
                f"summary/{key}": value for key, value in run.summary.items()
            })
            row.pop("summary/eval/hist", None)
            # config
            row.update(**{
                f"config/data/{key}": value for key, value in run.config["data"].items()
            })
            row.update(**{
                f"config/sae/{key}": value for key, value in run.config["sae"].items()
            })
            row.update(**{
                f"config/{key}": value
                for key, value in run.config.items()
                if key != "data"
            })

            with open(
                os.path.join(row["config/data/shard_root"], "metadata.json")
            ) as fd:
                metadata = json.load(fd)

            row["model_key"] = get_model_key(metadata)

            data_key = get_data_key(metadata)
            if data_key is None:
                print("Bad run: {run.id}")
                continue
            row["data_key"] = data_key

            row["config/d_vit"] = metadata["d_vit"]
            rows.append(row)

        df = pl.DataFrame(rows)
        return df

    tag = "baseline-v3.0"
    df = make_df(
        wandb.Api().runs(path="samuelstevens/saev", filters={"config.tag": tag})
    )
    df
    return df, make_df, tag


@app.cell
def __(df, pl):
    df.sort(by=pl.col("summary/losses/loss"), descending=False).group_by(
        pl.col("config/data/shard_root")
    ).first().select(
        "id",
        "config/data/shard_root",
        "config/sae/normalize_w_dec",
        "config/sae/remove_parallel_grads",
        "config/sae/n_reinit_samples",
        "summary/losses/loss",
        "summary/eval/n_dense",
        "summary/eval/n_dead",
    )
    return


@app.cell
def __(df, pl):
    df.sort(
        by=(pl.col("summary/eval/n_dense"), pl.col("summary/eval/n_dead")),
        descending=False,
    ).group_by(pl.col("config/data/shard_root")).first().select(
        "id",
        "config/data/shard_root",
        "config/sae/normalize_w_dec",
        "config/sae/remove_parallel_grads",
        "config/sae/n_reinit_samples",
        "summary/losses/loss",
        "summary/eval/n_dense",
        "summary/eval/n_dead",
    )
    return


@app.cell
def __(
    beartype,
    colors,
    df,
    is_pareto_efficient,
    np,
    pl,
    plt,
    tag,
    to_human,
):
    def plot_mse_l0_tradeoff(
        ax, df, *, add_labels: bool = False, plot_suboptimal: bool = True
    ):
        for width, color in zip(
            df.get_column("config/sae/d_sae").unique().sort(), colors
        ):
            df_w = df.filter(pl.col("config/sae/d_sae") == width)

            xs = df_w.get_column("summary/metrics/l0").to_numpy()
            ys = df_w.get_column("summary/losses/mse").to_numpy()

            i_sorted = np.argsort(xs)
            xs, ys = xs[i_sorted], ys[i_sorted]
            points = np.stack((xs, ys), axis=1)

            if len(points) > 1:
                mask = is_pareto_efficient(points)
                pareto_points = points[mask]

                other_points = points[~mask]
                if plot_suboptimal and other_points.size:
                    ax.scatter(
                        other_points[:, 0],
                        other_points[:, 1],
                        alpha=0.2,
                        color=color,
                    )
            else:
                pareto_points = points

            kwargs = dict(marker="o", color=color)
            if add_labels:
                kwargs["label"] = (f"Width {to_human(width)}",)
            ax.plot(pareto_points[:, 0], pareto_points[:, 1], **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.spines[["right", "top"]].set_visible(False)

    @beartype.beartype
    def plot_fig1(df):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            nrows=2, ncols=2, figsize=(5, 5), sharex=True, sharey=True
        )

        plot_mse_l0_tradeoff(
            ax1,
            df.filter(
                (pl.col("model_key") == "DINOv2 ViT-B/14")
                & (pl.col("data_key") == "ImageNet-1K")
            ),
        )
        ax1.set_title("DINOv2 on ImageNet-1K")

        plot_mse_l0_tradeoff(
            ax2,
            df.filter(
                (pl.col("model_key") == "CLIP ViT-B/16")
                & (pl.col("data_key") == "ImageNet-1K")
            ),
        )
        ax2.set_title("CLIP on ImageNet-1K")

        plot_mse_l0_tradeoff(
            ax3,
            df.filter(
                (pl.col("model_key") == "DINOv2 ViT-B/14")
                & (pl.col("data_key") == "iNat21")
            ),
        )
        ax3.set_title("DINOv2 on iNat21")

        plot_mse_l0_tradeoff(
            ax4,
            df.filter(
                (pl.col("model_key") == "CLIP ViT-B/16")
                & (pl.col("data_key") == "iNat21")
            ),
            add_labels=True,
        )
        ax4.set_title("CLIP on iNat21")
        ax4.legend(loc="best")

        ax1.set_ylabel("MSE")
        ax3.set_ylabel("MSE")

        ax3.set_xlabel("L0")
        ax4.set_xlabel("L0")

        fig.tight_layout()
        return fig

    fig_fpath = f"figure1-{tag}.pdf"
    fig = plot_fig1(df)
    # fig.savefig(fig_fpath, bbox_inches="tight")
    # mo.pdf(src=open(fig_fpath, "rb"))
    fig
    return fig, fig_fpath, plot_fig1, plot_mse_l0_tradeoff


@app.cell
def __(Array, Bool, Float, beartype, jaxtyped, np):
    @jaxtyped(typechecker=beartype.beartype)
    def is_pareto_efficient(points: Float[Array, "n 2"]) -> Bool[Array, " n"]:
        """ """
        # Sort points by x-value.
        i_sorted = np.argsort(points[:, 0])
        points = points[i_sorted]
        is_efficient = np.zeros(len(points), dtype=bool)
        min_y = np.inf
        for i, (x, y) in enumerate(points):
            if y < min_y:
                min_y = y
                is_efficient[i] = True

        # Un-sort is_efficient to match original points order.
        undo = np.zeros(len(points), dtype=int)
        undo[i_sorted] = np.arange(len(points))
        return is_efficient[undo]

    return (is_pareto_efficient,)


@app.cell
def __(np):
    from matplotlib import colormaps
    from PIL import Image

    data = np.linspace(0, 1, num=224 * 224).reshape((14 * 14, 16 * 16))
    colored_data = (colormaps.get_cmap("plasma")([0])[:, :, :3] * 256).astype(
        np.uint8
    )  # Convert to RGB uint8
    image = Image.fromarray(colored_data)
    image
    return Image, colored_data, colormaps, data, image


@app.cell
def __(colormaps, np):
    colormaps.get_cmap("plasma")(np.array(0))[:3]
    return


@app.cell
def __(colormaps, np):
    patches = np.array([0, 0.1, 2.0, 7.0, 0, 0, 0, 1])
    upper = patches.max()

    (colormaps.get_cmap("plasma")(patches / (upper + 1e-9))[:, :3] * 256).astype(
        np.uint8
    )
    return patches, upper


if __name__ == "__main__":
    app.run()
