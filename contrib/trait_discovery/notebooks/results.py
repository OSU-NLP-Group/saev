import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    def add_src_to_path():
        import os.path
        import sys

        for path in sys.path:
            if os.path.join("contrib", "trait_discovery", "notebooks") in path:
                sys.path.insert(0, os.path.dirname(path))
                return

    add_src_to_path()

    import gzip
    import json
    import os.path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import sklearn.metrics

    from src.lib import cub200

    return add_src_to_path, cub200, gzip, json, np, os, pl, plt, sklearn


@app.cell
def __(json, mo, os, pl):
    def load_df(root: str) -> pl.DataFrame:
        rows = []
        for fname in mo.status.progress_bar(os.listdir(root)):
            fpath = os.path.join(root, fname)

            if not os.path.isfile(fpath):
                print(f"Missing {fpath}.")
                continue

            with open(fpath) as fd:
                results = json.load(fd)

            rows.extend(results)

        return pl.DataFrame(rows)

    df = load_df(
        "/users/PAS1576/samuelstevens/projects/saev/contrib/trait_discovery/results"
    )
    df.group_by([
        "n_prototypes",
        "n_train",
        "seed",
        "vit_family",
        "vit_ckpt",
        "layer",
    ]).agg(pl.col("average_precision").mean().alias("mAP")).sort(
        by="mAP", descending=True
    )
    # df
    return df, load_df


@app.cell
def __(df, np, pl, plt):
    def graph(df):
        df = (
            df.group_by(["n_prototypes", "n_train", "vit_family", "vit_ckpt", "layer"])
            .agg(
                pl.col("average_precision").mean().alias("mAP"),
                pl.col("average_precision").quantile(0.05).alias("5%"),
                pl.col("average_precision").quantile(0.95).alias("95%"),
                pl.col("average_precision").quantile(0.50).alias("median"),
                pl.col("average_precision").std().alias("std"),
            )
            .sort(["n_prototypes", "n_train"])
        )

        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            sharex=True,
            sharey=True,
            dpi=300,
            figsize=(6, 6),
            layout="constrained",
        )
        axes = axes.reshape(-1)

        n_prototype_vals = sorted(df.get_column("n_prototypes").unique())
        layer_vals = sorted(df.get_column("layer").unique())

        colors = plt.cm.viridis(np.linspace(0, 1, len(n_prototype_vals)))
        markers = ["o", "s", "^", "v", "D", "P", "X", "*"]  # extend if needed

        for layer, ax in zip(layer_vals, axes):
            for color, marker, n in zip(colors, markers, n_prototype_vals):
                sub = df.filter(
                    (pl.col("n_prototypes") == n) & (pl.col("layer") == layer)
                )

                ax.plot(
                    sub.get_column("n_train"),
                    sub.get_column("mAP"),
                    color=color,
                    marker=marker,
                    label=f"{n}",
                )
                ax.set_title(f"Layer {layer}")

                ax.set_xlabel("$n$ Train")
                ax.set_ylabel("mAP")

                ax.spines[["top", "right"]].set_visible(False)
                # ax.set_ylim(0, 1)
        ax.legend(title="$n$ Prototypes")

        return fig

    graph(df)
    return (graph,)


@app.cell
def __(df, pl):
    print(
        df.group_by(["n_prototypes", "n_train", "vit_family", "vit_ckpt", "layer"])
        .agg(
            pl.col("average_precision").mean().alias("mAP"),
            pl.col("average_precision").quantile(0.05).alias("5%"),
            pl.col("average_precision").quantile(0.95).alias("95%"),
            pl.col("average_precision").quantile(0.50).alias("median"),
            pl.col("average_precision").std().alias("std"),
        )
        .sort(["n_prototypes", "n_train", "layer"])
        .rename({"n_prototypes": "Prototypes", "n_train": "Train", "layer": "Layer"})
        .drop("vit_family", "vit_ckpt", "std", "median")
        .to_pandas()
        .to_markdown(index=False, tablefmt="github")
    )
    return


if __name__ == "__main__":
    app.run()
