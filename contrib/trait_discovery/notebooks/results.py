import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    def add_src_to_path():
        import os.path
        import sys

        for path in sys.path:
            if os.path.join("contrib", "trait_discovery", "notebooks") in path:
                sys.path.insert(0, os.path.dirname(path))
                return

    add_src_to_path()

    import dataclasses
    import json
    import math
    import os.path
    import re

    import beartype
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    return beartype, dataclasses, json, math, np, os, pl, plt, re


@app.cell
def _(dataclasses):
    attributes_fpath = "/fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder/metadata/attributes/attributes.txt"

    @dataclasses.dataclass(frozen=True)
    class CubAttribute:
        idx: int
        name: str
        value: str

    def load_cub_attributes():
        attributes = []
        with open(attributes_fpath) as fd:
            for i, line in enumerate(fd):
                _, attr_raw = line.split()
                name, value = attr_raw.split("::")
                attribute = CubAttribute(i, name, value)
                attributes.append(attribute)

        return attributes

    attributes = load_cub_attributes()

    set([attr.value for attr in attributes])
    return


@app.cell
def _(json, mo, os, pl):
    def load_df(root: str, prefix: str = "") -> pl.DataFrame:
        rows = []
        for fname in mo.status.progress_bar(os.listdir(root)):
            if not fname.startswith(prefix):
                continue

            with open(os.path.join(root, fname)) as fd:
                results = json.load(fd)

            rows.extend(results)

        return pl.DataFrame(rows, infer_schema_length=None).unnest("extra")

    df = load_df(
        "/users/PAS1576/samuelstevens/projects/saev/contrib/trait_discovery/results",
        prefix="fishvista",
    ).filter(pl.col("layer") > 11)

    df.group_by([
        "n_prototypes",
        "n_train",
        "seed",
        "vit_family",
        "vit_ckpt",
        "layer",
        "method",
        "class_idx",
        "sae_ckpt",
    ]).agg(pl.col("average_precision").mean().alias("mAP")).sort(
        by="mAP", descending=True
    )
    # df
    return (df,)


@app.cell
def _(df, pl):
    df.filter(
        (
            pl.col("sae_ckpt")
            == "/fs/ess/PAS2136/samuelstevens/checkpoints/saev/dbo663go/sae.pt"
        )
        & (pl.col("class_idx") == 2)
        & (pl.col("n_train") >= 300)
    ).sort(by="average_precision", descending=True)
    return


@app.cell
def _(df, pl):
    df.filter(
        (
            pl.col("sae_ckpt")
            == "/fs/ess/PAS2136/samuelstevens/checkpoints/saev/6b18jnda/sae.pt"
        )
        & (pl.col("class_idx") == 9)
        & (pl.col("n_train") >= 300)
    ).sort(by="average_precision", descending=True)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    | Segmentation Class | Segmentation Name | Best Latent | mAP | Notes |
    |---|---|---|---|---|
    | 0 | Background/Body |
    | 1 | Head | 4 | 0.828 |
    | 2 | Eye | 24 | 0.814 |
    | 3 | Dorsal Fin | 5, 98 | ~0.55 | 98 looks good to me. 5 also hits the anal fin. |
    | 4 | Pectoral Fin | 13 | 0.747 |
    | 5 | Pelvic Fin | 32, 267 | ~0.44 | 32 sometimes hits the dorsal fin. 267 is only for fish with really "shallow" pelvic fins. |
    | 6 | Anal Fin | 26, 67, 112 | ~0.42 | 26 looks good to me. 67 sometimes fires for the dorsal fin. Same thing for 112. |
    | 7 | Caudal Fin (Tail) | 7 | 0.794 | 
    | 8 | Adipose Fin | | | 728, 406 (not listed) seem good to me. Why? |
    | 9 | Barbel | 990 | 0.112

    - 1635 looks like a great pelvic fin, but only for fish with really weak/small pelvic fins.
    """
    )
    return


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("method") == "random")
        & (pl.col("class_idx") == 7)
        & (pl.col("n_train") == 30_000)
        & (pl.col("n_prototypes") > 6000)
    )
    return


@app.cell
def _(df, pl):
    df.filter(
        (
            (
                pl.col("sae_ckpt")
                == "/fs/ess/PAS2136/samuelstevens/checkpoints/saev/6b18jnda/sae.pt"
            )
            | ((pl.col("method") == "random") & (pl.col("n_prototypes") == 8224))
        )
        & (pl.col("class_idx") == 1)
        & (pl.col("n_train") == 300)
    ).sort(by="average_precision", descending=True)
    return


@app.cell
def _(df, pl):
    print(
        " ".join(
            f"{i}"
            for i in df.filter(
                pl.col("sae_ckpt")
                == "/fs/ess/PAS2136/samuelstevens/checkpoints/saev/6b18jnda/sae.pt"
            )
            .get_column("best_prototype_idx")
            .unique()
            .to_list()
        )
    )
    return


@app.cell
def _(beartype, df, math, np, pl, plt, re):
    @beartype.beartype
    class Plotter:
        group_by: list[str] = [
            "n_prototypes",
            "n_train",
            "vit_family",
            "vit_ckpt",
            "layer",
            "method",
            "sae_ckpt",
            "lr",
            "wd",
        ]

        def __init__(self, df):
            self._df = df

            self._agg_df = (
                df.filter(~pl.col("class_idx").is_in([0, 9]))
                .group_by(self.group_by)
                .agg(
                    pl.col("average_precision").mean().alias("mAP"),
                    pl.col("average_precision").quantile(0.05).alias("5%"),
                    pl.col("average_precision").quantile(0.95).alias("95%"),
                    pl.col("average_precision").quantile(0.50).alias("median"),
                    pl.col("average_precision").std().alias("std"),
                )
                .sort(by="n_train")
            )

            # n prototypes -> color
            seen_n_prototypes = sorted(df.get_column("n_prototypes").unique())
            self.n_prototypes = {
                n: color
                for n, color in zip(
                    seen_n_prototypes,
                    plt.cm.viridis(np.linspace(0, 1, len(seen_n_prototypes))),
                )
            }

            # layer -> subplot
            self.seen_layers = sorted(df.get_column("layer").unique())
            n_cols = 3
            n_rows = math.ceil(len(self.seen_layers) / n_cols)
            fig, axes = plt.subplots(
                nrows=n_rows,
                ncols=n_cols,
                sharex=True,
                sharey=True,
                dpi=300,
                figsize=(8, 3 * n_rows),
                layout="constrained",
            )
            self.fig = fig
            self.axes = axes.reshape(-1)

            # method -> marker
            self.baseline_methods = {"random": "o"}
            self.sae_marker = "x"

            self.xlim = (10, 100_000)

        def plot(self, show_std: bool = False):
            for i, (ax, layer) in enumerate(zip(self.axes, self.seen_layers)):
                # Plot all baselines
                for n_prototypes in self.n_prototypes:
                    for method in self.baseline_methods:
                        self._baseline(
                            method,
                            ax,
                            layer=layer,
                            n_prototypes=n_prototypes,
                            show_std=show_std,
                        )

                    self._saes(
                        ax, layer=layer, n_prototypes=n_prototypes, show_std=show_std
                    )

                self._supervised(ax, show_std=show_std)

                ax.set_title(f"Layer {layer}")
                ax.set_xlabel("$n$ Train")
                ax.set_ylabel("mAP")
                ax.set_xscale("log")
                ax.set_xlim(*self.xlim)
                ax.spines[["top", "right"]].set_visible(False)
                # ax.set_ylim(0, 1)

            return self.fig, self.axes

        def _supervised(self, ax, *, show_std: bool):
            mAP = (
                self._agg_df.filter(
                    (pl.col("vit_family") == "dinov3")
                    & (pl.col("method") == "linear-clf")
                    & (pl.col("n_train") == -1)
                    & (pl.col("vit_ckpt").str.contains("vitl"))
                )
                .sort(by="mAP", descending=True)
                .item(0, "mAP")
            )

            ax.hlines(
                mAP,
                *self.xlim,
                label="Supervised Linear",
                color="tab:orange",
                alpha=0.5,
            )

        def _saes(self, ax, *, layer: int, n_prototypes: int, show_std: bool):
            df = self._agg_df.filter(
                (pl.col("n_prototypes") == n_prototypes)
                & (pl.col("layer") == layer)
                & (pl.col("vit_family") == "dinov3")
                & (pl.col("method") == "sae")
            )

            sae_ckpts = sorted(
                df.filter(pl.col("sae_ckpt").is_not_null())
                .get_column("sae_ckpt")
                .unique()
            )
            if not sae_ckpts:
                return

            for sae_ckpt in sae_ckpts:
                sub = df.filter(pl.col("sae_ckpt") == sae_ckpt)
                name = self._parse_sae_ckpt(sae_ckpt)

                if show_std:
                    ax.errorbar(
                        sub.get_column("n_train"),
                        sub.get_column("mAP"),
                        yerr=sub.get_column("std"),
                        color=self.n_prototypes[n_prototypes],
                        marker="s",
                        label=f"SAE({name}, {n_prototypes})",
                        alpha=0.8,
                        capsize=2.0,
                    )
                else:
                    ax.plot(
                        sub.get_column("n_train"),
                        sub.get_column("mAP"),
                        color=self.n_prototypes[n_prototypes],
                        marker="s",
                        label=f"SAE({name}, {n_prototypes})",
                        alpha=0.8,
                    )

            # ax.legend()

        def _parse_sae_ckpt(self, sae_ckpt: str) -> str:
            """/fs/ess/PAS2136/samuelstevens/checkpoints/saev/goztek1c/sae.pt -> goztek1c"""
            match = re.match(r".*/(.*?)/sae\.pt", sae_ckpt)
            if not match:
                raise ValueError(f"Cannot parse '{sae_ckpt}'")
            return match.group(1)

        def _baseline(
            self, method: str, ax, *, layer: int, n_prototypes: int, show_std: bool
        ):
            sub = self._agg_df.filter(
                (pl.col("n_prototypes") == n_prototypes)
                & (pl.col("layer") == layer)
                & (pl.col("vit_family") == "dinov3")
                & (pl.col("method") == method)
            )

            if sub.height == 0:
                return

            if show_std:
                ax.errorbar(
                    sub.get_column("n_train"),
                    sub.get_column("mAP"),
                    yerr=sub.get_column("std"),
                    color=self.n_prototypes[n_prototypes],
                    marker=self.baseline_methods[method],
                    label=f"{method}({n_prototypes})",
                    alpha=0.5,
                    capsize=2.0,
                )
            else:
                ax.plot(
                    sub.get_column("n_train"),
                    sub.get_column("mAP"),
                    color=self.n_prototypes[n_prototypes],
                    marker=self.baseline_methods[method],
                    label=f"{method}({n_prototypes})",
                    alpha=0.5,
                )

    fig, axes = Plotter(df).plot()
    fig
    return


@app.cell
def _(df, np, pl, plt):
    def grid_by_ntrain_nproto(df, *, metric_col="average_precision"):
        # Aggregate to per (n_prototypes, n_train, vit_family, layer)
        df = (
            df.filter(pl.col("n_train") > 0)
            .group_by(["n_prototypes", "n_train", "vit_family", "layer"])
            .agg(pl.col(metric_col).mean().alias("mAP"))
            .sort(["n_prototypes", "n_train", "vit_family", "layer"])
        )

        # Get unique sorted axis values
        proto_vals = sorted(df.get_column("n_prototypes").unique())
        train_vals = sorted(df.get_column("n_train").unique())
        layer_vals = sorted(df.get_column("layer").unique())

        # Normalize vit_family names and pick plotting order
        families = sorted(df.get_column("vit_family").unique())

        # Colors/markers per family (extend as needed)
        fam_markers = {fam: m for fam, m in zip(families, ["o", "s", "^", "D"])}
        fam_styles = {families[i]: "-" * (i + 1) for i in range(len(families))}

        colors = plt.cm.coolwarm(np.linspace(0, 1, len(families)))

        # Build figure
        nrows, ncols = len(proto_vals), len(train_vals)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(3 * ncols, 2.2 * nrows),
            sharex=True,
            sharey=True,
            dpi=300,
            layout="constrained",
            squeeze=False,
        )

        # For legend collection
        handles_labels = {}

        for i, n_proto in enumerate(proto_vals):
            for j, n_train in enumerate(train_vals):
                ax = axes[i, j]

                sub = df.filter(
                    (pl.col("n_prototypes") == n_proto) & (pl.col("n_train") == n_train)
                )

                # Draw each family
                for color, fam in zip(colors, families):
                    sub_f = sub.filter(pl.col("vit_family") == fam).sort("layer")
                    if not sub_f.height:
                        continue

                    (line,) = ax.plot(
                        sub_f.get_column("layer"),
                        sub_f.get_column("mAP"),
                        linestyle=fam_styles.get(fam, "-"),
                        marker=fam_markers.get(fam, "o"),
                        linewidth=1.5,
                        markersize=3.5,
                        alpha=0.5,
                        color=color,
                        label=fam,
                    )
                    handles_labels[fam] = line

                # Cosmetics
                ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
                ax.spines[["top", "right"]].set_visible(False)

                if i == 0:
                    ax.set_title(f"Train: {n_train}")
                if j == 0:
                    ax.set_ylabel(f"mAP\n(proto: {n_proto})")
                if i == nrows - 1:
                    ax.set_xlabel("ViT layer")

                ax.set_xticks(layer_vals)

                if i == 0 and j == 0:
                    ax.legend(frameon=True)

        return fig

    grid_by_ntrain_nproto(df)
    return


@app.cell
def _(df, pl):
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
        .drop("vit_ckpt", "std", "median")
        .to_pandas()
        .to_markdown(index=False, tablefmt="github")
    )
    return


if __name__ == "__main__":
    app.run()
