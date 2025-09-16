import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import torch

    return mo, np, pathlib, pl, plt, torch


@app.cell
def _(pathlib):
    root = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/saev/acts/butterflies/g49kbj2j/"
    )
    return (root,)


@app.cell
def _(pl, root):
    obs = pl.read_parquet(root / "img_obs.parquet").with_columns(
        i=pl.int_range(pl.len()).alias("index")
    )
    return (obs,)


@app.cell
def _(np, root, torch):
    def minmax_scale_cols(x, *, ignore_nan=False, clip=True):
        """
        Linearly scale each column of X to [0, 1].

        - ignore_nan=True: compute mins/maxes with NaNs ignored; NaNs stay NaN.
        - clip=True: clip numerical noise into [0, 1].
        """

        amin = np.nanmin(x, axis=0)
        amax = np.nanmax(x, axis=0)

        denom = amax - amin
        # Avoid divide-by-zero for constant columns
        safe = denom.copy()
        safe[safe == 0] = 1.0

        scaled = (x - amin) / safe
        # Force constant columns (amax == amin) to 0
        if np.any(denom == 0):
            scaled[:, denom == 0] = 0.0

        if clip:
            np.clip(scaled, 0.0, 1.0, out=scaled)

        return scaled

    x = torch.load(root / "img_acts.pt").numpy()
    # Scale each column so that it's 0-1.
    x = minmax_scale_cols(x)
    return (x,)


@app.cell
def _(mo, np, obs, pl, plt):
    def show_heatmap(x, query: str = "", n_latents: int = 800):
        # imgs = obs.sql(query).get_column("i").to_numpy()
        imgs = (
            pl.concat([
                obs.filter(pl.col("label").str.contains("cyrbia")).head(150),
                obs.filter(pl.col("label").str.contains("cythera")).head(150),
            ])
            .get_column("i")
            .to_numpy()
        )

        print(imgs.shape)

        if len(imgs) == 0:
            return mo.md("No imgs found for query.")
        latents = np.sort(np.flip(np.argsort(x[imgs].sum(axis=0)))[:n_latents])
        latents = list(range(600)) + [1281, 1731]
        print(x[imgs][:, latents].shape)

        fig, ax = plt.subplots(dpi=300, figsize=(8, 8))
        ax.imshow(x[imgs][:, latents])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Latents")
        ax.set_ylabel("Images")
        return fig

    return (show_heatmap,)


@app.cell
def _(mo):
    query_input = mo.ui.text_area(
        label="SQL Query",
        full_width=True,
        value="""select i from self where label like '%lativitta%' order by label, i limit 1000;""",
    )

    run_button = mo.ui.run_button(label="Show Image")
    return query_input, run_button


@app.cell
def _(query_input):
    query_input
    return


@app.cell
def _(run_button):
    run_button
    return


@app.cell
def _(mo, query_input, run_button, show_heatmap, x):
    mo.stop(not run_button.value)

    show_heatmap(x, query=query_input.value)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
