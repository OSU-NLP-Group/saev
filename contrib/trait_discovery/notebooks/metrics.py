import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import beartype
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    from jaxtyping import Float, jaxtyped

    return Float, beartype, jaxtyped, mo, np, pathlib, pl, torch


@app.cell
def _(pathlib):
    root = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/saev/acts/butterflies/g49kbj2j"
    )
    return (root,)


@app.cell
def _(pl, root, torch):
    obs = pl.read_parquet(root / "img_obs.parquet").with_columns(
        i=pl.int_range(pl.len()).alias("index")
    )

    x = torch.load(root / "img_acts.pt").numpy()
    return obs, x


@app.cell
def _(obs):
    target_map = {
        target: label
        for target, label in obs.select("target", "label").unique().iter_rows()
    }
    return (target_map,)


@app.cell
def _(Float, beartype, jaxtyped, mo, np, obs, percentiles, pl, x):
    @jaxtyped(typechecker=beartype.beartype)
    def get_precision(
        x_ns: Float[np.ndarray, "n_imgs d_sae"], obs: pl.DataFrame
    ) -> Float[np.ndarray, "n_classes d_sae"]:
        n_imgs, d_sae = x_ns.shape
        n_classes = obs.select("target").unique().height

        # purity[c, f] = #(images i with a_i,f > threshold AND Y_i = c) / #(images i with a_i,f > threshold)
        # O[n,d]: latent is "on"
        on_ns = x_ns > percentiles[None, :]
        denom_s = on_ns.sum(axis=0).astype(np.float32)

        # One-hot Y[n,c]
        labels = obs.get_column("target").to_numpy()
        y_nc = np.zeros((n_imgs, n_classes), dtype=np.int8)
        y_nc[labels >= 0, labels[labels >= 0]] = 1

        precision_cs = np.zeros((n_classes, d_sae), dtype=np.float32)
        for c_idx in mo.status.progress_bar(range(n_classes)):
            mask_n = labels == c_idx
            # Numerator: rows where Y==c, summed over images
            num_s = on_ns[mask_n, :].sum(axis=0).astype(np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                precision_cs[c_idx, :] = num_s / denom_s
        precision_cs[:, denom_s == 0] = -1
        return precision_cs

    prec = get_precision(x, obs)
    prec
    return (prec,)


@app.cell
def _(pl, prec, target_map):
    prec_df = pl.DataFrame([
        {"species": target_map[i], "precision": prec[i, latent], "latent": latent}
        for i, latent in enumerate(prec.argmax(axis=1))
    ])
    prec_df.filter(pl.col("species").str.contains("lat"))
    return (prec_df,)


@app.cell
def _(prec_df):
    print(" ".join(map(str, prec_df.get_column("latent").to_list())))
    return


@app.cell
def _(np, x):
    percentiles = np.quantile(x, 0.95, axis=0)
    percentiles
    return (percentiles,)


if __name__ == "__main__":
    app.run()
