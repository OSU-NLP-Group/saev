import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    def add_cub200_to_path():
        import sys
        import os.path

        for path in sys.path:
            if "cub200" in path:
                mod_path = os.path.dirname(path)

        if mod_path not in sys.path:
            sys.path.insert(0, mod_path)


    add_cub200_to_path()
    return (add_cub200_to_path,)


@app.cell
def __():
    import gzip
    import json
    import os.path

    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl
    import sklearn.metrics

    import cub200
    return cub200, gzip, json, np, os, pl, plt, sklearn


@app.cell
def __(cub200):
    cub_root = "/fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder"
    test_y_true_NT = cub200.data.load_attrs(cub_root, is_train=False).numpy()
    n_test, n_traits = test_y_true_NT.shape
    return cub_root, n_test, n_traits, test_y_true_NT


@app.cell
def __(n_test, n_traits, np, sklearn, test_y_true_NT):
    const_ap_T = np.array([sklearn.metrics.average_precision_score(test_y_true_NT[:,i], np.zeros(n_test)) for i in range(n_traits)])
    const_ap_T.mean()
    return (const_ap_T,)


@app.cell
def __(gzip, json, mo, np, os, pl, sklearn, test_y_true_NT):
    def load_bin_gz(fpath: str):
        with gzip.open(fpath, "rb") as fd:
            return np.load(fd)


    def load_df(root: str) -> pl.DataFrame:
        rows = []
        for dname in mo.status.progress_bar(os.listdir(root)):
            cfg_fpath = os.path.join(root, dname, "config.json")
            bin_fpath = os.path.join(root, dname, "scores.bin.gz")
            if not os.path.isfile(cfg_fpath):
                print(f"Missing config.json in {dname}.")
                continue

            if not os.path.isfile(bin_fpath):
                print(f"Missing scores.bin.gz in {dname}.")
                continue

            with open(cfg_fpath) as fd:
                cfg = json.load(fd)

            if isinstance(cfg, str):
                cfg = json.loads(cfg)

            if cfg["n_train"] < 0:
                cfg["n_train"] = 5794

            test_y_pred_NT = load_bin_gz(bin_fpath)
            aps = [
                sklearn.metrics.average_precision_score(test_y_true_NT[:, i], test_y_pred_NT[:, i])
                for i in range(312)
            ]
            ap_T = np.array(aps)
            cfg["ap"] = ap_T.tolist()
            rows.append(cfg)

        schema = {
            "n_prototypes": pl.Int32,
            "n_train": pl.Int32,
            "ap": pl.Array(pl.Float32, (312,)),
        }
        return pl.DataFrame(rows, schema=schema)


    df = load_df(
        "/users/PAS1576//samuelstevens/projects/saev/contrib/trait_discovery/data"
    )
    df.with_columns(pl.col("ap").arr.median().alias("median"))
    return df, load_bin_gz, load_df


@app.cell
def __(df, np, pl, plt):
    def graph(df, show_std=True):    
        df = (
            df.explode("ap").group_by(["n_prototypes", "n_train"])
            .agg(
                pl.col("ap").mean().alias("mAP"),
                pl.col("ap").quantile(0.05).alias("5%"),
                pl.col("ap").quantile(0.95).alias("95%"),
                pl.col("ap").std().alias('std')
            )
            .sort(["n_prototypes", "n_train"])
        )

        fig, ax = plt.subplots(dpi=300, figsize=(8,5))

        n_train_vals = sorted(df.get_column('n_train').unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(n_train_vals)))
        markers = ["o", "s", "^", "v", "D", "P", "X", "*"]  # extend if needed

        for col, mk, n in zip(colors, markers, n_train_vals):
            sub = df.filter(pl.col('n_train') == n)

            if show_std:
                ax.errorbar(
                    sub.get_column("n_prototypes"),
                    sub.get_column("mAP"),
                    yerr=sub["std"],
                    marker=mk,
                    color=col,
                    linestyle="-",
                    linewidth=1.5,
                    capsize=3,
                    label=str(n),
                )
            else:
                ax.plot(
                    sub.get_column("n_prototypes"),
                    sub.get_column("mAP"),
                    marker=mk,
                    color=col,
                    label=str(n),
                    linewidth=1.5,
                )

        ax.set_xticks(sub.get_column("n_prototypes"))
        ax.set_xlabel("$n$ Prototypes")
        ax.set_ylabel("mAP")
        ax.legend(loc="center right", title='$n$ Train')
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig


    graph(df, False)
    return (graph,)


@app.cell
def __(df, pl):
    tbl = (
        df.explode("ap").group_by(["n_prototypes", "n_train"])
        .agg(
            pl.col("ap").mean().alias("mAP"),
            pl.col("ap").quantile(0.05).alias("5%"),
            pl.col("ap").quantile(0.95).alias("95%"),
        )
        .sort(["n_prototypes", "n_train"])
        .rename({"n_prototypes": "Prototypes", "n_train": "Train"})
        .to_pandas()
    )

    print(tbl.to_markdown(index=False, tablefmt="github"))
    return (tbl,)


if __name__ == "__main__":
    app.run()
