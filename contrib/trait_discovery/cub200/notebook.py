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
    import dataclasses
    import gzip
    import logging
    import os.path
    import typing

    import beartype
    import numpy as np
    import torch
    import tyro
    from jaxtyping import Bool, Float, Int, jaxtyped
    from torch import Tensor
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    import saev.data
    import saev.nn
    import saev.utils.scheduling
    from saev import helpers

    import cub200.data

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("baselines")
    return (
        Bool,
        Float,
        Int,
        Tensor,
        average_precision_score,
        beartype,
        cub200,
        dataclasses,
        gzip,
        helpers,
        jaxtyped,
        log_format,
        logger,
        logging,
        np,
        os,
        plt,
        precision_recall_curve,
        saev,
        torch,
        typing,
        tyro,
    )


@app.cell
def __(cub200):
    cub_root = "/fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder"
    test_y_true_NT = cub200.data.load_attrs(cub_root, is_train=False).numpy()
    return cub_root, test_y_true_NT


@app.cell
def __(gzip, np):
    def load_bin_gz(fpath: str):
        with gzip.open(fpath, "rb") as fd:
            return np.load(fd)


    test_y_pred_NT = load_bin_gz(
        "/users/PAS1576//samuelstevens/projects/saev/contrib/trait_discovery/data/randvec-k1024-scores_NT.bin.gz"
    )
    return load_bin_gz, test_y_pred_NT


@app.cell
def __(
    average_precision_score,
    plt,
    precision_recall_curve,
    test_y_pred_NT,
    test_y_true_NT,
):
    def show_pr_curve(y_true, y_score):
        """
        Plot a Precision–Recall curve
        """
        y_true = y_true.astype(int)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        fig, ax = plt.subplots(dpi=600)

        print("recall:", recall)
        print("precision:", precision.shape)

        ax.step(recall, precision, where="post", linewidth=1)
        ax.set_xlim([0.0, 1.00])
        ax.set_ylim([0.0, 1.00])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR curve (AP = {ap:.3f})")
        ax.spines[["top", "right"]].set_visible(False)

        return fig


    show_pr_curve(test_y_true_NT[:, 261], test_y_pred_NT[:, 261])
    return (show_pr_curve,)


@app.cell
def __(average_precision_score, np, test_y_pred_NT, test_y_true_NT):
    ap_T = np.array([average_precision_score(test_y_true_NT[:, i], test_y_pred_NT[:, i]) for i in range(312)])
    return (ap_T,)


@app.cell
def __(ap_T):
    ap_T.mean()
    return


@app.cell
def __(ap_T, plt):
    _ = plt.hist(ap_T)
    plt.show()
    return


@app.cell
def __(ap_T):
    ap_T.argsort()
    return


@app.cell
def __(test_y_true_NT):
    test_y_true_NT.shape
    return


@app.cell
def __(test_y_true_NT):
    test_y_true_NT.astype(int)
    return


@app.cell
def __(test_y_pred_NT):
    test_y_pred_NT
    return


@app.cell
def __(test_y_true_NT):
    test_y_true_NT[:, -1]
    return


if __name__ == "__main__":
    app.run()
