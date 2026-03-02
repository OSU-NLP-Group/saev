import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # Figures for "Towards Open-Ended Visual Scientific Discovery with Sparse Autoencoders"

    This notebook hardcodes run ids for reproducible figures.
    """)
    return


@app.cell
def _():
    import json
    import pathlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import wandb

    import saev.colors
    return json, mo, pathlib, pl, plt, saev, wandb


@app.cell
def _():
    SAE_PROJECT = "samuelstevens/saev"
    BASELINES_PROJECT = "samuelstevens/tdiscovery"
    NMSE_SHARD = "3e27794f"
    RUNS_ROOT_BY_PROJECT = {
        "saev": "/fs/ess/PAS2136/samuelstevens/saev/runs",
        "tdiscovery": "/fs/ess/PAS2136/samuelstevens/tdiscovery/saev/runs",
    }

    SAE_FRONTIER_RUN_IDS = {
        "sae_relu": [
            "o1p9wl76",  # l0=4.6951, mse_per_dim=1637.764365
            "4mlqkei5",  # l0=12.5161, mse_per_dim=1534.299025
            # "afyydka9",  # l0=15.8899, mse_per_dim=1476.179415
            "wrnz7h7h",  # l0=17.2249, mse_per_dim=1447.692099
            "xayzq0hf",  # l0=28.8599, mse_per_dim=1241.725317
            "0pdum8cq",  # l0=64.1767, mse_per_dim=669.398748
            "t1na9yxo",  # l0=95.4205, mse_per_dim=434.158730
            "extl56w1",  # l0=159.2812, mse_per_dim=395.304504
            "6iom0amk",  # l0=264.1799, mse_per_dim=390.610977
            "i1ujcsi6",  # l0=379.2087, mse_per_dim=304.383475
            # "ho86h0gp",  # l0=443.6865, mse_per_dim=278.592489
            "as770651",  # l0=446.2740, mse_per_dim=272.134337
            "yt2roil5",  # l0=1459.4689, mse_per_dim=89.008199
            "dt1y8m94",  # l0=1747.1838, mse_per_dim=0.952541
        ],
        "matryoshka_relu": [
            "lnleoyf6",  # l0=0.0000, mse_per_dim=1832.850685
            "ibt2fgta",  # l0=1.0016, mse_per_dim=1760.521387
            "6l12fjm9",  # l0=1.9635, mse_per_dim=1714.864075
            "5mv59srt",  # l0=2.9002, mse_per_dim=1679.079346
            # "rfic94if",  # l0=3.2941, mse_per_dim=1548.449883
            "t1vh0qy1",  # l0=3.3380, mse_per_dim=1547.279089
            "mccrm7u8",  # l0=6.5307, mse_per_dim=1083.430236
            "t88ez13w",  # l0=7.9325, mse_per_dim=993.290721
            # "fxcpfysr",  # l0=9.3983, mse_per_dim=991.013977
            "kd2pd8rs",  # l0=83.0466, mse_per_dim=757.653901
            "9drbwvhg",  # l0=164.6356, mse_per_dim=402.543121
            # "09srbijj",  # l0=165.9475, mse_per_dim=399.818371
            "1qynjykb",  # l0=214.8222, mse_per_dim=379.635062
            "0pz90ly4",  # l0=801.8717, mse_per_dim=81.512418
            # "ybm0jqi4",  # l0=827.4175, mse_per_dim=73.074294
            # "kn0f5a3v",  # l0=931.4901, mse_per_dim=29.285765
            "2pdk23cz",  # l0=940.7733, mse_per_dim=26.932229
            "9fn4l6rf",  # l0=1574.4723, mse_per_dim=8.880592
        ],
        "matryoshka_topk": [
            # Source: contrib/trait_discovery/sweeps/003_auxk/probe1d.py, in1k_run_ids[23].
            # AuxK-only subset for layer 23 (top_k in {16, 64, 256}).
            # main.tex Figure 2 (fig:mse-l0) uses final-layer DINOv3 ViT-L/16 activations, which is config layer=23.
            "flqkcam7",  # top_k=16, auxk
            "s3pqewz1",  # top_k=64, auxk
            "l8hooa3r",  # top_k=256, auxk
        ],
    }

    BASELINE_RUN_IDS = {
        "kmeans": [
            "myy5btgw",
        ],
        "pca": [
            "qmbo5jxw",
            "kwh4twl0",
            "za1xuhhn",
            "a1x1laxm",
            "unu6dbfb",
            "dzv7ha4u",
        ],
        "semi_nmf": [
            "lm51bf37",
            "em7hzdw0",
            "cmf1j0gd",
            "q6qtn8f6",
            "rv1wfbws",
            "k9sot7dd",
        ],
    }
    return (
        BASELINES_PROJECT,
        BASELINE_RUN_IDS,
        NMSE_SHARD,
        RUNS_ROOT_BY_PROJECT,
        SAE_FRONTIER_RUN_IDS,
        SAE_PROJECT,
    )


@app.cell
def _(
    BASELINES_PROJECT,
    BASELINE_RUN_IDS,
    NMSE_SHARD,
    RUNS_ROOT_BY_PROJECT,
    SAE_FRONTIER_RUN_IDS,
    SAE_PROJECT,
    json,
    pathlib,
    pl,
    wandb,
):
    api = wandb.Api()


    def get_metrics_fpath(run_id: str, project: str) -> pathlib.Path:
        runs_root = pathlib.Path(RUNS_ROOT_BY_PROJECT[project])
        return runs_root / run_id / "inference" / NMSE_SHARD / "metrics.json"


    def get_disk_metrics(run_id: str, project: str) -> dict[str, float] | None:
        metrics_fpath = get_metrics_fpath(run_id, project)
        if not metrics_fpath.exists():
            return None
        with open(metrics_fpath) as fd:
            metrics = json.load(fd)

        req = {
            "normalized_mse",
            "mse_per_dim",
            "mse_per_token",
            "baseline_mse_per_dim",
            "baseline_mse_per_token",
            "sse_recon",
            "sse_baseline",
            "n_tokens",
            "d_model",
            "n_elements",
        }
        missing = sorted(req.difference(metrics.keys()))
        if missing:
            return None

        return {
            "nmse": float(metrics["normalized_mse"]),
            "mse_per_dim": float(metrics["mse_per_dim"]),
        }


    rows: list[dict[str, object]] = []
    skipped_run_ids: list[str] = []

    for method, run_ids in SAE_FRONTIER_RUN_IDS.items():
        for run_id in run_ids:
            run = api.run(f"{SAE_PROJECT}/{run_id}")
            disk_metrics = get_disk_metrics(run_id, "saev")
            if disk_metrics is None:
                skipped_run_ids.append(run_id)
                continue
            rows.append(
                {
                    "id": run.id,
                    "method": method,
                    "l0": float(run.summary["eval/l0"]),
                    "mse_per_dim": disk_metrics["mse_per_dim"],
                    "nmse": disk_metrics["nmse"],
                    "layer": int(run.config["train_data"]["layer"]),
                }
            )

    for method, run_ids in BASELINE_RUN_IDS.items():
        for run_id in run_ids:
            run = api.run(f"{BASELINES_PROJECT}/{run_id}")
            disk_metrics = get_disk_metrics(run_id, "tdiscovery")
            if disk_metrics is None:
                skipped_run_ids.append(run_id)
                continue
            if method == "kmeans":
                l0 = 1.0
            else:
                l0 = float(run.config["k"])

            rows.append(
                {
                    "id": run.id,
                    "method": method,
                    "l0": l0,
                    "mse_per_dim": disk_metrics["mse_per_dim"],
                    "nmse": disk_metrics["nmse"],
                    "layer": int(run.config["train_data"]["layer"]),
                }
            )

    df = pl.DataFrame(rows).sort("method", "l0")
    df
    return df, skipped_run_ids


@app.cell
def _(df, mo, skipped_run_ids: list[str]):
    msg = f"Loaded {df.height} runs from disk metrics only (no wandb metric fallback)."
    if skipped_run_ids:
        msg = (
            msg
            + f" Skipped {len(skipped_run_ids)} runs with missing/old metrics schema: {', '.join(skipped_run_ids)}."
        )
    mo.md(msg)
    return


@app.cell
def _(df, pl, plt, saev):
    def plot_mse_l0_tradeoff(df):
        fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=300, layout="constrained")
        alpha = 0.7
        xticks = [1, 4, 16, 64, 256, 1024]
        csv_parts: list[pl.DataFrame] = []

        # Switch this between "nmse" and "mse_per_dim" for the y-axis.
        y_key = "nmse"
        y_label_by_key = {
            "nmse": "Normalized MSE ($\\downarrow$)",
            "mse_per_dim": "MSE Per Dim ($\\downarrow$)",
        }
        y_scale_by_key = {
            "nmse": "linear",
            "mse_per_dim": "log",
        }
        assert y_key in y_label_by_key, y_key

        method_style = {
            "kmeans": dict(
                color=saev.colors.SEA_RGB01,
                marker="x",
                linestyle="None",
                label="$k$-Means",
            ),
            "pca": dict(
                color=saev.colors.ORANGE_RGB01, marker="^", linestyle="-", label="PCA"
            ),
            "semi_nmf": dict(
                color=saev.colors.GOLD_RGB01,
                marker="D",
                linestyle="--",
                label="Semi-NMF",
            ),
            "sae_relu": dict(
                color=saev.colors.BLUE_RGB01,
                marker="^",
                linestyle="--",
                label="SAE (ReLU)",
            ),
            "matryoshka_relu": dict(
                color=saev.colors.SCARLET_RGB01,
                marker="s",
                linestyle="-.",
                label="Matryoshka (ReLU)",
            ),
            "matryoshka_topk": dict(
                color=saev.colors.BLACK_RGB01,
                marker="o",
                linestyle="-",
                label="Matryoshka (TopK)",
            ),
        }

        for method, style in method_style.items():
            sub_df = df.filter(
                (pl.col("method") == method)
                & pl.col(y_key).is_not_null()
                & (pl.col("l0") > 0)
            ).sort("l0")
            if sub_df.is_empty():
                continue
            csv_parts.append(sub_df)
            xs = sub_df["l0"].to_numpy()
            ys = sub_df[y_key].to_numpy()
            if style["linestyle"] == "None":
                ax.scatter(
                    xs,
                    ys,
                    color=style["color"],
                    marker=style["marker"],
                    s=64,
                    linewidth=2.5,
                    alpha=alpha,
                    label=style["label"],
                )
                continue

            ax.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                alpha=alpha,
                label=style["label"],
            )

        ax.set_xlabel("L0 ($\\downarrow$)")
        ax.set_ylabel(y_label_by_key[y_key])
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale(y_scale_by_key[y_key])
        ax.minorticks_off()
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
        ax.grid(True, linewidth=0.3, alpha=0.7)
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(frameon=True, ncol=1)
        csv_df = (
            pl.concat(csv_parts)
            .sort("method", "l0")
            .select("id", "layer", "method", "l0", "mse_per_dim", "nmse")
        )

        out_prefix = "contrib/trait_discovery/docs/assets/dinov3_vitl16_in1k_baselines"
        fig.savefig(f"{out_prefix}.pdf")
        csv_df.write_csv(f"{out_prefix}.csv")
        return fig


    plot_mse_l0_tradeoff(df)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
