import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # Figures for "Towards Open-Ended Visual Scientific Discovery with Sparse Autoencoders"

    This notebook hardcodes run ids for reproducible figures.

    ## Coding Style For This Notebook

    - Use `RunSpec` + `load_df(specs)` as the single data-loading interface.
    - Define run IDs inside each figure function/cell (local scope), not as global dicts.
    - Keep each figure self-contained:
      - create `specs: list[RunSpec]`
      - call `load_df(specs)`
      - make the plot
      - save figure artifacts (`.pdf`, `.csv`) to a fixed output path.
    - Do not create notebook-global loop variables like `method`; keep iteration inside figure functions.
    - Add short provenance comments next to run IDs (sweep file, tag, or paper figure reference) when relevant.
    - Prefer explicit, reproducible constants over dynamic discovery for publication figures.

    ## Typical Pattern

    1. Write `plot_<figure_name>()`.
    2. Inside it, define `specs = [RunSpec(...), ...]`.
    3. `df, skipped = load_df(specs)`.
    4. Plot from `df`, then write `csv_df` and save both `pdf` and `csv`.
    """)
    return


@app.cell
def _():
    import dataclasses
    import json
    import pathlib
    import typing as tp

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import wandb

    import saev.colors
    return beartype, dataclasses, json, mo, pathlib, pl, plt, saev, tp, wandb


@app.cell
def _():
    SAE_PROJECT = "samuelstevens/saev"
    BASELINES_PROJECT = "samuelstevens/tdiscovery"
    NMSE_SHARD = "3e27794f"
    RUNS_ROOT_BY_PROJECT = {
        "saev": "/fs/ess/PAS2136/samuelstevens/saev/runs",
        "tdiscovery": "/fs/ess/PAS2136/samuelstevens/tdiscovery/saev/runs",
    }
    return BASELINES_PROJECT, NMSE_SHARD, RUNS_ROOT_BY_PROJECT, SAE_PROJECT


@app.cell
def _(beartype, dataclasses, tp):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class RunSpec:
        """Source specification for one method line in a figure."""

        method: str
        """What method we're using to do dictionary learning."""
        run_ids: list[str]
        """List of wandb run ids."""
        project: tp.Literal['saev', 'tdiscovery'] = "saev"
        """Wandb project (`saev` or `tdiscovery`)."""
        l0_key: str | int | float = "eval/l0"
        """How to compute L0. Allowed patterns: `eval/<summary_key>` (for W&B summary keys like `eval/l0`), `config/<config_key>` (for top-level config keys like `config/k`), or a numeric constant (for fixed baselines like `1.0`)."""
    return (RunSpec,)


@app.cell
def _(
    BASELINES_PROJECT,
    NMSE_SHARD,
    RUNS_ROOT_BY_PROJECT,
    RunSpec,
    SAE_PROJECT,
    beartype,
    json,
    pathlib,
    pl,
    wandb,
):
    api = wandb.Api()

    @beartype.beartype
    def get_metrics_fpath(run_id: str, project: str) -> pathlib.Path:
        runs_root = pathlib.Path(RUNS_ROOT_BY_PROJECT[project])
        return runs_root / run_id / "inference" / NMSE_SHARD / "metrics.json"

    @beartype.beartype
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

    @beartype.beartype
    def get_l0(run: wandb.apis.public.Run, l0_key: str | int | float) -> float | None:
        if isinstance(l0_key, int | float) and not isinstance(l0_key, bool):
            return float(l0_key)

        if not isinstance(l0_key, str):
            return None

        if l0_key.startswith("config/"):
            config_key = l0_key.removeprefix("config/")
            if "/" in config_key:
                return None
            value = run.config.get(config_key)
        else:
            value = run.summary.get(l0_key)
            if value is None:
                value = run.config.get(l0_key)

        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        return None

    @beartype.beartype
    def load_df(specs: list[RunSpec]) -> tuple[pl.DataFrame, list[str]]:
        rows: list[dict[str, object]] = []
        skipped_run_ids: list[str] = []

        for spec in specs:
            run_project = SAE_PROJECT if spec.project == "saev" else BASELINES_PROJECT
            for run_id in spec.run_ids:
                disk_metrics = get_disk_metrics(run_id, spec.project)
                if disk_metrics is None:
                    skipped_run_ids.append(run_id)
                    continue

                run = api.run(f"{run_project}/{run_id}")
                l0 = get_l0(run, spec.l0_key)
                if l0 is None:
                    skipped_run_ids.append(run_id)
                    continue

                layer = run.config.get("train_data", {}).get("layer")
                if not isinstance(layer, int):
                    skipped_run_ids.append(run_id)
                    continue

                rows.append({
                    "id": run.id,
                    "method": spec.method,
                    "l0": l0,
                    "mse_per_dim": disk_metrics["mse_per_dim"],
                    "nmse": disk_metrics["nmse"],
                    "layer": layer,
                })

        if not rows:
            return (
                pl.DataFrame(
                    schema={
                        "id": pl.String,
                        "method": pl.String,
                        "l0": pl.Float64,
                        "mse_per_dim": pl.Float64,
                        "nmse": pl.Float64,
                        "layer": pl.Int64,
                    }
                ),
                skipped_run_ids,
            )

        return pl.DataFrame(rows).sort("method", "l0"), skipped_run_ids
    return (load_df,)


@app.cell
def _(RunSpec, load_df, mo, pl, plt, saev):
    def plot_mse_l0_tradeoff():
        specs = [
            RunSpec(
                "sae_relu",
                [
                    "o1p9wl76",
                    "4mlqkei5",
                    "wrnz7h7h",
                    "xayzq0hf",
                    "0pdum8cq",
                    "t1na9yxo",
                    "extl56w1",
                    "6iom0amk",
                    "i1ujcsi6",
                    "as770651",
                    "yt2roil5",
                    "dt1y8m94",
                ],
            ),
            RunSpec(
                "matryoshka_relu",
                [
                    "lnleoyf6",
                    "ibt2fgta",
                    "6l12fjm9",
                    "5mv59srt",
                    "t1vh0qy1",
                    "mccrm7u8",
                    "t88ez13w",
                    "kd2pd8rs",
                    "9drbwvhg",
                    "1qynjykb",
                    "0pz90ly4",
                    "2pdk23cz",
                    "9fn4l6rf",
                ],
            ),
            RunSpec(
                "matryoshka_topk",
                # Source: contrib/trait_discovery/sweeps/003_auxk/probe1d.py, in1k_run_ids[23].
                # AuxK-only subset for layer 23 (top_k in {16, 64, 256}).
                ["flqkcam7", "s3pqewz1", "l8hooa3r"],
            ),
            RunSpec("kmeans", ["myy5btgw"], project="tdiscovery", l0_key=1.0),
            RunSpec(
                "pca",
                [
                    "qmbo5jxw",
                    "kwh4twl0",
                    "za1xuhhn",
                    "a1x1laxm",
                    "unu6dbfb",
                    "dzv7ha4u",
                ],
                project="tdiscovery",
                l0_key="config/k",
            ),
            RunSpec(
                "semi_nmf",
                [
                    "lm51bf37",
                    "em7hzdw0",
                    "cmf1j0gd",
                    "q6qtn8f6",
                    "rv1wfbws",
                    "k9sot7dd",
                ],
                project="tdiscovery",
                l0_key="config/k",
            ),
        ]
        df, skipped_run_ids = load_df(specs)
        msg = f"Loaded {df.height} runs from disk metrics only (no wandb metric fallback)."
        if skipped_run_ids:
            msg += f" Skipped {len(skipped_run_ids)} runs with missing metrics/l0: {', '.join(skipped_run_ids)}."
        mo.output.append(mo.md(msg))

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
            pl
            .concat(csv_parts)
            .sort("method", "l0")
            .select("id", "layer", "method", "l0", "mse_per_dim", "nmse")
        )
        out_prefix = "contrib/trait_discovery/docs/assets/dinov3_vitl16_in1k_baselines"
        fig.savefig(f"{out_prefix}.pdf")
        csv_df.write_csv(f"{out_prefix}.csv")

        return fig

    plot_mse_l0_tradeoff()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
