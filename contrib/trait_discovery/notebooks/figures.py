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
    - `load_df` auto-scans run inference shards and adds shard-prefixed columns (for example `3e27794f/normalized_mse`, `3802cb66/mean_ap`).
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

    import beartype
    import marimo as mo
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import wandb
    from adjustText import adjust_text

    import saev.colors
    import saev.data
    import saev.disk

    return (
        adjust_text,
        beartype,
        dataclasses,
        json,
        mo,
        np,
        pathlib,
        pe,
        pl,
        plt,
        saev,
        wandb,
    )


@app.cell
def _():
    SAE_PROJECT = "samuelstevens/saev"
    BASELINES_PROJECT = "samuelstevens/tdiscovery"
    NMSE_SHARD = "3e27794f"
    SHARDS_ROOT = "/fs/scratch/PAS2136/samuelstevens/saev/shards"
    RUNS_ROOT_BY_PROJECT = {
        "saev": "/fs/ess/PAS2136/samuelstevens/saev/runs",
        "tdiscovery": "/fs/ess/PAS2136/samuelstevens/tdiscovery/saev/runs",
    }
    return BASELINES_PROJECT, NMSE_SHARD, RUNS_ROOT_BY_PROJECT, SAE_PROJECT, SHARDS_ROOT


@app.cell
def _(beartype, dataclasses):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class RunSpec:
        """Source specification for one method line in a figure."""

        method: str
        """What method we're using to do dictionary learning."""
        run_ids: list[str]
        """List of wandb run ids."""
        project: str = "saev"
        """Wandb project (`saev` or `tdiscovery`)."""
        l0_key: str | int | float = "eval/l0"
        """How to compute L0. Allowed patterns: `eval/<summary_key>` (for W&B summary keys like `eval/l0`), `config/<config_key>` (for top-level config keys like `config/k`), or a numeric constant (for fixed baselines like `1.0`)."""

    return (RunSpec,)


@app.cell
def _(
    BASELINES_PROJECT,
    RUNS_ROOT_BY_PROJECT,
    RunSpec,
    SAE_PROJECT,
    beartype,
    json,
    np,
    pathlib,
    pl,
    saev,
    wandb,
):
    api = wandb.Api()

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
    def add_shard_metrics(
        row: dict[str, object], shard: str, metrics_fpath: pathlib.Path
    ) -> None:
        if not metrics_fpath.exists():
            return

        try:
            with open(metrics_fpath) as fd:
                metrics = json.load(fd)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(metrics, dict):
            return

        for key, value in metrics.items():
            if not isinstance(value, int | float) or isinstance(value, bool):
                continue
            row[f"{shard}/{key}"] = float(value)

    @beartype.beartype
    def add_shard_probe_metrics(
        row: dict[str, object], shard: str, shard_dpath: pathlib.Path
    ) -> None:
        best_mean_ap = None
        for probe_fpath in shard_dpath.glob("probe1d_metrics__train-*.npz"):
            try:
                with np.load(probe_fpath) as fd:
                    if "ap" not in fd:
                        continue
                    mean_ap = float(fd["ap"].mean().item())
            except (OSError, ValueError):
                continue
            if best_mean_ap is None or mean_ap > best_mean_ap:
                best_mean_ap = mean_ap

        if best_mean_ap is not None:
            row[f"{shard}/mean_ap"] = best_mean_ap

    @beartype.beartype
    def load_df(specs: list[RunSpec]) -> tuple[pl.DataFrame, list[str]]:
        rows: list[dict[str, object]] = []
        skipped_run_ids: list[str] = []

        for spec in specs:
            run_project = SAE_PROJECT if spec.project == "saev" else BASELINES_PROJECT
            runs_root = pathlib.Path(RUNS_ROOT_BY_PROJECT[spec.project])

            for run_id in spec.run_ids:
                run_dpath = runs_root / run_id
                if not run_dpath.exists():
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

                row: dict[str, object] = {
                    "id": run.id,
                    "method": spec.method,
                    "l0": l0,
                    "layer": layer,
                }

                disk_run = saev.disk.Run(run_dpath)
                if disk_run.inference.exists():
                    for shard_dpath in sorted(disk_run.inference.iterdir()):
                        if not shard_dpath.is_dir():
                            continue
                        shard = shard_dpath.name
                        add_shard_metrics(row, shard, shard_dpath / "metrics.json")
                        add_shard_probe_metrics(row, shard, shard_dpath)

                rows.append(row)

        if not rows:
            return (
                pl.DataFrame(
                    schema={
                        "id": pl.String,
                        "method": pl.String,
                        "l0": pl.Float64,
                        "layer": pl.Int64,
                    }
                ),
                skipped_run_ids,
            )

        return pl.DataFrame(rows).sort("method", "l0"), skipped_run_ids

    return (load_df,)


@app.cell
def _(NMSE_SHARD, RunSpec, load_df, mo, pl, plt, saev):
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
        msg = (
            f"Loaded {df.height} runs from disk scans (all available inference shards)."
        )
        if skipped_run_ids:
            msg += (
                f" Skipped {len(skipped_run_ids)} runs with missing run path/l0/layer: "
                f"{', '.join(skipped_run_ids)}."
            )
        mo.output.append(mo.md(msg))

        fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=300, layout="constrained")
        alpha = 0.7
        xticks = [1, 4, 16, 64, 256, 1024]
        csv_parts: list[pl.DataFrame] = []

        # Switch this between "nmse" and "mse_per_dim" for the y-axis.
        y_key = "nmse"
        y_col_by_key = {
            "nmse": f"{NMSE_SHARD}/normalized_mse",
            "mse_per_dim": f"{NMSE_SHARD}/mse_per_dim",
        }
        y_label_by_key = {
            "nmse": "Normalized MSE ($\\downarrow$)",
            "mse_per_dim": "MSE Per Dim ($\\downarrow$)",
        }
        y_scale_by_key = {
            "nmse": "linear",
            "mse_per_dim": "log",
        }
        assert y_key in y_label_by_key, y_key
        y_col = y_col_by_key[y_key]

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
                & pl.col(y_col).is_not_null()
                & (pl.col("l0") > 0)
            ).sort("l0")
            if sub_df.is_empty():
                continue
            csv_parts.append(sub_df)
            xs = sub_df["l0"].to_numpy()
            ys = sub_df[y_col].to_numpy()
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
            .select(
                "id",
                "layer",
                "method",
                "l0",
                f"{NMSE_SHARD}/mse_per_dim",
                f"{NMSE_SHARD}/normalized_mse",
            )
        )
        out_prefix = "contrib/trait_discovery/docs/assets/dinov3_vitl16_in1k_baselines"
        fig.savefig(f"{out_prefix}.pdf")
        csv_df.write_csv(f"{out_prefix}.csv")

        return fig

    plot_mse_l0_tradeoff()
    return


@app.cell
def _(NMSE_SHARD, RunSpec, load_df, mo, pl, plt):
    def plot_layer_comparison():
        specs = [
            RunSpec(
                "Layer 14",
                [
                    "jsqj2arm",
                    "3hr3d3w0",
                    "lq18pdy9",
                    "kk60aru4",
                    "fh0jta0t",
                    "y8q60ohz",
                    "u7tz0xii",
                    "ag8agm56",
                    "ja4zp5kn",
                    "220r8j1q",
                    "fjravp6a",
                    "yhu9d2z9",
                    "fkl5sxba",
                ],
            ),
            RunSpec(
                "Layer 16",
                [
                    "fi7qafny",
                    "txmrh5nd",
                    "u33xj1ig",
                    "qtvsac3e",
                    "aq8vvjub",
                    "xfgouwrz",
                    "t56bqbvp",
                    "rsmrpkly",
                    "obctvep6",
                    "e9oeml82",
                    "ateood6p",
                    "god3i07b",
                    "na632sj5",
                    "rhjhoav8",
                ],
            ),
            RunSpec(
                "Layer 18",
                [
                    "di427rrs",
                    "xvcht4ti",
                    "edx9q34f",
                    "pn1f9cge",
                    "0r1iy3es",
                    "n7pv6rkj",
                    "k9i6zi1v",
                    "4rhpmk3f",
                    "syuerpif",
                    "x90r98th",
                    "egid27oa",
                    "jqx6qdxv",
                    "pevusep7",
                    "vrepu5ey",
                    "pfjnrjjq",
                    "7rg0o6tk",
                    "av2qk4oj",
                    "vkdu21ck",
                    "2o9yaiuo",
                    "8i936qx0",
                ],
            ),
            RunSpec(
                "Layer 20",
                [
                    "p5sppjgl",
                    "y6osup5x",
                    "yi5zik0k",
                    "aa30r3nm",
                    "sq1ccr13",
                    "0tj48gqd",
                    "c94z9ib1",
                    "7dr58kwn",
                    "2uqtzyv6",
                    "s96104bm",
                    "kbiotiaj",
                ],
            ),
            RunSpec(
                "Layer 22",
                [
                    "7w24prz9",
                    "uus4op1x",
                    "4cqe9fha",
                    "qcyausyf",
                    "i6pxw0q9",
                    "zyj9edre",
                    "jul72wj6",
                    "ophe0g6m",
                    "x7py290w",
                    "wakiyun9",
                    "71u6kzuq",
                    "t1ip1brk",
                    "ajvxj1a6",
                    "pz4up9fd",
                    "jlyegk4k",
                    "36al8yw7",
                    "n5b6p8du",
                    "9qywvc6q",
                    "a01f97t0",
                ],
            ),
            RunSpec(
                "Layer 24",
                [
                    "ibt2fgta",
                    "6l12fjm9",
                    "rfic94if",
                    "t1vh0qy1",
                    "mccrm7u8",
                    "t88ez13w",
                    "eosnewqp",
                    "fxcpfysr",
                    "kd2pd8rs",
                    "9drbwvhg",
                    "1qynjykb",
                    "0pz90ly4",
                    "ybm0jqi4",
                    "kn0f5a3v",
                    "2pdk23cz",
                    "3kkf33w6",
                    "9fn4l6rf",
                ],
            ),
        ]
        # Source: W&B query for tag `in1k-v0.4.1` (DINOv3 ViT-L, IN1K val shard `3e27794f`,
        # objective.n_prefixes present), then per-layer cumulative-min frontier in eval/mse
        # over eval/l0 with l0 >= 1.
        df, skipped_run_ids = load_df(specs)
        y_col = f"{NMSE_SHARD}/mse_per_dim"
        missing_metric_run_ids = (
            df
            .filter(
                pl.col("method").str.starts_with("Layer ") & pl.col(y_col).is_null()
            )
            .get_column("id")
            .to_list()
        )
        msg = f"Loaded {df.height} runs for layer comparison from disk shard scans."
        if skipped_run_ids:
            msg += (
                f" Skipped {len(skipped_run_ids)} runs: {', '.join(skipped_run_ids)}."
            )
        mo.output.append(mo.md(msg))
        if skipped_run_ids or missing_metric_run_ids:
            if missing_metric_run_ids:
                mo.output.append(
                    mo.md(
                        f"Missing `{y_col}` for {len(missing_metric_run_ids)} runs: {', '.join(missing_metric_run_ids)}."
                    )
                )
            mo.output.append(
                mo.md(
                    "Layer comparison requires complete disk metrics. Run `contrib/trait_discovery/sweeps/010_eccv/matryoshka_relu_layers_nmse.py` and rerun this cell."
                )
            )
            return None

        fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300, layout="constrained")
        alpha = 0.8
        csv_parts: list[pl.DataFrame] = []

        layer_values = [14, 16, 18, 20, 22, 24]
        cmap = plt.get_cmap("plasma")
        colors = [
            cmap(i / (len(layer_values) - 1))[:3] for i in range(len(layer_values))
        ]

        for layer, color in zip(layer_values, colors):
            method = f"Layer {layer}"
            sub_df = df.filter(
                (pl.col("method") == method)
                & pl.col(y_col).is_not_null()
                & (pl.col("l0") > 0)
            ).sort("l0")
            if sub_df.is_empty():
                continue

            csv_parts.append(sub_df)
            ax.plot(
                sub_df["l0"].to_numpy(),
                sub_df[y_col].to_numpy(),
                color=color,
                marker="o",
                alpha=alpha,
                label=method,
                clip_on=False,
            )

        ax.set_xlabel("L0 ($\\downarrow$)")
        ax.set_ylabel("MSE ($\\downarrow$)")
        ax.grid(True, linewidth=0.3, alpha=0.7)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e0)
        ax.set_ylim(1e0)
        ax.legend(loc="lower right")

        csv_df = (
            pl
            .concat(csv_parts)
            .sort("method", "l0")
            .select("id", "layer", "method", "l0", y_col)
        )
        out_prefix = "contrib/trait_discovery/docs/assets/dinov3_vitl_layers_in1k_sae"
        fig.savefig(f"{out_prefix}.pdf")
        csv_df.write_csv(f"{out_prefix}.csv")

        return fig

    plot_layer_comparison()
    return


@app.cell
def _(RunSpec, load_df, mo, pl, plt, saev):
    def plot_vit_layers_ade20k_map():
        specs = [
            RunSpec(
                "DINOv3 ViT-L/16",
                [
                    "ag8agm56",
                    "e9oeml82",
                    "vrepu5ey",
                    "0tj48gqd",
                    "71u6kzuq",
                    "extl56w1",
                ],
            ),
            RunSpec(
                "DINOv3 ViT-B/16",
                [
                    "8hyzbyht",
                    "knz7yndg",
                    "q6pg7hl9",
                    "zvx4qkov",
                    "jpnwfh3w",
                    "6crsj9gj",
                ],
            ),
            RunSpec(
                "DINOv3 ViT-S/16",
                [
                    "gcpbnr9n",
                    "pvtt26ky",
                    "5gjy7lwi",
                    "flfplqsa",
                    "jt45lucm",
                    "5ewxrjg4",
                ],
            ),
        ]
        # Source: W&B query for tag `in1k-v0.4.1`, then for each model/layer take the run
        # with the highest ADE20K val mean AP from inference/<probe_shard>/probe1d_metrics__train-*.npz.
        probe_col_by_method = {
            "DINOv3 ViT-L/16": "3802cb66/mean_ap",
            "DINOv3 ViT-B/16": "66a5d2c1/mean_ap",
            "DINOv3 ViT-S/16": "5e195bbf/mean_ap",
        }
        df, skipped_run_ids = load_df(specs)
        missing_probe_run_ids: list[str] = []
        for method, probe_col in probe_col_by_method.items():
            missing_probe_run_ids.extend(
                df
                .filter((pl.col("method") == method) & pl.col(probe_col).is_null())
                .get_column("id")
                .to_list()
            )

        msg = f"Loaded {df.height} runs for ViT-layer ADE20K mAP from disk shard scans."
        if skipped_run_ids:
            msg += (
                f" Skipped {len(skipped_run_ids)} runs: {', '.join(skipped_run_ids)}."
            )
        mo.output.append(mo.md(msg))
        if missing_probe_run_ids:
            mo.output.append(
                mo.md(
                    "This figure requires complete probe metrics on disk for all hardcoded runs."
                )
            )
            mo.output.append(
                mo.md(
                    f"Missing probe mAP columns for {len(missing_probe_run_ids)} runs: {', '.join(missing_probe_run_ids)}."
                )
            )
            return None

        fig, ax1 = plt.subplots(figsize=(4.5, 3.5), dpi=300, layout="constrained")

        handles = []
        for method, color, marker, linestyle, label in [
            (
                "DINOv3 ViT-B/16",
                saev.colors.ORANGE_RGB01,
                "s",
                "--",
                "ViT-B/16",
            ),
            (
                "DINOv3 ViT-S/16",
                saev.colors.BLUE_RGB01,
                "^",
                "-.",
                "ViT-S/16",
            ),
        ]:
            sub_df = df.filter(pl.col("method") == method).sort("layer")
            if sub_df.is_empty():
                continue
            xs = sub_df["layer"].to_numpy() + 1
            ys = sub_df[probe_col_by_method[method]].to_numpy()
            (handle,) = ax1.plot(
                xs,
                ys,
                color=color,
                alpha=0.8,
                marker=marker,
                linestyle=linestyle,
                label=label,
            )
            handles.append(handle)

        ax1.grid(True, linewidth=0.3, alpha=0.5)
        ax1.set_xlabel("ViT-{B,S}/16 Layer")
        ax1.set_ylabel("mAP ($\\uparrow$)")
        ax1.set_ylim(0, 0.5)
        ax1.set_xticks([7, 8, 9, 10, 11, 12])
        ax1.set_xlim(6.75, 12.25)

        ax2 = ax1.twiny()
        sub_df = df.filter(pl.col("method") == "DINOv3 ViT-L/16").sort("layer")
        if not sub_df.is_empty():
            xs = sub_df["layer"].to_numpy() + 1
            ys = sub_df[probe_col_by_method["DINOv3 ViT-L/16"]].to_numpy()
            (handle,) = ax2.plot(
                xs,
                ys,
                color=saev.colors.SEA_RGB01,
                alpha=0.8,
                marker="o",
                label="ViT-L/16",
            )
            handles.insert(0, handle)
        ax2.set_xlabel("ViT-L/16 Layer")
        ax2.set_xticks([14, 16, 18, 20, 22, 24])
        ax2.set_xlim(13.5, 24.5)

        ax1.legend(handles=handles, loc="upper left")

        csv_df = df.sort("method", "layer").select(
            "id",
            "method",
            "layer",
            probe_col_by_method["DINOv3 ViT-L/16"],
            probe_col_by_method["DINOv3 ViT-B/16"],
            probe_col_by_method["DINOv3 ViT-S/16"],
        )
        out_prefix = "contrib/trait_discovery/docs/assets/dinov3_in1k_ade20k_layers_map"
        fig.savefig(f"{out_prefix}.pdf")
        csv_df.write_csv(f"{out_prefix}.csv")

        return fig

    plot_vit_layers_ade20k_map()
    return


@app.cell
def _(
    RUNS_ROOT_BY_PROJECT,
    RunSpec,
    SHARDS_ROOT,
    adjust_text,
    load_df,
    mo,
    np,
    pathlib,
    pe,
    pl,
    plt,
    saev,
):
    fishvista_names = [
        "Background",
        "Head",
        "Eye",
        "Dorsal fin",
        "Pectoral fin",
        "Pelvic fin",
        "Anal fin",
        "Caudal fin",
        "Adipose fin",
        "Barbel",
    ]

    def plot_fishvista_prevalence_ap():
        from saev.data import Metadata

        specs = [
            RunSpec(
                "fishvista_topk_matryoshka",
                # Source: contrib/trait_discovery/sweeps/006_proposal_audit/cls_train.py,
                # fishvista_run_ids[23].
                ["pdikj9bl", "hfpct5ae", "s465wgg4", "dc86xg8z", "bpz34d80"],
            )
        ]
        probe_col = "8692dfa9/mean_ap"
        val_shard = "8692dfa9"
        df, skipped_run_ids = load_df(specs)
        if skipped_run_ids:
            msg = f"Skipped {len(skipped_run_ids)} runs with missing run path/l0/layer: {', '.join(skipped_run_ids)}."
            mo.output.append(mo.md(msg))

        candidate_df = df.filter(
            (pl.col("method") == "fishvista_topk_matryoshka")
            & pl.col(probe_col).is_not_null()
        ).sort(probe_col, descending=True)
        if candidate_df.is_empty():
            mo.output.append(
                mo.md(
                    f"No candidate runs with `{probe_col}`. Ensure probe metrics exist on disk."
                )
            )
            return None

        best = candidate_df.row(0, named=True)
        run_id = best["id"]
        best_mean_ap = float(best[probe_col])

        run = saev.disk.Run(pathlib.Path(RUNS_ROOT_BY_PROJECT["saev"]) / run_id)
        probe_fpaths = sorted(
            (run.inference / val_shard).glob("probe1d_metrics__train-*.npz")
        )
        if not probe_fpaths:
            mo.output.append(
                mo.md(f"Missing probe file for run `{run_id}` in shard `{val_shard}`.")
            )
            return None

        best_probe_fpath = None
        ap_c = None
        best_probe_map = -float("inf")
        for probe_fpath in probe_fpaths:
            with np.load(probe_fpath) as fd:
                if "ap" not in fd:
                    continue
                ap = fd["ap"]
            mean_ap = float(ap.mean().item())
            if mean_ap <= best_probe_map:
                continue
            best_probe_map = mean_ap
            best_probe_fpath = probe_fpath
            ap_c = ap

        if best_probe_fpath is None or ap_c is None:
            mo.output.append(
                mo.md(f"Probe files for run `{run_id}` do not contain AP vectors.")
            )
            return None

        train_shard = best_probe_fpath.stem.rsplit("train-", maxsplit=1)[-1]
        train_shards_dpath = pathlib.Path(SHARDS_ROOT) / train_shard
        if not train_shards_dpath.exists():
            mo.output.append(
                mo.md(f"Missing train shard directory `{train_shards_dpath}`.")
            )
            return None

        md = Metadata.load(train_shards_dpath)
        labels = np.memmap(
            train_shards_dpath / "labels.bin",
            mode="r",
            dtype=np.uint8,
            shape=(md.n_examples, md.content_tokens_per_example),
        ).reshape(-1)

        _, counts = np.unique(labels, return_counts=True, equal_nan=False, sorted=True)
        by_freq = np.argsort(counts)[::-1]
        xs = counts[by_freq]
        ys = ap_c[by_freq]

        fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300, layout="constrained")
        ax.scatter(xs, ys, alpha=0.8, color=np.array(saev.colors.ALL_RGB01)[by_freq])

        texts = []
        for x, y, class_i in zip(xs, ys, by_freq):
            texts.append(
                ax.text(
                    x,
                    y,
                    fishvista_names[class_i],
                    fontsize=11,
                    ha="center",
                    va="center",
                    color=saev.colors.ALL_RGB01[class_i],
                    path_effects=[
                        pe.withStroke(
                            linewidth=0.5,
                            foreground=saev.colors.BLACK_RGB01,
                            alpha=0.5,
                        )
                    ],
                )
            )

        ax.set_ylabel("Average Precision")
        ax.set_ylim(0, 1)
        ax.set_xscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Number of Samples")
        adjust_text(texts)

        csv_df = pl.DataFrame({
            "class_id": by_freq.tolist(),
            "class_name": [fishvista_names[i] for i in by_freq.tolist()],
            "n_samples": xs.tolist(),
            "average_precision": ys.tolist(),
            "run_id": [run_id] * len(by_freq),
            "train_probe_shard": [train_shard] * len(by_freq),
            "val_probe_shard": [val_shard] * len(by_freq),
        })
        out_prefix = (
            "contrib/trait_discovery/docs/assets/dinov3_fishvista_prevalence_ap"
        )
        fig.savefig(f"{out_prefix}.pdf")
        fig.savefig(
            "contrib/trait_discovery/docs/reports/eccv2026/figures/dinov3_fishvista_prevalence_ap.pdf"
        )
        csv_df.write_csv(f"{out_prefix}.csv")

        mo.output.append(
            mo.md(f"Selected run `{run_id}` with `{probe_col}={best_mean_ap:.6f}`.")
        )
        return fig

    plot_fishvista_prevalence_ap()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
