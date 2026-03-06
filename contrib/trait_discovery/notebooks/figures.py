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
    - `load_df` auto-scans run inference shards and adds shard-prefixed columns.
      - Metrics JSON columns: `<val_shard>/<metric_name>` (for example `3e27794f/normalized_mse`).
      - Probe columns with explicit train-shard provenance: `<val_shard>/<metric_name>__train-<train_shard>` (for example `3802cb66/mean_ap__train-614861a0`).
      - Local probe summary on a shard: `<val_shard>/probe_r`.
    - Loader helpers should only add columns, they should not pick max/min runs or aggregate across shards.
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

    ## Table Pattern

    1. Write `<table_name>()`.
    2. Inside it, define `specs = [RunSpec(...), ...]`.
    3. `df, skipped = load_df(specs)`.
    4. Select rows with explicit shard columns (for example best by `3802cb66/train_probe_r__train-614861a0`).
    5. Return a tidy table `DataFrame` and save a CSV artifact for reproducibility.
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
    get_baseline_ce,
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
        row: dict[str, object],
        inference_dpath: pathlib.Path,
        shard: str,
        shard_dpath: pathlib.Path,
    ) -> None:
        loss_by_shard: dict[str, np.ndarray | None] = {}

        def get_loss(shard_id: str) -> np.ndarray | None:
            if shard_id in loss_by_shard:
                return loss_by_shard[shard_id]

            probe_fpath = inference_dpath / shard_id / "probe1d_metrics.npz"
            if not probe_fpath.exists():
                loss_by_shard[shard_id] = None
                return None
            try:
                with np.load(probe_fpath) as fd:
                    if "loss" not in fd:
                        loss_by_shard[shard_id] = None
                        return None
                    loss = fd["loss"]
            except (OSError, ValueError):
                loss_by_shard[shard_id] = None
                return None
            if loss.ndim != 2:
                loss_by_shard[shard_id] = None
                return None
            loss_by_shard[shard_id] = loss
            return loss

        val_loss = get_loss(shard)
        val_base_ce = get_baseline_ce(shard)
        if val_loss is not None and val_base_ce is not None:
            n_classes = val_loss.shape[1]
            best_i = np.argmin(val_loss, axis=0)
            val_ce = float(val_loss[best_i, np.arange(n_classes)].mean().item())
            row[f"{shard}/probe_r"] = 1 - val_ce / val_base_ce

        for probe_fpath in shard_dpath.glob("probe1d_metrics__train-*.npz"):
            train_shard = probe_fpath.stem.rsplit("train-", maxsplit=1)[-1]
            if not train_shard:
                continue
            try:
                with np.load(probe_fpath) as fd:
                    ap = fd["ap"] if "ap" in fd else None
                    top_labels = fd["top_labels"] if "top_labels" in fd else None
            except (OSError, ValueError):
                continue

            if isinstance(ap, np.ndarray) and ap.ndim == 1:
                row[f"{shard}/mean_ap__train-{train_shard}"] = float(ap.mean().item())
                row[f"{shard}/cov_at_0_3__train-{train_shard}"] = float(
                    (ap > 0.3).mean().item()
                )
                row[f"{shard}/cov_at_0_5__train-{train_shard}"] = float(
                    (ap > 0.5).mean().item()
                )
                row[f"{shard}/cov_at_0_7__train-{train_shard}"] = float(
                    (ap > 0.7).mean().item()
                )

            if val_loss is None:
                continue

            train_loss = get_loss(train_shard)
            if train_loss is None:
                continue
            if train_loss.shape != val_loss.shape:
                continue

            n_classes = train_loss.shape[1]
            best_i = np.argmin(train_loss, axis=0)
            train_ce = float(train_loss[best_i, np.arange(n_classes)].mean().item())
            val_ce = float(val_loss[best_i, np.arange(n_classes)].mean().item())

            train_base_ce = get_baseline_ce(train_shard)
            if train_base_ce is not None:
                row[f"{shard}/train_probe_r__train-{train_shard}"] = (
                    1 - train_ce / train_base_ce
                )
            if val_base_ce is not None:
                row[f"{shard}/val_probe_r__train-{train_shard}"] = (
                    1 - val_ce / val_base_ce
                )

            if not isinstance(top_labels, np.ndarray):
                continue
            if top_labels.ndim != 2:
                continue
            if top_labels.shape[0] <= int(best_i.max()):
                continue

            top_labels_ck = top_labels[best_i, :16]
            max_count_c = np.apply_along_axis(
                lambda labels_k: np.bincount(
                    labels_k.astype(np.int64), minlength=256
                ).max(),
                1,
                top_labels_ck,
            )
            row[f"{shard}/purity_16__train-{train_shard}"] = float(
                (max_count_c / 16).mean().item()
            )

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
                    "project": spec.project,
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
                        add_shard_probe_metrics(
                            row, disk_run.inference, shard, shard_dpath
                        )

                rows.append(row)

        if not rows:
            return (
                pl.DataFrame(
                    schema={
                        "id": pl.String,
                        "method": pl.String,
                        "project": pl.String,
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
            pl.concat(csv_parts)
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
            df.filter(
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
            pl.concat(csv_parts)
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
            "DINOv3 ViT-L/16": "3802cb66/mean_ap__train-614861a0",
            "DINOv3 ViT-B/16": "66a5d2c1/mean_ap__train-fd5781e8",
            "DINOv3 ViT-S/16": "5e195bbf/mean_ap__train-781f8739",
        }
        df, skipped_run_ids = load_df(specs)
        missing_probe_cols = [
            probe_col
            for probe_col in probe_col_by_method.values()
            if probe_col not in df.columns
        ]
        if missing_probe_cols:
            mo.output.append(
                mo.md(f"Missing probe columns: {', '.join(missing_probe_cols)}.")
            )
            return None
        missing_probe_run_ids: list[str] = []
        for method, probe_col in probe_col_by_method.items():
            missing_probe_run_ids.extend(
                df.filter((pl.col("method") == method) & pl.col(probe_col).is_null())
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
def _(SHARDS_ROOT, beartype, mo, np, pathlib):
    @mo.cache
    @beartype.beartype
    def get_baseline_ce(shard: str) -> float | None:
        from saev.data import Metadata

        shards_dpath = pathlib.Path(SHARDS_ROOT) / shard
        labels_fpath = shards_dpath / "labels.bin"
        if not labels_fpath.exists():
            return None

        md = Metadata.load(shards_dpath)
        labels = np.memmap(
            labels_fpath,
            mode="r",
            dtype=np.uint8,
            shape=(md.n_examples, md.content_tokens_per_example),
        ).reshape(-1)
        if labels.size == 0:
            return None

        n_classes = int(labels.max()) + 1
        counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
        prob = counts / counts.sum()
        prob = np.clip(prob, 1e-12, 1 - 1e-12)
        ce = -(prob * np.log(prob) + (1 - prob) * np.log(1 - prob))
        return float(ce.mean().item())

    return (get_baseline_ce,)


@app.cell
def _(
    RunSpec,
    load_df,
    pl,
):
    def sae_vs_baselines_table():
        specs = [
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
            RunSpec(
                "vanilla_sae",
                # Source: contrib/trait_discovery/sweeps/010_eccv/relu_nmse.py,
                # in1k_relu_run_ids for layer 23.
                [
                    "o1p9wl76",
                    "4mlqkei5",
                    "afyydka9",
                    "wrnz7h7h",
                    "xayzq0hf",
                    "0pdum8cq",
                    "rggc5m8n",
                    "kqlfguzz",
                    "uojvw5o4",
                    "t1na9yxo",
                    "extl56w1",
                    "6iom0amk",
                    "i1ujcsi6",
                    "ho86h0gp",
                    "as770651",
                    "yt2roil5",
                    "dt1y8m94",
                    "xu8n5209",
                    "p2ycew4h",
                    "e2jsvsbx",
                    "cjtedfwa",
                    "iy4mtm9y",
                    "l2yvdllc",
                    "99l40o12",
                ],
            ),
        ]

        df, _skipped = load_df(specs)
        cols_by_method = {
            "kmeans": {
                "sae_train_val_shard": "3e27794f",
                "probe_fit_val_shard": "3802cb66",
                "train_probe_r": "3802cb66/train_probe_r__train-614861a0",
                "val_probe_r": "3802cb66/val_probe_r__train-614861a0",
                "val_probe_map": "3802cb66/mean_ap__train-614861a0",
                "val_probe_purity_16": "3802cb66/purity_16__train-614861a0",
                "val_probe_cov_at_0_3": "3802cb66/cov_at_0_3__train-614861a0",
            },
            "pca": {
                "sae_train_val_shard": "3e27794f",
                "probe_fit_val_shard": "3802cb66",
                "train_probe_r": "3802cb66/train_probe_r__train-614861a0",
                "val_probe_r": "3802cb66/val_probe_r__train-614861a0",
                "val_probe_map": "3802cb66/mean_ap__train-614861a0",
                "val_probe_purity_16": "3802cb66/purity_16__train-614861a0",
                "val_probe_cov_at_0_3": "3802cb66/cov_at_0_3__train-614861a0",
            },
            "semi_nmf": {
                "sae_train_val_shard": "3e27794f",
                "probe_fit_val_shard": "3802cb66",
                "train_probe_r": "3802cb66/train_probe_r__train-614861a0",
                "val_probe_r": "3802cb66/val_probe_r__train-614861a0",
                "val_probe_map": "3802cb66/mean_ap__train-614861a0",
                "val_probe_purity_16": "3802cb66/purity_16__train-614861a0",
                "val_probe_cov_at_0_3": "3802cb66/cov_at_0_3__train-614861a0",
            },
            "vanilla_sae": {
                "sae_train_val_shard": "3e27794f",
                "probe_fit_val_shard": "3802cb66",
                "train_probe_r": "3802cb66/train_probe_r__train-614861a0",
                "val_probe_r": "3802cb66/val_probe_r__train-614861a0",
                "val_probe_map": "3802cb66/mean_ap__train-614861a0",
                "val_probe_purity_16": "3802cb66/purity_16__train-614861a0",
                "val_probe_cov_at_0_3": "3802cb66/cov_at_0_3__train-614861a0",
            },
        }

        def get_float(row: dict[str, object], key: str) -> float | None:
            if key not in row:
                return None
            value = row[key]
            if not isinstance(value, int | float) or isinstance(value, bool):
                return None
            return float(value)

        def get_shard_l0(row: dict[str, object], shard: str) -> float | None:
            shard_l0 = get_float(row, f"{shard}/l0")
            if shard_l0 is not None:
                return shard_l0
            return get_float(row, "l0")

        def make_null_row(method: str) -> dict[str, object]:
            return {
                "method": method,
                "run_id": None,
                "layer": None,
                "sae_train_val_nmse": None,
                "sae_train_val_l0": None,
                "probe_fit_val_nmse": None,
                "probe_fit_val_l0": None,
                "probe_fit_val_probe_r": None,
                "probe_fit_val_probe_map": None,
                "probe_fit_val_probe_purity_16": None,
                "probe_fit_val_probe_cov_at_0_3": None,
            }

        rows = []
        for method, cols in cols_by_method.items():
            train_probe_r_col = cols["train_probe_r"]
            if train_probe_r_col not in df.columns:
                rows.append(make_null_row(method))
                continue

            sub_df = df.filter(
                (pl.col("method") == method) & pl.col(train_probe_r_col).is_not_null()
            ).sort(train_probe_r_col, descending=True)
            if sub_df.is_empty():
                rows.append(make_null_row(method))
                continue

            best = sub_df.row(0, named=True)
            train_val_shard = cols["sae_train_val_shard"]
            probe_val_shard = cols["probe_fit_val_shard"]
            rows.append({
                "method": method,
                "run_id": best["id"] if "id" in best else None,
                "layer": int(best["layer"]) if "layer" in best else None,
                "sae_train_val_nmse": get_float(
                    best, f"{train_val_shard}/normalized_mse"
                ),
                "sae_train_val_l0": get_shard_l0(best, train_val_shard),
                "probe_fit_val_nmse": get_float(
                    best, f"{probe_val_shard}/normalized_mse"
                ),
                "probe_fit_val_l0": get_shard_l0(best, probe_val_shard),
                "probe_fit_val_probe_r": get_float(best, cols["val_probe_r"]),
                "probe_fit_val_probe_map": get_float(best, cols["val_probe_map"]),
                "probe_fit_val_probe_purity_16": get_float(
                    best, cols["val_probe_purity_16"]
                ),
                "probe_fit_val_probe_cov_at_0_3": get_float(
                    best, cols["val_probe_cov_at_0_3"]
                ),
            })

        table_df = pl.DataFrame(rows).sort("method")

        table_df.write_csv(
            "contrib/trait_discovery/docs/assets/sae_vs_baselines_table.csv"
        )
        return table_df

    sae_vs_baselines_table()
    return


@app.cell
def _(RunSpec, load_df, pl):
    def vit_size_table():
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
        cols_by_method = {
            "DINOv3 ViT-L/16": {
                "sae_train_val_shard": "3e27794f",
                "probe_fit_val_shard": "3802cb66",
                "train_probe_r": "3802cb66/train_probe_r__train-614861a0",
                "val_probe_r": "3802cb66/val_probe_r__train-614861a0",
                "val_probe_map": "3802cb66/mean_ap__train-614861a0",
                "val_probe_purity_16": "3802cb66/purity_16__train-614861a0",
                "val_probe_cov_at_0_3": "3802cb66/cov_at_0_3__train-614861a0",
            },
            "DINOv3 ViT-B/16": {
                "sae_train_val_shard": "8762551e",
                "probe_fit_val_shard": "66a5d2c1",
                "train_probe_r": "66a5d2c1/train_probe_r__train-fd5781e8",
                "val_probe_r": "66a5d2c1/val_probe_r__train-fd5781e8",
                "val_probe_map": "66a5d2c1/mean_ap__train-fd5781e8",
                "val_probe_purity_16": "66a5d2c1/purity_16__train-fd5781e8",
                "val_probe_cov_at_0_3": "66a5d2c1/cov_at_0_3__train-fd5781e8",
            },
            "DINOv3 ViT-S/16": {
                "sae_train_val_shard": "52ec5790",
                "probe_fit_val_shard": "5e195bbf",
                "train_probe_r": "5e195bbf/train_probe_r__train-781f8739",
                "val_probe_r": "5e195bbf/val_probe_r__train-781f8739",
                "val_probe_map": "5e195bbf/mean_ap__train-781f8739",
                "val_probe_purity_16": "5e195bbf/purity_16__train-781f8739",
                "val_probe_cov_at_0_3": "5e195bbf/cov_at_0_3__train-781f8739",
            },
        }

        df, _skipped = load_df(specs)

        def get_float(row: dict[str, object], key: str) -> float | None:
            if key not in row:
                return None
            value = row[key]
            if not isinstance(value, int | float) or isinstance(value, bool):
                return None
            return float(value)

        def get_shard_l0(row: dict[str, object], shard: str) -> float | None:
            shard_l0 = get_float(row, f"{shard}/l0")
            if shard_l0 is not None:
                return shard_l0
            return get_float(row, "l0")

        def make_null_row(method: str) -> dict[str, object]:
            return {
                "method": method,
                "run_id": None,
                "layer": None,
                "sae_train_val_nmse": None,
                "sae_train_val_l0": None,
                "probe_fit_val_nmse": None,
                "probe_fit_val_l0": None,
                "probe_fit_val_probe_r": None,
                "probe_fit_val_probe_map": None,
                "probe_fit_val_probe_purity_16": None,
                "probe_fit_val_probe_cov_at_0_3": None,
            }

        rows = []
        for method, cols in cols_by_method.items():
            train_probe_r_col = cols["train_probe_r"]
            if train_probe_r_col not in df.columns:
                rows.append(make_null_row(method))
                continue

            sub_df = df.filter(
                (pl.col("method") == method) & pl.col(train_probe_r_col).is_not_null()
            ).sort(train_probe_r_col, descending=True)
            if sub_df.is_empty():
                rows.append(make_null_row(method))
                continue

            best = sub_df.row(0, named=True)
            train_val_shard = cols["sae_train_val_shard"]
            probe_val_shard = cols["probe_fit_val_shard"]
            rows.append({
                "method": method,
                "run_id": best["id"] if "id" in best else None,
                "layer": int(best["layer"]) if "layer" in best else None,
                "sae_train_val_nmse": get_float(
                    best, f"{train_val_shard}/normalized_mse"
                ),
                "sae_train_val_l0": get_shard_l0(best, train_val_shard),
                "probe_fit_val_nmse": get_float(
                    best, f"{probe_val_shard}/normalized_mse"
                ),
                "probe_fit_val_l0": get_shard_l0(best, probe_val_shard),
                "probe_fit_val_probe_r": get_float(best, cols["val_probe_r"]),
                "probe_fit_val_probe_map": get_float(best, cols["val_probe_map"]),
                "probe_fit_val_probe_purity_16": get_float(
                    best, cols["val_probe_purity_16"]
                ),
                "probe_fit_val_probe_cov_at_0_3": get_float(
                    best, cols["val_probe_cov_at_0_3"]
                ),
            })

        table_df = pl.DataFrame(rows).sort("method")
        table_df.write_csv("contrib/trait_discovery/docs/assets/vit_size_table.csv")
        return table_df

    vit_size_table()
    return


@app.cell
def _(RunSpec, load_df, pl):
    def vit_family_table():
        specs = [
            RunSpec(
                "DINOv3 ViT-L/16 (Matryoshka+TopK)",
                # Source: contrib/trait_discovery/sweeps/010_eccv/matryoshka_topk_nmse.py
                ["flqkcam7", "s3pqewz1", "l8hooa3r"],
            ),
            RunSpec(
                "PE-core ViT-L/14 (Matryoshka+TopK)",
                # Source: contrib/trait_discovery/sweeps/009_pe_core/inference.py
                ["h4gy7fke", "ywydn3z5", "omk5qhxf", "f3a9b41q", "r69kzt74"],
            ),
        ]
        cols_by_method = {
            "DINOv3 ViT-L/16 (Matryoshka+TopK)": {
                "sae_train_val_shard": "3e27794f",
                "probe_fit_val_shard": "3802cb66",
                "train_probe_r": "3802cb66/train_probe_r__train-614861a0",
                "val_probe_r": "3802cb66/val_probe_r__train-614861a0",
                "val_probe_map": "3802cb66/mean_ap__train-614861a0",
                "val_probe_purity_16": "3802cb66/purity_16__train-614861a0",
                "val_probe_cov_at_0_3": "3802cb66/cov_at_0_3__train-614861a0",
            },
            "PE-core ViT-L/14 (Matryoshka+TopK)": {
                "sae_train_val_shard": "a7f78fe3",
                "probe_fit_val_shard": "80219cbf",
                "train_probe_r": "80219cbf/train_probe_r__train-fa2b7ff0",
                "val_probe_r": "80219cbf/val_probe_r__train-fa2b7ff0",
                "val_probe_map": "80219cbf/mean_ap__train-fa2b7ff0",
                "val_probe_purity_16": "80219cbf/purity_16__train-fa2b7ff0",
                "val_probe_cov_at_0_3": "80219cbf/cov_at_0_3__train-fa2b7ff0",
            },
        }

        df, _skipped = load_df(specs)

        def get_float(row: dict[str, object], key: str) -> float | None:
            if key not in row:
                return None
            value = row[key]
            if not isinstance(value, int | float) or isinstance(value, bool):
                return None
            return float(value)

        def get_shard_l0(row: dict[str, object], shard: str) -> float | None:
            shard_l0 = get_float(row, f"{shard}/l0")
            if shard_l0 is not None:
                return shard_l0
            return get_float(row, "l0")

        def make_null_row(method: str) -> dict[str, object]:
            return {
                "method": method,
                "run_id": None,
                "layer": None,
                "sae_train_val_nmse": None,
                "sae_train_val_l0": None,
                "probe_fit_val_nmse": None,
                "probe_fit_val_l0": None,
                "probe_fit_val_probe_r": None,
                "probe_fit_val_probe_map": None,
                "probe_fit_val_probe_purity_16": None,
                "probe_fit_val_probe_cov_at_0_3": None,
            }

        rows = []
        for method, cols in cols_by_method.items():
            train_probe_r_col = cols["train_probe_r"]
            if train_probe_r_col not in df.columns:
                rows.append(make_null_row(method))
                continue

            sub_df = df.filter(
                (pl.col("method") == method) & pl.col(train_probe_r_col).is_not_null()
            ).sort(train_probe_r_col, descending=True)
            if sub_df.is_empty():
                rows.append(make_null_row(method))
                continue

            best = sub_df.row(0, named=True)
            train_val_shard = cols["sae_train_val_shard"]
            probe_val_shard = cols["probe_fit_val_shard"]
            rows.append({
                "method": method,
                "run_id": best["id"] if "id" in best else None,
                "layer": int(best["layer"]) if "layer" in best else None,
                "sae_train_val_nmse": get_float(
                    best, f"{train_val_shard}/normalized_mse"
                ),
                "sae_train_val_l0": get_shard_l0(best, train_val_shard),
                "probe_fit_val_nmse": get_float(
                    best, f"{probe_val_shard}/normalized_mse"
                ),
                "probe_fit_val_l0": get_shard_l0(best, probe_val_shard),
                "probe_fit_val_probe_r": get_float(best, cols["val_probe_r"]),
                "probe_fit_val_probe_map": get_float(best, cols["val_probe_map"]),
                "probe_fit_val_probe_purity_16": get_float(
                    best, cols["val_probe_purity_16"]
                ),
                "probe_fit_val_probe_cov_at_0_3": get_float(
                    best, cols["val_probe_cov_at_0_3"]
                ),
            })

        table_df = pl.DataFrame(rows).sort("method")
        table_df.write_csv("contrib/trait_discovery/docs/assets/vit_family_table.csv")
        return table_df

    vit_family_table()
    return


@app.cell
def _(RunSpec, load_df, pl):
    def sae_variants_table():
        specs = [
            RunSpec(
                "vanilla_relu",
                # Source: contrib/trait_discovery/sweeps/010_eccv/relu_nmse.py
                [
                    "o1p9wl76",
                    "4mlqkei5",
                    "afyydka9",
                    "wrnz7h7h",
                    "xayzq0hf",
                    "0pdum8cq",
                    "rggc5m8n",
                    "kqlfguzz",
                    "uojvw5o4",
                    "t1na9yxo",
                    "extl56w1",
                    "6iom0amk",
                    "i1ujcsi6",
                    "ho86h0gp",
                    "as770651",
                    "yt2roil5",
                    "dt1y8m94",
                    "xu8n5209",
                    "p2ycew4h",
                    "e2jsvsbx",
                    "cjtedfwa",
                    "iy4mtm9y",
                    "l2yvdllc",
                    "99l40o12",
                ],
            ),
            RunSpec(
                "matryoshka_relu",
                # Source: contrib/trait_discovery/sweeps/010_eccv/matryoshka_relu_nmse.py
                [
                    "lnleoyf6",
                    "ibt2fgta",
                    "6l12fjm9",
                    "5mv59srt",
                    "rfic94if",
                    "t1vh0qy1",
                    "u3gj24az",
                    "mccrm7u8",
                    "q91eu62e",
                    "t88ez13w",
                    "eosnewqp",
                    "yfpdczj7",
                    "fxcpfysr",
                    "xg2vom0w",
                    "rqdpylmi",
                    "s3dxavbq",
                    "2zf86reb",
                    "1247ezti",
                    "kd2pd8rs",
                    "9drbwvhg",
                    "09srbijj",
                    "1qynjykb",
                    "02ors1ov",
                    "vxgyr2du",
                    "x6ho5md2",
                    "0pz90ly4",
                    "ybm0jqi4",
                    "kn0f5a3v",
                    "2pdk23cz",
                    "3kkf33w6",
                    "9fn4l6rf",
                    "9pdmmk1r",
                    "o1vnl1yp",
                ],
            ),
            RunSpec(
                "matryoshka_topk",
                # Source: contrib/trait_discovery/sweeps/010_eccv/matryoshka_topk_nmse.py
                ["flqkcam7", "s3pqewz1", "l8hooa3r"],
            ),
        ]

        df, _skipped = load_df(specs)
        cols_by_method = {
            "vanilla_relu": {
                "sae_train_val_shard": "3e27794f",
                "probe_fit_val_shard": "3802cb66",
                "train_probe_r": "3802cb66/train_probe_r__train-614861a0",
                "val_probe_r": "3802cb66/val_probe_r__train-614861a0",
                "val_probe_map": "3802cb66/mean_ap__train-614861a0",
                "val_probe_purity_16": "3802cb66/purity_16__train-614861a0",
                "val_probe_cov_at_0_3": "3802cb66/cov_at_0_3__train-614861a0",
            },
            "matryoshka_relu": {
                "sae_train_val_shard": "3e27794f",
                "probe_fit_val_shard": "3802cb66",
                "train_probe_r": "3802cb66/train_probe_r__train-614861a0",
                "val_probe_r": "3802cb66/val_probe_r__train-614861a0",
                "val_probe_map": "3802cb66/mean_ap__train-614861a0",
                "val_probe_purity_16": "3802cb66/purity_16__train-614861a0",
                "val_probe_cov_at_0_3": "3802cb66/cov_at_0_3__train-614861a0",
            },
            "matryoshka_topk": {
                "sae_train_val_shard": "3e27794f",
                "probe_fit_val_shard": "3802cb66",
                "train_probe_r": "3802cb66/train_probe_r__train-614861a0",
                "val_probe_r": "3802cb66/val_probe_r__train-614861a0",
                "val_probe_map": "3802cb66/mean_ap__train-614861a0",
                "val_probe_purity_16": "3802cb66/purity_16__train-614861a0",
                "val_probe_cov_at_0_3": "3802cb66/cov_at_0_3__train-614861a0",
            },
        }

        def get_float(row: dict[str, object], key: str) -> float | None:
            if key not in row:
                return None
            value = row[key]
            if not isinstance(value, int | float) or isinstance(value, bool):
                return None
            return float(value)

        def get_shard_l0(row: dict[str, object], shard: str) -> float | None:
            shard_l0 = get_float(row, f"{shard}/l0")
            if shard_l0 is not None:
                return shard_l0
            return get_float(row, "l0")

        def make_null_row(method: str) -> dict[str, object]:
            return {
                "method": method,
                "run_id": None,
                "layer": None,
                "sae_train_val_nmse": None,
                "sae_train_val_l0": None,
                "probe_fit_val_nmse": None,
                "probe_fit_val_l0": None,
                "probe_fit_val_probe_r": None,
                "probe_fit_val_probe_map": None,
                "probe_fit_val_probe_purity_16": None,
                "probe_fit_val_probe_cov_at_0_3": None,
            }

        rows = []
        for method, cols in cols_by_method.items():
            train_probe_r_col = cols["train_probe_r"]
            if train_probe_r_col not in df.columns:
                rows.append(make_null_row(method))
                continue

            sub_df = df.filter(
                (pl.col("method") == method) & pl.col(train_probe_r_col).is_not_null()
            ).sort(train_probe_r_col, descending=True)
            if sub_df.is_empty():
                rows.append(make_null_row(method))
                continue

            best = sub_df.row(0, named=True)
            train_val_shard = cols["sae_train_val_shard"]
            probe_val_shard = cols["probe_fit_val_shard"]
            rows.append({
                "method": method,
                "run_id": best["id"] if "id" in best else None,
                "layer": int(best["layer"]) if "layer" in best else None,
                "sae_train_val_nmse": get_float(
                    best, f"{train_val_shard}/normalized_mse"
                ),
                "sae_train_val_l0": get_shard_l0(best, train_val_shard),
                "probe_fit_val_nmse": get_float(
                    best, f"{probe_val_shard}/normalized_mse"
                ),
                "probe_fit_val_l0": get_shard_l0(best, probe_val_shard),
                "probe_fit_val_probe_r": get_float(best, cols["val_probe_r"]),
                "probe_fit_val_probe_map": get_float(best, cols["val_probe_map"]),
                "probe_fit_val_probe_purity_16": get_float(
                    best, cols["val_probe_purity_16"]
                ),
                "probe_fit_val_probe_cov_at_0_3": get_float(
                    best, cols["val_probe_cov_at_0_3"]
                ),
            })

        table_df = pl.DataFrame(rows).sort("method")

        table_df.write_csv("contrib/trait_discovery/docs/assets/sae_variants_table.csv")
        return table_df

    sae_variants_table()
    return


@app.cell
def _(RunSpec, load_df, pl):
    def ade20k_vs_fishvista_table():
        specs = [
            RunSpec(
                "ADE20K (Matryoshka+TopK, DINOv3 ViT-L/16)",
                # Source: contrib/trait_discovery/sweeps/010_eccv/matryoshka_topk_nmse.py
                ["flqkcam7", "s3pqewz1", "l8hooa3r"],
            ),
            RunSpec(
                "FishVista (Matryoshka+TopK, DINOv3 ViT-L/16)",
                # Source: contrib/trait_discovery/sweeps/006_proposal_audit/cls_train.py
                # fishvista_run_ids[23].
                ["pdikj9bl", "hfpct5ae", "s465wgg4", "dc86xg8z", "bpz34d80"],
            ),
        ]
        cols_by_method = {
            "ADE20K (Matryoshka+TopK, DINOv3 ViT-L/16)": {
                "sae_train_val_shard": "3e27794f",
                "probe_fit_val_shard": "3802cb66",
                "train_probe_r": "3802cb66/train_probe_r__train-614861a0",
                "val_probe_r": "3802cb66/val_probe_r__train-614861a0",
                "val_probe_map": "3802cb66/mean_ap__train-614861a0",
                "val_probe_purity_16": "3802cb66/purity_16__train-614861a0",
                "val_probe_cov_at_0_3": "3802cb66/cov_at_0_3__train-614861a0",
            },
            "FishVista (Matryoshka+TopK, DINOv3 ViT-L/16)": {
                "sae_train_val_shard": "5dcb2f75",
                "probe_fit_val_shard": "8692dfa9",
                "train_probe_r": "8692dfa9/train_probe_r__train-5dcb2f75",
                "val_probe_r": "8692dfa9/val_probe_r__train-5dcb2f75",
                "val_probe_map": "8692dfa9/mean_ap__train-5dcb2f75",
                "val_probe_purity_16": "8692dfa9/purity_16__train-5dcb2f75",
                "val_probe_cov_at_0_3": "8692dfa9/cov_at_0_3__train-5dcb2f75",
            },
        }

        df, _skipped = load_df(specs)

        def get_float(row: dict[str, object], key: str) -> float | None:
            if key not in row:
                return None
            value = row[key]
            if not isinstance(value, int | float) or isinstance(value, bool):
                return None
            return float(value)

        def get_shard_l0(row: dict[str, object], shard: str) -> float | None:
            shard_l0 = get_float(row, f"{shard}/l0")
            if shard_l0 is not None:
                return shard_l0
            return get_float(row, "l0")

        def make_null_row(method: str) -> dict[str, object]:
            return {
                "method": method,
                "run_id": None,
                "layer": None,
                "sae_train_val_nmse": None,
                "sae_train_val_l0": None,
                "probe_fit_val_nmse": None,
                "probe_fit_val_l0": None,
                "probe_fit_val_probe_r": None,
                "probe_fit_val_probe_map": None,
                "probe_fit_val_probe_purity_16": None,
                "probe_fit_val_probe_cov_at_0_3": None,
            }

        rows = []
        for method, cols in cols_by_method.items():
            train_probe_r_col = cols["train_probe_r"]
            if train_probe_r_col not in df.columns:
                rows.append(make_null_row(method))
                continue

            sub_df = df.filter(
                (pl.col("method") == method) & pl.col(train_probe_r_col).is_not_null()
            ).sort(train_probe_r_col, descending=True)
            if sub_df.is_empty():
                rows.append(make_null_row(method))
                continue

            best = sub_df.row(0, named=True)
            train_val_shard = cols["sae_train_val_shard"]
            probe_val_shard = cols["probe_fit_val_shard"]
            rows.append({
                "method": method,
                "run_id": best["id"] if "id" in best else None,
                "layer": int(best["layer"]) if "layer" in best else None,
                "sae_train_val_nmse": get_float(
                    best, f"{train_val_shard}/normalized_mse"
                ),
                "sae_train_val_l0": get_shard_l0(best, train_val_shard),
                "probe_fit_val_nmse": get_float(
                    best, f"{probe_val_shard}/normalized_mse"
                ),
                "probe_fit_val_l0": get_shard_l0(best, probe_val_shard),
                "probe_fit_val_probe_r": get_float(best, cols["val_probe_r"]),
                "probe_fit_val_probe_map": get_float(best, cols["val_probe_map"]),
                "probe_fit_val_probe_purity_16": get_float(
                    best, cols["val_probe_purity_16"]
                ),
                "probe_fit_val_probe_cov_at_0_3": get_float(
                    best, cols["val_probe_cov_at_0_3"]
                ),
            })

        table_df = pl.DataFrame(rows).sort("method")
        table_df.write_csv(
            "contrib/trait_discovery/docs/assets/ade20k_vs_fishvista_table.csv"
        )
        return table_df

    ade20k_vs_fishvista_table()
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
        probe_col = "8692dfa9/mean_ap__train-5dcb2f75"
        val_shard = "8692dfa9"
        df, skipped_run_ids = load_df(specs)
        if probe_col not in df.columns:
            mo.output.append(
                mo.md(
                    f"Missing probe column `{probe_col}`. Ensure probe metrics exist on disk."
                )
            )
            return None
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
