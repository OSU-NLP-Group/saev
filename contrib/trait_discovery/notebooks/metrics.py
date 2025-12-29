import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import json
    import pathlib
    import pickle

    import beartype
    import marimo as mo
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import numpy as np
    import pandera.polars as pa
    import polars as pl
    import wandb
    from adjustText import adjust_text

    import saev.colors
    import saev.data
    import saev.disk

    return (
        adjust_text,
        base64,
        beartype,
        json,
        mo,
        np,
        pa,
        pathlib,
        pe,
        pickle,
        pl,
        plt,
        saev,
        wandb,
    )


@app.cell
def _(pathlib):
    runs_root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
    shards_root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")

    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "saev"
    return WANDB_USERNAME, runs_root, shards_root


@app.cell
def _(beartype, pathlib):
    @beartype.beartype
    def get_inference_probe_metric_fpaths(
        run_dpath: pathlib.Path,
    ) -> list[pathlib.Path]:
        if not run_dpath.is_dir():
            return []

        inference_dpath = run_dpath / "inference"
        if not inference_dpath.is_dir():
            return []

        probe_metric_fpaths: list[pathlib.Path] = []
        for shard_dpath in inference_dpath.iterdir():
            if not shard_dpath.is_dir():
                continue

            probe_metrics_fpath = shard_dpath / "probe1d_metrics.npz"
            if not probe_metrics_fpath.is_file():
                continue

            probe_metric_fpaths.append(probe_metrics_fpath)

        return probe_metric_fpaths

    return (get_inference_probe_metric_fpaths,)


@app.cell
def _(
    beartype,
    get_baseline_ce,
    get_inference_probe_metric_fpaths,
    get_wandb_run,
    json,
    mo,
    mode,
    np,
    pa,
    pathlib,
    pl,
    runs_root,
    saev,
    shards_root,
):
    class ProbeResultsSchema(pa.DataFrameModel):
        run_id: str

        model: str = pa.Field()
        layer: int = pa.Field(ge=0)
        objective: str = pa.Field(isin=["vanilla", "matryoshka"])

        # From train.py
        sae_data: str = pa.Field()
        sae_val_mse: float = pa.Field(ge=0)
        sae_val_l0: float = pa.Field(ge=0)
        sae_val_l1: float = pa.Field(ge=0)

        # From inference.py; normalized MSE on ADE20K
        train_nmse: float = pa.Field()
        val_nmse: float = pa.Field()

        # From probe1d.py
        frac_w_neg: float = pa.Field(ge=0)
        frac_best_w_neg: float = pa.Field(ge=0)
        # train
        train_probe_shards: str = pa.Field()
        train_probe_ce: float = pa.Field(ge=0)
        train_baseline_ce: float = pa.Field(ge=0)
        train_probe_r: float = pa.Field()
        # val
        val_probe_shards: str = pa.Field()
        val_probe_ce: float = pa.Field(ge=0)
        val_baseline_ce: float = pa.Field(ge=0)
        val_probe_r: float = pa.Field()

        # metrics
        val_mean_ap: float = pa.Field()
        val_mean_prec: float = pa.Field()
        val_mean_recall: float = pa.Field()
        val_mean_f1: float = pa.Field()
        val_mean_purity_16: float = pa.Field()
        cov_at_0_3: float = pa.Field(ge=0)
        cov_at_0_5: float = pa.Field(ge=0)
        cov_at_0_7: float = pa.Field(ge=0)

    @beartype.beartype
    def get_probe_split_label(shards_dpath: pathlib.Path) -> str | None:
        try:
            metadata = saev.data.Metadata.load(shards_dpath)
        except Exception as err:
            print(f"Failed to load metadata for {shards_dpath}: {err}")
            return None

        data_cfg = metadata.make_data_cfg()
        split = getattr(data_cfg, "split", None)
        if split is None:
            return None

        split_name = str(split).lower()
        if split_name in {"train", "training"}:
            return "train"
        if split_name in {"val", "validation"}:
            return "val"
        return None

    @beartype.beartype
    @pa.check_types
    def load_probe_results_df() -> pa.typing.DataFrame[ProbeResultsSchema]:
        rows = []

        run_probe_fpaths: list[tuple[saev.disk.Run, list[pathlib.Path]]] = []
        n_probe_metrics = 0
        for run_dpath in runs_root.iterdir():
            if not run_dpath.is_dir():
                print(f"Skipping {run_dpath}: not a directory.")
                continue

            probe_fpaths = get_inference_probe_metric_fpaths(run_dpath)
            if not probe_fpaths:
                continue

            run = saev.disk.Run(run_dpath)
            run_probe_fpaths.append((run, probe_fpaths))
            n_probe_metrics += len(probe_fpaths)

        print(f"Found {n_probe_metrics} probe1d_metrics.npz files.")
        for run, probe_fpaths in mo.status.progress_bar(run_probe_fpaths):
            wandb_run = get_wandb_run(run.run_id, project="saev")
            if not wandb_run:
                print(f"Skipping {run.run_id}: no wandb run.")
                continue

            if "config/objective/n_prefixes" in wandb_run and isinstance(
                wandb_run["config/objective/n_prefixes"], int
            ):
                objective = "matryoshka"
            else:
                objective = "vanilla"

            split_map: dict[str, tuple[pathlib.Path, str, pathlib.Path]] = {}
            for metrics_fpath in probe_fpaths:
                shard_id = metrics_fpath.parent.name
                shards_dpath = shards_root / shard_id

                if not shards_dpath.exists():
                    print(
                        f"Skipping run {run.run_id}: shards directory {shards_dpath} missing."
                    )
                    continue

                split_label = get_probe_split_label(shards_dpath)
                if split_label is None:
                    print(
                        f"Skipping shards {shard_id}: unknown split (run {run.run_id})."
                    )
                    continue

                if split_label in split_map:
                    print(
                        f"Skipping run {run.run_id}: duplicate {split_label} probe metrics discovered."
                    )
                    split_map = {}
                    break

                split_map[split_label] = (metrics_fpath, shard_id, shards_dpath)

            if not split_map:
                print(f"Skipping run {run.run_id}: no splits.")
                continue

            if "train" not in split_map or "val" not in split_map:
                print(
                    f"Skipping run {run.run_id}: expected train and val probe metrics, got {sorted(split_map.keys())}."
                )
                continue

            train_probe_metrics_fpath, train_shards, train_shards_dpath = split_map[
                "train"
            ]
            val_probe_metrics_fpath, val_shards, val_shards_dpath = split_map["val"]

            with np.load(train_probe_metrics_fpath) as fd:
                train_loss = fd["loss"]
                w = fd["weights"]

            with np.load(val_probe_metrics_fpath) as fd:
                val_loss = fd["loss"]

            assert train_loss.ndim == 2
            assert val_loss.ndim == 2
            assert train_loss.shape == val_loss.shape

            n_latents, n_classes = train_loss.shape

            best_i = np.argmin(train_loss, axis=0)
            train_ce = train_loss[best_i, np.arange(n_classes)].mean().item()
            val_ce = val_loss[best_i, np.arange(n_classes)].mean().item()

            train_base_ce = get_baseline_ce(train_shards_dpath).mean().item()
            val_base_ce = get_baseline_ce(val_shards_dpath).mean().item()

            # Normalized MSE
            train_nmse = 1.0
            path = run.inference / train_shards / "metrics.json"
            if path.is_file():
                train_nmse = json.loads(path.read_text())["normalized_mse"]

            val_nmse = 1.0
            path = run.inference / val_shards / "metrics.json"
            if path.is_file():
                val_nmse = json.loads(path.read_text())["normalized_mse"]

            # Get mAP from /fs/ess/PAS2136/samuelstevens/saev/runs/knz7yndg/inference/66a5d2c1/avgprec__train-fd5781e8.npz
            taus = [0.3, 0.5, 0.7]
            mean_ap, mean_prec, mean_recall, mean_f1 = 0.0, 0.0, 0.0, 0.0
            cov = {f"cov_at_{tau}".replace(".", "_"): 0.0 for tau in taus}
            purity = 0.0

            k = 16
            path = (
                run.inference
                / val_shards
                / f"probe1d_metrics__train-{train_shards}.npz"
            )
            if path.is_file():
                with np.load(path) as fd:
                    ap_c = fd["ap"]
                    prec_c = fd["precision"]
                    recall_c = fd["recall"]
                    f1_c = fd["f1"]

                    if "top_labels" not in fd:
                        print(f"{run.run_id} missing 'top_labels'.")
                        continue

                    top_labels_dk = fd["top_labels"]

                mean_ap = ap_c.mean().item()
                mean_prec = prec_c.mean().item()
                mean_recall = recall_c.mean().item()
                mean_f1 = f1_c.mean().item()
                cov = {
                    f"cov_at_{tau}".replace(".", "_"): (ap_c > tau).mean().item()
                    for tau in taus
                }

                _, count = mode(top_labels_dk[best_i, :16], axis=1)
                purity = (count / k).mean().item()

            rows.append({
                "run_id": run.run_id,
                "model": wandb_run["model_key"],
                "layer": wandb_run["config/val_data/layer"],
                "objective": objective,
                "sae_data": wandb_run["data_key"],
                "sae_val_mse": wandb_run["summary/eval/mse"],
                "sae_val_l0": wandb_run["summary/eval/l0"],
                "sae_val_l1": wandb_run["summary/eval/l1"],
                "train_nmse": train_nmse,
                "val_nmse": val_nmse,
                "frac_w_neg": (w < 0).mean().item(),
                "frac_best_w_neg": (w[best_i, np.arange(n_classes)] < 0).mean().item(),
                "train_probe_shards": train_shards,
                "train_probe_ce": train_ce,
                "train_baseline_ce": train_base_ce,
                "train_probe_r": 1 - train_ce / train_base_ce,
                "val_probe_shards": val_shards,
                "val_probe_ce": val_ce,
                "val_baseline_ce": val_base_ce,
                "val_probe_r": 1 - val_ce / val_base_ce,
                "val_mean_ap": mean_ap,
                "val_mean_prec": mean_prec,
                "val_mean_recall": mean_recall,
                "val_mean_f1": mean_f1,
                f"val_mean_purity_{k}": purity,
                **cov,
            })

        return pl.DataFrame(rows)

    df = load_probe_results_df()
    df
    return (df,)


@app.cell
def _(df, pl):
    df.filter(pl.col("val_mean_ap") == 0)
    return


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("model") == "BioCLIP 2 ViT-L/14")
        & (pl.col("sae_data") == "FishVista (Img)")
    )
    return


@app.cell
def _(df, pl, plt):
    def _():
        filtered = df.filter(
            (pl.col("model") == "DINOv3 ViT-S/16")
            & (pl.col("train_probe_shards") == "781f8739")
            & (pl.col("val_probe_shards") == "5e195bbf")
        )

        fig, (ax_ce, ax_r) = plt.subplots(
            ncols=2, dpi=200, figsize=(8, 3), layout="constrained"
        )

        # Plot cross entropy
        train_probe_ce = filtered.get_column("train_probe_ce").to_numpy()
        val_probe_ce = filtered.get_column("val_probe_ce").to_numpy()

        train_baseline_ce = filtered.get_column("train_baseline_ce").to_numpy()
        val_baseline_ce = filtered.get_column("val_baseline_ce").to_numpy()

        assert all(ce == train_baseline_ce[0] for ce in train_baseline_ce)
        assert all(ce == val_baseline_ce[0] for ce in val_baseline_ce)

        train_baseline_ce = train_baseline_ce[0:1]
        val_baseline_ce = val_baseline_ce[0:1]

        min_ce = min(
            train_probe_ce.min(),
            val_probe_ce.min(),
            train_baseline_ce.item(),
            val_baseline_ce.item(),
        )
        max_ce = max(
            train_probe_ce.max(),
            val_probe_ce.max(),
            train_baseline_ce.item(),
            val_baseline_ce.item(),
        )

        ax_ce.plot([min_ce, max_ce], [min_ce, max_ce], color="tab:red", alpha=0.1)
        ax_ce.fill_between(
            [min_ce, max_ce],
            [max_ce, max_ce],
            [min_ce, max_ce],
            alpha=0.3,
            color="tab:red",
            linewidth=0,
            label="Overfitting",
        )
        ax_ce.scatter(train_probe_ce, val_probe_ce, label="Probe CE", alpha=0.5)
        ax_ce.scatter(
            train_baseline_ce, val_baseline_ce, label="Baseline CE", alpha=0.5
        )

        ax_ce.grid(True, linewidth=0.3, alpha=0.5)
        ax_ce.spines[["right", "top"]].set_visible(False)
        # ax.set_xscale("log")
        ax_ce.set_xlabel("Train CE ($\\downarrow$)")
        ax_ce.set_ylabel("Val CE ($\\downarrow$)")

        ax_ce.legend()

        xs = filtered.get_column("train_probe_r")
        ys = filtered.get_column("val_probe_r")
        min_r = min(xs.min(), ys.min())
        max_r = max(xs.max(), ys.max())

        ax_r.plot([min_r, max_r], [min_r, max_r], color="tab:red", alpha=0.1)
        ax_r.fill_between(
            [min_r, max_r],
            [min_r, min_r],
            [min_r, max_r],
            alpha=0.3,
            color="tab:red",
            linewidth=0,
            label="Overfitting",
        )
        ax_r.scatter(xs, ys, label="Probe R", alpha=0.5)

        ax_r.grid(True, linewidth=0.3, alpha=0.5)
        ax_r.spines[["right", "top"]].set_visible(False)
        # ax.set_xscale("log")
        ax_r.set_xlabel("Train R ($\\uparrow$)")
        ax_r.set_ylabel("Val R ($\\uparrow$)")

        ax_r.legend()

        fig.suptitle("Measuring Overfitting")

        return fig

    _()
    return


@app.cell
def _(beartype, df, np, pl, plt):
    @beartype.beartype
    def plot_layerwise_explained_variance(
        model: str, shards: str, layers: list[int], n_layers: int
    ):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=6,
            dpi=300,
            figsize=(12, 2.4),
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.reshape(-1)

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            filtered = df.filter(
                (pl.col("model") == model)
                & (pl.col("val_probe_shards") == shards)
                & (pl.col("layer") == layer)
                & (pl.col("val_nmse") <= 1.0)
                & (pl.col("objective") != "vanilla")
            ).sort(by="layer")

            xs = filtered.get_column("val_nmse").to_numpy()
            xs = 1 - xs
            ys = filtered.get_column("val_probe_r").to_numpy()

            ax.set_title(f"Layer {layer + 1}/{n_layers}")

            try:
                line = np.polynomial.Polynomial.fit(xs, ys, deg=2)
                x_lin, y_lin = line.linspace(domain=[0, 1])
                print(model, layer, y_lin.max())
                ax.plot(
                    x_lin,
                    y_lin,
                    color="tab:orange",
                    linestyle=(0, (3, 1)),
                    label="$y=ax^2+bx+c$",
                    alpha=0.6,
                )
            except ValueError:
                pass

            ax.scatter(xs, ys, color="tab:blue", alpha=0.8, zorder=3, clip_on=False)

            ax.grid(True, linewidth=0.3, alpha=0.5)
            ax.spines[["right", "top"]].set_visible(False)
            # ax.set_xscale("log")

            # ax.set_xlabel("$\\log_{10}$ L0")
            # ax.set_xlabel("Normalized MSE")
            ax.set_xlabel("Explained Variance")
            if i in (0,):
                ax.set_ylabel("Val Probe R ($\\uparrow$)")
            if i in (5,):
                ax.legend()

            ax.set_ylim(0, 0.5)
            ax.set_xlim(0, 1.0)

        return fig

    return (plot_layerwise_explained_variance,)


@app.cell
def _(mo, plot_layerwise_explained_variance):
    def _():
        vitl_fig = plot_layerwise_explained_variance(
            "DINOv3 ViT-L/16", "3802cb66", [13, 15, 17, 19, 21, 23], 24
        )
        vitl_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vitl16_in1k_ade20k_probe1d.pdf"
        )

        vitb_fig = plot_layerwise_explained_variance(
            "DINOv3 ViT-B/16", "66a5d2c1", [6, 7, 8, 9, 10, 11], 12
        )
        vitb_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vitb16_in1k_ade20k_probe1d.pdf"
        )

        vits_fig = plot_layerwise_explained_variance(
            "DINOv3 ViT-S/16", "5e195bbf", [6, 7, 8, 9, 10, 11], 12
        )
        vits_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vits16_in1k_ade20k_probe1d.pdf"
        )

        return mo.vstack([
            mo.md("# Explained Variance vs. Probe R\n ---"),
            vits_fig,
            vitb_fig,
            vitl_fig,
            mo.md("---"),
        ])

    _()
    return


@app.cell
def _(beartype, df, np, pl, plt):
    @beartype.beartype
    def plot_layerwise_log_l0(
        model: str, shards: str, layers: list[int], n_layers: int
    ):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=6,
            dpi=300,
            figsize=(12, 2.4),
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.reshape(-1)

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            filtered = df.filter(
                (pl.col("model") == model)
                & (pl.col("val_probe_shards") == shards)
                & (pl.col("layer") == layer)
                & (pl.col("val_nmse") <= 1.0)
                & (pl.col("objective") == "matryoshka")
            ).sort(by="layer")

            xs = filtered.get_column("sae_val_l0").to_numpy()
            ys = filtered.get_column("val_probe_r").to_numpy()

            ax.set_title(f"Layer {layer + 1}/{n_layers}")

            try:
                line = np.polynomial.Polynomial.fit(np.log10(xs), ys, deg=1)
                x_linspace, y_linspace = line.linspace()
                ax.plot(
                    10**x_linspace,
                    y_linspace,
                    color="tab:orange",
                    linestyle=(0, (3, 1)),
                    label="$y=ax^2+bx+c$",
                    alpha=0.6,
                )
            except ValueError:
                pass

            ax.scatter(xs, ys, color="tab:blue", alpha=0.8, zorder=3, clip_on=False)

            ax.grid(True, linewidth=0.3, alpha=0.5)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")

            ax.set_xlabel("L0")
            if i in (0,):
                ax.set_ylabel("Val Probe R ($\\uparrow$)")
            if i in (5,):
                ax.legend()

            ax.set_ylim(0, 0.6)

        return fig

    return (plot_layerwise_log_l0,)


@app.cell
def _(mo, plot_layerwise_log_l0):
    def _():
        vitl_fig = plot_layerwise_log_l0(
            "DINOv3 ViT-L/16", "3802cb66", [13, 15, 17, 19, 21, 23], 24
        )
        vitl_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vitl16_in1k_ade20k_probe1d_logl0.pdf"
        )

        vitb_fig = plot_layerwise_log_l0(
            "DINOv3 ViT-B/16", "66a5d2c1", [6, 7, 8, 9, 10, 11], 12
        )
        vitb_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vitb16_in1k_ade20k_probe1d_logl0.pdf"
        )

        vits_fig = plot_layerwise_log_l0(
            "DINOv3 ViT-S/16", "5e195bbf", [6, 7, 8, 9, 10, 11], 12
        )
        vits_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vits16_in1k_ade20k_probe1d_logl0.pdf"
        )

        return mo.vstack([
            mo.md("# $\\log_{10}$ L0 vs. Probe R\n ---"),
            vits_fig,
            vitb_fig,
            vitl_fig,
            mo.md("---"),
        ])

    _()
    return


@app.cell
def _(BLUE_RGB01, ORANGE_RGB01, beartype, df, mo, np, pl, plt):
    @beartype.beartype
    def plot_layerwise_mse(
        model: str, shards: str, layers: list[int], n_layers: int, title: str = ""
    ):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=6,
            dpi=300,
            figsize=(12, 2.4),
            layout="constrained",
            sharey=True,
        )
        axes = axes.reshape(-1)

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            filtered = df.filter(
                (pl.col("model") == model)
                & (pl.col("val_probe_shards") == shards)
                & (pl.col("layer") == layer)
                & (pl.col("objective") == "matryoshka")
                & (pl.col("val_nmse") <= 1.0)
            ).sort(by="layer")

            xs = filtered.get_column("val_nmse").to_numpy()
            ys = filtered.get_column("val_mean_ap").to_numpy()

            ax.set_title(f"Layer {layer + 1}/{n_layers}")

            ax.scatter(
                xs, ys, color=BLUE_RGB01, alpha=0.7, zorder=3, label="Pareto SAEs"
            )

            try:
                line = np.polynomial.Polynomial.fit(xs, ys, deg=2)
                x_line, y_line = line.linspace(domain=[0, 1])
                ax.plot(
                    x_line,
                    y_line,
                    color=ORANGE_RGB01,
                    linestyle=(0, (3, 1)),
                    label="$y=ax^2+bx+c$",
                    alpha=0.3,
                )
            except ValueError:
                pass

            ax.grid(True, linewidth=0.3, alpha=0.5)
            ax.spines[["right", "top"]].set_visible(False)

            ax.set_xlabel("Val. NMSE")
            if i in (0,):
                ax.set_ylabel("Val. mAP ($\\uparrow$)")
            if i in (0,):
                ax.legend(loc="upper right")

            ax.set_ylim(0, 0.4)
            ax.set_xlim(0, 1.0)
            fig.suptitle(title or model.removeprefix("DINOv3 "))

        return fig

    def _():
        vitl_fig = plot_layerwise_mse(
            "DINOv3 ViT-L/16",
            "3802cb66",
            [13, 15, 17, 19, 21, 23],
            24,
            title="ViT-L/16 (303M)",
        )
        vitl_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vitl16_in1k_ade20k_probe1d_nmse_map.pdf"
        )

        vitb_fig = plot_layerwise_mse(
            "DINOv3 ViT-B/16",
            "66a5d2c1",
            [6, 7, 8, 9, 10, 11],
            12,
            title="ViT-B/16 (86M)",
        )
        vitb_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vitb16_in1k_ade20k_probe1d_nmse_map.pdf"
        )

        vits_fig = plot_layerwise_mse(
            "DINOv3 ViT-S/16",
            "5e195bbf",
            [6, 7, 8, 9, 10, 11],
            12,
            title="ViT-S/16 (22M)",
        )
        vits_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vits16_in1k_ade20k_probe1d_nmse_map.pdf"
        )

        return mo.vstack([
            mo.md("# Training NMSE vs. Probe R"),
            vits_fig,
            vitb_fig,
            vitl_fig,
            mo.md("---"),
        ])

    _()
    return


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("model") == "DINOv3 ViT-L/16")
        & (pl.col("val_probe_shards") == "3802cb66")
        & (pl.col("objective") == "matryoshka")
    ).sort(by="val_probe_r", descending=True).head(2).select(
        "run_id",
        "model",
        "layer",
        "train_probe_shards",
        "objective",
        "sae_val_mse",
        "sae_val_l0",
        "val_probe_r",
        "val_nmse",
        "val_mean_ap",
        "cov_at_0_3",
        "cov_at_0_5",
        "cov_at_0_7",
    )
    return


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("model") == "DINOv3 ViT-L/16")
        & (pl.col("val_probe_shards") == "3802cb66")
        & (pl.col("objective") == "vanilla")
    ).sort(by="val_probe_r", descending=True).head(2).select(
        "run_id",
        "model",
        "layer",
        "train_probe_shards",
        "objective",
        "sae_val_mse",
        "sae_val_l0",
        "val_probe_r",
        "val_nmse",
        "val_mean_ap",
        "cov_at_0_3",
        "cov_at_0_5",
        "cov_at_0_7",
    )
    return


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("model") == "DINOv3 ViT-B/16")
        & (pl.col("val_probe_shards") == "66a5d2c1")
        & (pl.col("objective") == "matryoshka")
    ).sort(by="val_probe_r", descending=True).head(2).select(
        "run_id",
        "model",
        "layer",
        "objective",
        "sae_val_mse",
        "sae_val_l0",
        "val_probe_r",
        "val_nmse",
        "val_mean_ap",
        "cov_at_0_3",
        "cov_at_0_5",
        "cov_at_0_7",
    )
    return


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("model") == "DINOv3 ViT-S/16")
        & (pl.col("val_probe_shards") == "5e195bbf")
        & (pl.col("objective") == "matryoshka")
    ).sort(by="val_probe_r", descending=True).head(2).select(
        "run_id",
        "model",
        "layer",
        "objective",
        "sae_val_mse",
        "sae_val_l0",
        "val_probe_r",
        "val_nmse",
        "val_mean_ap",
        "cov_at_0_3",
        "cov_at_0_5",
        "cov_at_0_7",
    )
    return


@app.cell
def _(BLUE_RGB01, ORANGE_RGB01, SEA_RGB01, beartype, df, pl, plt):
    @beartype.beartype
    def _():
        fig, ax1 = plt.subplots(
            nrows=1, ncols=1, dpi=300, figsize=(4.5, 3.5), layout="constrained"
        )

        handles = []

        grouped = (
            df.filter(
                (pl.col("model") == "DINOv3 ViT-B/16")
                & (pl.col("val_probe_shards") == "66a5d2c1")
            )
            .group_by(pl.col("layer"))
            .agg(pl.col("val_mean_ap").max())
            .sort(by="layer")
        )

        ys = grouped.get_column("val_mean_ap").to_numpy()
        xs = grouped.get_column("layer").to_numpy() + 1

        (handle,) = ax1.plot(
            xs,
            ys,
            color=ORANGE_RGB01,
            alpha=0.8,
            marker="s",
            label="ViT-S/16",
            linestyle="--",
        )
        handles.append(handle)

        grouped = (
            df.filter(
                (pl.col("model") == "DINOv3 ViT-S/16")
                & (pl.col("val_probe_shards") == "5e195bbf")
            )
            .group_by(pl.col("layer"))
            .agg(pl.col("val_mean_ap").max())
            .sort(by="layer")
        )

        ys = grouped.get_column("val_mean_ap").to_numpy()
        xs = grouped.get_column("layer").to_numpy() + 1

        (handle,) = ax1.plot(
            xs,
            ys,
            color=BLUE_RGB01,
            alpha=0.8,
            marker="^",
            label="ViT-S/16",
            linestyle="-.",
        )
        handles.append(handle)

        ax1.grid(True, linewidth=0.3, alpha=0.5)
        ax1.set_xlabel("ViT-{B,S}/16 Layer")
        ax1.set_ylabel("mAP ($\\uparrow$)")
        ax1.set_ylim(0, 0.5)

        ax2 = ax1.twiny()

        grouped = (
            df.filter(
                (pl.col("model") == "DINOv3 ViT-L/16")
                & (pl.col("val_probe_shards") == "3802cb66")
            )
            .group_by(pl.col("layer"))
            .agg(pl.col("val_mean_ap").max())
            .sort(by="layer")
        )

        xs = grouped.get_column("layer").to_numpy() + 1
        ys = grouped.get_column("val_mean_ap").to_numpy()

        (handle,) = ax2.plot(
            xs, ys, color=SEA_RGB01, alpha=0.8, label="ViT-L/16", marker="o"
        )
        handles.insert(0, handle)
        ax2.set_xlabel("ViT-L/16 Layer")

        ax1.legend(handles=handles, loc="upper left")

        fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_in1k_ade20k_layers_map.pdf"
        )
        return fig

    _()
    return


@app.cell
def _(BLUE_RGB01, ORANGE_RGB01, SEA_RGB01, beartype, df, pl, plt):
    @beartype.beartype
    def _():
        fig, ax1 = plt.subplots(
            nrows=1, ncols=1, dpi=300, figsize=(4.5, 3.5), layout="constrained"
        )
        y_col = "cov_at_0_5"

        handles = []

        grouped = (
            df.filter(
                (pl.col("model") == "DINOv3 ViT-B/16")
                & (pl.col("val_probe_shards") == "66a5d2c1")
            )
            .group_by(pl.col("layer"))
            .agg(pl.col(y_col).max())
            .sort(by="layer")
        )

        ys = grouped.get_column(y_col).to_numpy()
        xs = grouped.get_column("layer").to_numpy() + 1

        (handle,) = ax1.plot(
            xs, ys, color=ORANGE_RGB01, alpha=0.8, marker="s", label="ViT-B/16"
        )
        handles.append(handle)

        grouped = (
            df.filter(
                (pl.col("model") == "DINOv3 ViT-S/16")
                & (pl.col("val_probe_shards") == "5e195bbf")
            )
            .group_by(pl.col("layer"))
            .agg(pl.col(y_col).max())
            .sort(by="layer")
        )

        ys = grouped.get_column(y_col).to_numpy()
        xs = grouped.get_column("layer").to_numpy() + 1

        (handle,) = ax1.plot(
            xs, ys, color=BLUE_RGB01, alpha=0.8, marker="^", label="ViT-S/16"
        )
        handles.append(handle)

        ax1.grid(True, linewidth=0.3, alpha=0.5)
        ax1.set_xlabel("ViT-{B,S}/16 Layer")
        ax1.set_ylabel("Cov@0.5 ($\\uparrow$)")

        ax2 = ax1.twiny()

        grouped = (
            df.filter(
                (pl.col("model") == "DINOv3 ViT-L/16")
                & (pl.col("val_probe_shards") == "3802cb66")
            )
            .group_by(pl.col("layer"))
            .agg(pl.col(y_col).max())
            .sort(by="layer")
        )

        xs = grouped.get_column("layer").to_numpy() + 1
        ys = grouped.get_column(y_col).to_numpy()

        (handle,) = ax2.plot(
            xs, ys, color=SEA_RGB01, alpha=0.8, label="ViT-L/16", marker="o"
        )
        handles.insert(0, handle)
        ax2.set_xlabel("ViT-L/16 Layer")

        ax1.legend(handles=handles, loc="upper left")

        fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_in1k_ade20k_layers_coverage.pdf"
        )
        return fig

    _()
    return


@app.cell
def _(beartype, df, np, pl, plt):
    @beartype.beartype
    def plot_layerwise_map(model: str, shards: str, layers: list[int], n_layers: int):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=6,
            dpi=300,
            figsize=(12, 2.4),
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.reshape(-1)

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            filtered = df.filter(
                (pl.col("model") == model)
                & (pl.col("val_probe_shards") == shards)
                & (pl.col("layer") == layer)
                & (pl.col("objective") == "matryoshka")
            ).sort(by="layer")

            xs = filtered.get_column("val_nmse").to_numpy()
            xs = 1 - xs
            ys = filtered.get_column("val_mean_ap").to_numpy()

            ax.set_title(f"Layer {layer + 1}/{n_layers}")

            try:
                line = np.polynomial.Polynomial.fit(xs, ys, deg=2)
                x_lin, y_lin = line.linspace(domain=[0, 1])
                print(model, layer, y_lin.max())
                ax.plot(
                    x_lin,
                    y_lin,
                    color="tab:orange",
                    linestyle=(0, (3, 1)),
                    label="$y=ax^2+bx+c$",
                    alpha=0.6,
                )
            except ValueError:
                pass

            ax.scatter(xs, ys, color="tab:blue", alpha=0.8, zorder=3, clip_on=False)

            ax.grid(True, linewidth=0.3, alpha=0.5)
            ax.spines[["right", "top"]].set_visible(False)
            # ax.set_xscale("log")

            # ax.set_xlabel("$\\log_{10}$ L0")
            # ax.set_xlabel("Normalized MSE")
            ax.set_xlabel("Explained Variance")
            if i in (0,):
                ax.set_ylabel("mAP ($\\uparrow$)")
            if i in (5,):
                ax.legend()

            ax.set_ylim(0, 0.5)
            ax.set_xlim(0, 1.0)

        return fig

    return (plot_layerwise_map,)


@app.cell
def _(mo, plot_layerwise_map):
    def _():
        vitl_fig = plot_layerwise_map(
            "DINOv3 ViT-L/16", "3802cb66", [13, 15, 17, 19, 21, 23], 24
        )
        vitl_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vitl16_in1k_ade20k_mean_ap.pdf"
        )

        vitb_fig = plot_layerwise_map(
            "DINOv3 ViT-B/16", "66a5d2c1", [6, 7, 8, 9, 10, 11], 12
        )
        vitb_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vitb16_in1k_ade20k_mean_ap.pdf"
        )

        vits_fig = plot_layerwise_map(
            "DINOv3 ViT-S/16", "5e195bbf", [6, 7, 8, 9, 10, 11], 12
        )
        vits_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_vits16_in1k_ade20k_mean_ap.pdf"
        )

        return mo.vstack([vits_fig, vitb_fig, vitl_fig])

    _()
    return


@app.cell
def _(pathlib):
    def get_class_names():
        class_names = ["none"]
        path = pathlib.Path(
            "/fs/ess/PAS2136/samuelstevens/datasets/ADEChallengeData2016/objectInfo150.txt"
        )
        for line in path.read_text().split("\n")[1:]:
            if not line:
                continue
            i, _, _, _, name = line.split("\t")
            assert int(i) == len(class_names)
            class_names.append(name)

        return class_names

    get_class_names()
    return (get_class_names,)


@app.cell
def _(df, mo, runs_root):
    ckpt_dropdown = mo.ui.dropdown(
        sorted([
            run_id
            for (run_id,) in df.select("run_id").iter_rows()
            if (runs_root / run_id).is_dir()
        ]),
        label="Checkpoint",
        searchable=True,
    )
    return (ckpt_dropdown,)


@app.cell
def _(ckpt_dropdown):
    ckpt_dropdown
    return


@app.cell
def _(
    beartype,
    get_baseline_ce,
    get_class_names,
    mo,
    mode,
    np,
    pathlib,
    plt,
    runs_root,
    saev,
    shards_root,
):
    @mo.cache
    @beartype.beartype
    def get_class_prevalence(shards_dir: pathlib.Path):
        md = saev.data.Metadata.load(shards_dir)
        labels = np.memmap(
            shards_dir / "labels.bin",
            mode="r",
            dtype=np.uint8,
            shape=(md.n_examples, md.content_tokens_per_example),
        ).reshape(-1)

        unique, counts = np.unique(
            labels, return_counts=True, equal_nan=False, sorted=True
        )
        by_freq = np.argsort(counts)[::-1]

        return by_freq, counts[by_freq]

    def plot_prevalence_vs_metric():
        fig, axes = plt.subplots(
            figsize=(8, 9), nrows=3, ncols=2, dpi=300, layout="constrained", sharex=True
        )
        (ax1, ax2, ax3, ax4, ax5, ax6) = axes.reshape(-1)

        # ViT-S/16
        train_shards = "781f8739"
        val_shards = "5e195bbf"
        run = saev.disk.Run(runs_root / "gc6iqrf2")

        # ViT-B/16
        # train_shards = "fd5781e8"
        # val_shards = "66a5d2c1"
        # run = saev.disk.Run(runs_root / "qoc1660r")

        # ViT-L/16
        train_shards = "614861a0"
        val_shards = "3802cb66"
        run = saev.disk.Run(runs_root / "mccrm7u8")

        # ViT-L/16 (Vanilla)
        train_shards = "614861a0"
        val_shards = "3802cb66"
        run = saev.disk.Run(runs_root / "0pdum8cq")

        with np.load(run.inference / train_shards / "probe1d_metrics.npz") as fd:
            train_loss = fd["loss"]

        with np.load(run.inference / val_shards / "probe1d_metrics.npz") as fd:
            val_loss = fd["loss"]

        assert train_loss.ndim == 2
        assert val_loss.ndim == 2
        assert train_loss.shape == val_loss.shape

        n_latents, n_classes = train_loss.shape

        best_i = np.argmin(train_loss, axis=0)
        train_ce = train_loss[best_i, np.arange(n_classes)]
        val_ce = val_loss[best_i, np.arange(n_classes)]

        train_base_ce = get_baseline_ce(shards_root / train_shards)
        val_base_ce = get_baseline_ce(shards_root / val_shards)

        path = run.inference / val_shards / f"probe1d_metrics__train-{train_shards}.npz"
        assert path.exists()
        with np.load(path) as fd:
            ap_c = fd["ap"]
            prec_c = fd["precision"]
            recall_c = fd["recall"]
            f1_c = fd["f1"]
            top_labels_dk = fd["top_labels"]

        class_names = get_class_names()
        for i, (latent, name) in enumerate(zip(best_i, class_names[:20])):
            print(latent, name, ap_c[i].item())
        print("...")

        best_classes_i = np.argsort(ap_c)[::-1]
        for rank, class_i in enumerate(best_classes_i[:30]):
            latent = best_i[class_i]
            name = class_names[class_i]
            ap = ap_c[class_i].item()
            k = 16
            _, count = mode(top_labels_dk[latent, :k])
            print(
                f"#{rank + 1}: latent {latent} scores {ap:.3f} on class '{name}' with purity@{k} {count.item() / k:.3f}."
            )

        by_freq, counts = get_class_prevalence(shards_root / train_shards)
        xs = counts
        # xs = np.arange(151)

        ax1.scatter(xs, ap_c[by_freq], alpha=0.5)
        ax1.set_ylabel("Average Precision")
        ax1.set_ylim(0, 1)

        _, counts = mode(top_labels_dk[best_i, :k], axis=1)
        print(f"Purity@{k}: {(counts / k).mean().item()}")
        ax2.scatter(xs, (counts / k)[by_freq], alpha=0.5, clip_on=False)
        ax2.set_ylabel("Purity@32")
        ax2.set_ylim(0, 1)

        ax3.scatter(
            xs, prec_c[by_freq], alpha=0.5, color="tab:orange", label="Precision"
        )
        ax3.scatter(xs, recall_c[by_freq], alpha=0.5, color="tab:green", label="Recall")
        ax3.legend()
        ax3.set_ylabel("Precision/Recall")
        ax3.set_ylim(0, 1)

        ax4.scatter(xs, f1_c[by_freq], alpha=0.5)
        ax4.set_ylabel("F1")
        ax4.set_ylim(0, 1)

        ax5.scatter(xs, val_base_ce[by_freq], alpha=0.5)
        ax5.set_ylabel("Baseline Loss")
        ax5.set_yscale("log")
        ax5.set_ylim(1e-4, 0.5)

        ax6.scatter(xs, val_ce[by_freq], alpha=0.5, label="ADE20K Classes")
        ax6.set_ylabel("Probe Loss")
        ax6.set_yscale("log")
        ax6.set_ylim(1e-4, 0.5)

        for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
            ax.grid(True, linewidth=0.3, alpha=0.5)
            ax.spines[["right", "top"]].set_visible(False)

        ax5.set_xlabel("Number of Training Samples")
        ax6.set_xlabel("Number of Training Samples")

        ax1.set_xscale("log")
        ax6.legend()

        fig.suptitle("Best ViT-B/16 Checkpoint on ADE20K")

        return fig

    plot_prevalence_vs_metric()
    return (get_class_prevalence,)


@app.cell
def _(get_baseline_ce, mode, np, plt, runs_root, saev, shards_root):
    def plot_latent_vs_purity(*, k: int = 32):
        fig, ax = plt.subplots(
            figsize=(4.5, 3),
            nrows=1,
            ncols=1,
            dpi=300,
            layout="constrained",
            sharex=True,
        )

        # ViT-S/16
        train_shards = "781f8739"
        val_shards = "5e195bbf"
        run = saev.disk.Run(runs_root / "gc6iqrf2")

        # ViT-B/16
        # train_shards = "fd5781e8"
        # val_shards = "66a5d2c1"
        # run = saev.disk.Run(runs_root / "qoc1660r")

        # ViT-L/16
        train_shards = "614861a0"
        val_shards = "3802cb66"
        run = saev.disk.Run(runs_root / "mccrm7u8")

        # ViT-L/16 (Vanilla)
        train_shards = "614861a0"
        val_shards = "3802cb66"
        run = saev.disk.Run(runs_root / "0pdum8cq")

        with np.load(run.inference / train_shards / "probe1d_metrics.npz") as fd:
            train_loss = fd["loss"]

        with np.load(run.inference / val_shards / "probe1d_metrics.npz") as fd:
            val_loss = fd["loss"]

        assert train_loss.ndim == 2
        assert val_loss.ndim == 2
        assert train_loss.shape == val_loss.shape

        n_latents, n_classes = train_loss.shape

        best_i = np.argmin(train_loss, axis=0)
        train_ce = train_loss[best_i, np.arange(n_classes)]
        val_ce = val_loss[best_i, np.arange(n_classes)]

        train_base_ce = get_baseline_ce(shards_root / train_shards)
        val_base_ce = get_baseline_ce(shards_root / val_shards)

        path = run.inference / val_shards / f"probe1d_metrics__train-{train_shards}.npz"
        assert path.exists()
        with np.load(path) as fd:
            ap_c = fd["ap"]
            prec_c = fd["precision"]
            recall_c = fd["recall"]
            f1_c = fd["f1"]
            top_labels_dk = fd["top_labels"]

        d_sae, _ = top_labels_dk.shape

        _, counts_d = mode(top_labels_dk[:, :k], axis=1)
        purity_d = counts_d / k
        ax.scatter(np.arange(d_sae), purity_d, alpha=0.2, marker=".", s=8)
        return fig

    plot_latent_vs_purity(k=256)
    return


@app.cell
def _(beartype, mo, np, pathlib, saev):
    @mo.cache
    @beartype.beartype
    def get_baseline_ce(shards_dir: pathlib.Path):
        md = saev.data.Metadata.load(shards_dir)
        labels = np.memmap(
            shards_dir / "labels.bin",
            mode="r",
            dtype=np.uint8,
            shape=(md.n_examples, md.content_tokens_per_example),
        )

        n_samples = np.prod(labels.shape)

        n_classes = labels.max() + 1
        # Convert to one-hot encoding
        y = np.zeros((n_samples, n_classes), dtype=float)
        y[np.arange(n_samples), labels.reshape(n_samples)] = 1.0

        prob = y.mean(axis=0)
        return -(prob * np.log(prob) + (1 - prob) * np.log(1 - prob))

    return (get_baseline_ce,)


@app.cell
def _(WANDB_USERNAME, base64, beartype, json, os, pickle, saev, wandb):
    class MetadataAccessError(Exception):
        """Exception raised when metadata cannot be accessed or parsed."""

        pass

    @beartype.beartype
    def find_metadata(shard_root: str, wandb_metadata: dict | None = None):
        # First check if metadata is available from WandB config
        if wandb_metadata is not None:
            return wandb_metadata

        # Fall back to reading from file system for backward compatibility
        if not os.path.exists(shard_root):
            msg = f"""
    ERROR: Shard root '{shard_root}' not found. You need to either:

    1. Run this notebook on the same machine where the shards are located.
    2. Copy the shards to this machine at path: {shard_root}
    3. Update the filtering criteria to only show checkpoints with available data""".strip()
            raise MetadataAccessError(msg)

        metadata_path = os.path.join(shard_root, "metadata.json")
        if not os.path.exists(metadata_path):
            raise MetadataAccessError("Missing metadata.json file")

        try:
            with open(metadata_path) as fd:
                return json.load(fd)
        except json.JSONDecodeError:
            raise MetadataAccessError("Malformed metadata.json file")

    @beartype.beartype
    def get_wandb_run(run_id: str, *, project: str):
        run = wandb.Api(timeout=30).run(f"{WANDB_USERNAME}/{project}/{run_id}")

        row = {}
        row["id"] = run.id

        try:
            row.update(**{
                f"summary/{key}": value for key, value in run.summary.items()
            })
        except AttributeError as err:
            print(f"Run {run.id} has a problem in run.summary._json_dict: {err}")
            return {}

        # config
        row.update(**{
            f"config/train_data/{key}": value
            for key, value in run.config.pop("train_data").items()
        })
        row.update(**{
            f"config/val_data/{key}": value
            for key, value in run.config.pop("val_data").items()
        })
        if "sae" in run.config:
            row.update(**{
                f"config/sae/{key}": value
                for key, value in run.config.pop("sae").items()
            })

        if "objective" in run.config:
            row.update(**{
                f"config/objective/{key}": value
                for key, value in run.config.pop("objective").items()
            })

        row.update(**{f"config/{key}": value for key, value in run.config.items()})

        try:
            # Check if metadata is available in WandB config
            wandb_metadata = row.get("config/train_data/metadata")
            metadata = find_metadata(row["config/train_data/shards"], wandb_metadata)
        except MetadataAccessError as err:
            print(f"Bad run {run.id}: {err}")
            return {}

        row["model_key"] = get_model_key(metadata)

        data_key = get_data_key(metadata)
        if data_key is None:
            print(f"Bad run {run.id}: unknown data.")
            return {}

        row["data_key"] = data_key
        row["config/d_model"] = metadata["d_model"]

        return row

    @beartype.beartype
    def get_model_key(metadata: dict[str, object]) -> str:
        family = next(
            metadata[key]
            for key in ("vit_family", "model_family", "family")
            if key in metadata
        )

        ckpt = next(
            metadata[key]
            for key in ("vit_ckpt", "model_ckpt", "ckpt")
            if key in metadata
        )

        if family == "dinov2" and ckpt == "dinov2_vitb14_reg":
            return "DINOv2 ViT-B/14 (reg)"
        if family == "dinov2" and ckpt == "dinov2_vitl14_reg":
            return "DINOv2 ViT-L/14 (reg)"
        if family == "dinov3" and "vitl" in ckpt:
            return "DINOv3 ViT-L/16"
        if family == "dinov3" and "vitb" in ckpt:
            return "DINOv3 ViT-B/16"
        if family == "dinov3" and "vits" in ckpt:
            return "DINOv3 ViT-S/16"
        if family == "clip" and ckpt == "ViT-B-16/openai":
            return "CLIP ViT-B/16"
        if family == "clip" and ckpt == "hf-hub:imageomics/bioclip":
            return "BioCLIP ViT-B/16"
        if family == "clip" and ckpt == "hf-hub:imageomics/bioclip-2":
            return "BioCLIP 2 ViT-L/14"
        if family == "siglip" and ckpt == "hf-hub:timm/ViT-L-16-SigLIP2-256":
            return "SigLIP2 ViT-L/16"

        print(f"Unknown model: {(family, ckpt)}")
        return ckpt

    @beartype.beartype
    def get_data_key(metadata: dict[str, object]) -> str | None:
        data_cfg = pickle.loads(base64.b64decode(metadata["data"].encode("utf8")))

        if isinstance(data_cfg, saev.data.datasets.ImgSegFolder) and "ADE" in str(
            data_cfg.root
        ):
            return f"ADE20K/{data_cfg.split}"

        if isinstance(data_cfg, saev.data.datasets.Imagenet):
            return f"IN1K/{data_cfg.split}"

        if isinstance(
            data_cfg, saev.data.datasets.ImgFolder
        ) and "fish-vista-imgfolder" in str(data_cfg.root):
            return "FishVista (Img)"

        print(f"Unknown data: {data_cfg}")
        return None

    return (get_wandb_run,)


@app.cell
def _(adjust_text, beartype, df, mo, np, pl, plt):
    @beartype.beartype
    def plot_layerwise_fishvista(
        model: str, shards: str, layers: list[int], n_layers: int, show_ids: int = 0
    ):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            dpi=300,
            figsize=(6, 2.4),
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        axes = axes.reshape(-1)

        x_col = "val_nmse"

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            filtered = df.filter(
                (pl.col("model") == model)
                & (pl.col("val_probe_shards") == shards)
                & (pl.col("layer") == layer)
                & (pl.col("val_nmse") <= 1.0)
                & (pl.col("objective") == "matryoshka")
            ).sort(by=x_col)

            xs = filtered.get_column(x_col).to_numpy()
            xs = 1 - xs
            ys = filtered.get_column("val_mean_ap").to_numpy()
            ids = filtered.get_column("run_id")

            ax.set_title(f"Layer {layer + 1}/{n_layers}")

            try:
                line = np.polynomial.Polynomial.fit(xs, ys, deg=2)
                x_line, y_line = line.linspace(domain=[0, 1])
                ax.plot(
                    x_line,
                    y_line,
                    color="tab:orange",
                    linestyle=(0, (3, 1)),
                    label="$y=ax^2+bx+c$",
                    alpha=0.6,
                )
            except ValueError:
                pass

            ax.scatter(xs, ys, color="tab:blue", alpha=0.8, zorder=3, clip_on=False)

            ax.grid(True, linewidth=0.3, alpha=0.5)
            ax.spines[["right", "top"]].set_visible(False)

            ax.set_xlabel("Explained Variance")
            if i in (0,):
                ax.set_ylabel("Val mAP ($\\uparrow$)")
            if i in (2,):
                ax.legend()

            ax.set_ylim(0, 0.7)
            ax.set_xlim(0, 1.0)

            if show_ids:
                texts = []
                best_j = np.argsort(ys)[::-1][:show_ids].tolist()
                for j in best_j:
                    print(f"Run {ids[j]} had {x_col}={xs[j]:.3f} and mAP={ys[j]:.3f}")
                    txt = ax.text(
                        xs[j], ys[j], ids[j], fontsize=8, ha="left", va="bottom"
                    )
                    texts.append(txt)
                adjust_text(texts=texts, ax=ax)

        fig.suptitle(f"{model}")

        return fig

    def _():
        dinov3_fig = plot_layerwise_fishvista(
            "DINOv3 ViT-L/16", "8692dfa9", [19, 21, 23], 24, show_ids=2
        )
        dinov3_fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_fishvista_mean_ap.pdf"
        )

        bioclip2_fig = plot_layerwise_fishvista(
            "BioCLIP 2 ViT-L/14", "1bc9cc5d", [19, 21, 23], 24, show_ids=2
        )
        bioclip2_fig.savefig(
            "contrib/trait_discovery/docs/assets/bioclip2_fishvista_mean_ap.pdf"
        )

        return mo.vstack([dinov3_fig, bioclip2_fig])

    _()
    return


@app.cell
def _(
    adjust_text,
    df,
    get_baseline_ce,
    get_class_prevalence,
    mo,
    mode,
    np,
    pe,
    pl,
    plt,
    runs_root,
    saev,
    shards_root,
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

    colors = [
        saev.colors.BLACK_RGB01,
        saev.colors.BLUE_RGB01,
        saev.colors.CYAN_RGB01,
        saev.colors.SEA_RGB01,
    ]

    def plot_fishvista_best_latents():
        fig, ax = plt.subplots(
            figsize=(4.5, 3), nrows=1, ncols=1, dpi=300, layout="constrained"
        )

        run = saev.disk.Run(runs_root / "9rrslm9e")
        train_shards, val_shards = next(
            df.filter(pl.col("run_id") == run.run_id)
            .select("train_probe_shards", "val_probe_shards")
            .iter_rows()
        )

        with np.load(run.inference / train_shards / "probe1d_metrics.npz") as fd:
            train_loss = fd["loss"]

        with np.load(run.inference / val_shards / "probe1d_metrics.npz") as fd:
            val_loss = fd["loss"]

        assert train_loss.ndim == 2
        assert val_loss.ndim == 2
        assert train_loss.shape == val_loss.shape

        n_latents, n_classes = train_loss.shape

        best_i = np.argmin(train_loss, axis=0)
        train_ce = train_loss[best_i, np.arange(n_classes)]
        val_ce = val_loss[best_i, np.arange(n_classes)]

        train_base_ce = get_baseline_ce(shards_root / train_shards)
        val_base_ce = get_baseline_ce(shards_root / val_shards)

        path = run.inference / val_shards / f"probe1d_metrics__train-{train_shards}.npz"

        assert path.exists()
        with np.load(path) as fd:
            ap_c = fd["ap"]
            prec_c = fd["precision"]
            recall_c = fd["recall"]
            f1_c = fd["f1"]
            top_labels_dk = fd["top_labels"]

        k = 16
        _, counts = mode(top_labels_dk[best_i, :k], axis=1)
        print(f"Purity@{k}: {(counts / k).mean().item()}")

        # for i, latent in enumerate(best_i):
        #     print(latent, fishvista_id_to_name[i], ap_c[i].item())

        by_freq, counts = get_class_prevalence(shards_root / train_shards)
        xs = counts

        # xs = np.arange(10)

        ys = ap_c[by_freq]

        ax.scatter(xs, ys, alpha=0.8, color=np.array(saev.colors.ALL_RGB01)[by_freq])

        texts = []
        for x, y, name_i in zip(xs, ys, by_freq):
            texts.append(
                ax.text(
                    x,
                    y,
                    fishvista_names[name_i],
                    fontsize=11,
                    ha="center",
                    va="center",
                    color=saev.colors.ALL_RGB01[name_i],
                    path_effects=[
                        pe.withStroke(
                            linewidth=0.5, foreground=saev.colors.BLACK_RGB01, alpha=0.5
                        )
                    ],
                    # color=saev.colors.BLACK_RGB01,
                    # path_effects=[
                    #     pe.withStroke(
                    #         linewidth=0.5, foreground=saev.colors.ALL_RGB01[name_i]
                    #     )
                    # ],
                )
            )

        ax.set_ylabel("Average Precision")
        ax.set_ylim(0, 1)
        ax.set_xscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Number of Samples")

        adjust_text(texts)

        fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_fishvista_prevalence_ap.pdf"
        )

        lines = [
            f"Run ID: {run.run_id}",
            f"Val Shards: {val_shards}",
            f"Latents: {' '.join(map(str, best_i))}",
        ]
        return mo.vstack([fig, mo.md("\n".join(f"- {line}" for line in lines))])

    plot_fishvista_best_latents()
    return


@app.cell
def _(df, mo, pl):
    mo.vstack([
        mo.md("# FishVista Tables"),
        df.filter(
            (pl.col("model") == "DINOv3 ViT-L/16")
            & (pl.col("val_probe_shards") == "8692dfa9")
            & (pl.col("objective") == "matryoshka")
        )
        .sort(by="val_probe_r", descending=True)
        .head(2)
        .select(
            "run_id",
            "model",
            "layer",
            "train_probe_shards",
            "objective",
            "sae_val_mse",
            "sae_val_l0",
            "val_probe_r",
            "val_nmse",
            "val_mean_ap",
            "cov_at_0_3",
            "cov_at_0_5",
            "cov_at_0_7",
        ),
    ])
    return


@app.cell
def _():
    BLACK_HEX = "001219"
    BLACK_RGB = (0, 18, 25)
    BLACK_RGB01 = tuple(c / 256 for c in BLACK_RGB)

    BLUE_HEX = "005f73"
    BLUE_RGB = (0, 95, 115)
    BLUE_RGB01 = tuple(c / 256 for c in BLUE_RGB)

    CYAN_HEX = "0a9396"
    CYAN_RGB = (10, 147, 150)
    CYAN_RGB01 = tuple(c / 256 for c in CYAN_RGB)

    SEA_HEX = "94d2bd"
    SEA_RGB = (148, 210, 189)
    SEA_RGB01 = tuple(c / 256 for c in SEA_RGB)

    CREAM_HEX = "e9d8a6"
    CREAM_RGB = (233, 216, 166)
    CREAM_RGB01 = tuple(c / 256 for c in CREAM_RGB)

    GOLD_HEX = "ee9b00"
    GOLD_RGB = (238, 155, 0)
    GOLD_RGB01 = tuple(c / 256 for c in GOLD_RGB)

    ORANGE_HEX = "ca6702"
    ORANGE_RGB = (202, 103, 2)
    ORANGE_RGB01 = tuple(c / 256 for c in ORANGE_RGB)

    RUST_HEX = "bb3e03"
    RUST_RGB = (187, 62, 3)
    RUST_RGB01 = tuple(c / 256 for c in RUST_RGB)

    SCARLET_HEX = "ae2012"
    SCARLET_RGB = (174, 32, 18)
    SCARLET_RGB01 = tuple(c / 256 for c in SCARLET_RGB)

    RED_HEX = "9b2226"
    RED_RGB = (155, 34, 38)
    RED_RGB01 = tuple(c / 256 for c in RED_RGB)
    return BLUE_RGB01, ORANGE_RGB01, SCARLET_RGB01, SEA_RGB01


@app.cell
def _(np):
    def mode(a, axis=0):
        scores = np.unique(np.ravel(a))  # get ALL unique values
        testshape = list(a.shape)
        testshape[axis] = 1
        oldmostfreq = np.zeros(testshape)
        oldcounts = np.zeros(testshape)

        for score in scores:
            template = a == score
            counts = np.expand_dims(np.sum(template, axis), axis)
            mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
            oldcounts = np.maximum(counts, oldcounts)
            oldmostfreq = mostfrequent

        return mostfrequent, oldcounts

    return (mode,)


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(
    BLUE_RGB01,
    ORANGE_RGB01,
    SCARLET_RGB01,
    SEA_RGB01,
    beartype,
    df,
    pl,
    plt,
):
    @beartype.beartype
    def _():
        fig, ax1 = plt.subplots(
            nrows=1, ncols=1, dpi=300, figsize=(4.5, 3.5), layout="constrained"
        )

        alpha = 0.8

        handles = []

        y_col = "val_probe_r"
        # y_col = "val_mean_ap"
        # y_col = "cov_at_0_3"
        # y_col = "val_mean_purity_16"

        kmeans_ys = {
            "val_probe_r": [0.0018156593136140486],
            "val_mean_ap": [0.03009829670190811],
            "cov_at_0_3": [0.026490066225165563],
            "val_mean_purity_16": [0.7959437086092715],
        }
        xs = [1.0]
        handle = ax1.scatter(
            xs,
            kmeans_ys[y_col],
            color=SEA_RGB01,
            alpha=alpha,
            marker="x",
            linewidth=2,
            label="$k$-Means",
        )
        handles.append(handle)

        pca_ys = {
            "val_probe_r": [
                0.25954435989504354,
                0.23698444965787047,
                0.209391147799891,
                0.12567529411122225,
                0.03843596455449838,
            ],
            "val_mean_ap": [
                0.09314471483230591,
                0.06568679213523865,
                0.04002770781517029,
                0.026518363505601883,
                0.011821510270237923,
            ],
            "cov_at_0_3": [
                0.07947019867549669,
                0.07947019867549669,
                0.026490066225165563,
                0.019867549668874173,
                0.006622516556291391,
            ],
            "val_mean_purity_16": [
                0.6121688741721855,
                0.5918874172185431,
                0.7591059602649006,
                0.8807947019867549,
                0.6875,
            ],
        }
        xs = [256.0, 64.0, 16.0, 4.0, 1.0]
        handle = ax1.scatter(
            xs, pca_ys[y_col], color=ORANGE_RGB01, alpha=alpha, marker="^", label="PCA"
        )
        handles.append(handle)

        grouped = df.filter(
            (pl.col("model") == "DINOv3 ViT-L/16")
            & (pl.col("val_probe_shards") == "3802cb66")
            & (pl.col("objective") == "vanilla")
            & (pl.col("layer") == 23)
            & (pl.col("val_probe_r") >= 0)
        ).sort(by="sae_val_l0")

        ys = grouped.get_column(y_col).to_numpy()
        xs = grouped.get_column("sae_val_l0").to_numpy() + 1

        handle = ax1.scatter(
            xs, ys, color=BLUE_RGB01, alpha=alpha, marker="^", label="SAE"
        )
        handles.append(handle)

        grouped = df.filter(
            (pl.col("model") == "DINOv3 ViT-L/16")
            & (pl.col("val_probe_shards") == "3802cb66")
            & (pl.col("objective") == "matryoshka")
            & (pl.col("layer") == 23)
            & (pl.col("val_probe_r") >= 0)
        ).sort(by="sae_val_l0")

        ys = grouped.get_column(y_col).to_numpy()
        xs = grouped.get_column("sae_val_l0").to_numpy() + 1
        print("DINOv3 ViT-L/16", xs.tolist(), ys.tolist())

        handle = ax1.scatter(
            xs, ys, color=SCARLET_RGB01, alpha=0.8, marker="s", label="Matryoshka"
        )
        handles.append(handle)

        ax1.grid(True, linewidth=0.3, alpha=0.5)
        ax1.set_xlabel("L$_0$")
        ax1.set_ylabel("Probe $R$ ($\\uparrow$)")
        # ax1.set_ylabel("mAP ($\\uparrow$)")
        # ax1.set_ylabel("Coverage@$\\tau$ ($\\uparrow$)")
        # ax1.set_ylabel("Purity@$k$ ($\\uparrow$)")

        ax1.spines[["right", "top"]].set_visible(False)
        ax1.legend(handles=handles)
        # ax1.set_ylim(-0.025, 0.525)
        ax1.set_ylim(-0.025, 0.925)
        ax1.set_xlim(-15, 415)

        fig.savefig(
            "contrib/trait_discovery/docs/assets/dinov3_in1k_ade20k_methods_r.pdf"
        )
        # fig.savefig("contrib/trait_discovery/docs/assets/dinov3_in1k_ade20k_methods_map.pdf")
        # fig.savefig(
        #     "contrib/trait_discovery/docs/assets/dinov3_in1k_ade20k_methods_cov.pdf"
        # )
        # fig.savefig(
        #     "contrib/trait_discovery/docs/assets/dinov3_in1k_ade20k_methods_purity.pdf"
        # )
        return fig

    _()
    return


@app.cell
def _(df, pl):
    def _():
        for model, shards in [
            ("DINOv3 ViT-S/16", "5e195bbf"),
            ("DINOv3 ViT-B/16", "66a5d2c1"),
            ("DINOv3 ViT-L/16", "3802cb66"),
        ]:
            for model, layer, nmse, l0, probe_r, map, purity, cov in (
                (
                    df.filter(
                        (pl.col("model") == model)
                        & (pl.col("val_probe_shards") == shards)
                        & (pl.col("objective") == "matryoshka")
                    )
                    .with_columns(
                        (pl.col("train_probe_r") == pl.col("train_probe_r").max())
                        .over("layer")
                        .alias("best_probe_r")
                    )
                    .filter(pl.col("best_probe_r"))
                    .sort(by="layer")
                )
                .select(
                    "model",
                    pl.col("layer") + 1,
                    "val_nmse",
                    "sae_val_l0",
                    "val_probe_r",
                    "val_mean_ap",
                    "val_mean_purity_16",
                    "cov_at_0_3",
                )
                .iter_rows()
            ):
                print(
                    f"{model[7:]} & {layer} & {nmse:.3f} & {l0:.1f} & {probe_r:.3f} & {map:.3f} & {purity:.3f} & {cov:.3f} \\\\"
                )

    _()
    return


if __name__ == "__main__":
    app.run()
