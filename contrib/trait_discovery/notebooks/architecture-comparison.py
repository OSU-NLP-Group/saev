import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import concurrent.futures
    import json
    import os.path
    import pickle

    import beartype
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import tqdm
    import wandb
    from jaxtyping import Float, jaxtyped

    import saev.data.datasets

    return (
        Float,
        base64,
        beartype,
        concurrent,
        jaxtyped,
        json,
        mo,
        np,
        os,
        pickle,
        pl,
        plt,
        saev,
        wandb,
    )


@app.cell
def _():
    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "saev"
    WANDB_TAG = "fishvista-v0.4.3"
    return WANDB_PROJECT, WANDB_TAG, WANDB_USERNAME


@app.cell
def _(
    WANDB_PROJECT,
    WANDB_TAG,
    WANDB_USERNAME,
    beartype,
    concurrent,
    get_data_key,
    get_model_key,
    load_freqs,
    load_mean_values,
    mo,
    pl,
    wandb,
):
    @beartype.beartype
    def _row_from_run(run) -> dict[str, object] | None:
        row = {"id": run.id}

        row.update(**{f"summary/{key}": value for key, value in run.summary.items()})

        try:
            row["summary/eval/freqs"] = load_freqs(run)
        except Exception as err:
            print(f"Run {run.id} failed loading freqs: {err}")
            return None

        try:
            row["summary/eval/mean_values"] = load_mean_values(run)
        except Exception as err:
            print(f"Run {run.id} failed loading mean values: {err}")
            return None

        config = dict(run.config)

        try:
            train_data = config.pop("train_data")
            val_data = config.pop("val_data")
            sae_cfg = config.pop("sae")
            obj_cfg = config.pop("objective")
        except KeyError as err:
            print(f"Run {run.id} missing config section: {err}.")
            return None

        row.update(**{
            f"config/train_data/{key}": value for key, value in train_data.items()
        })
        row.update(**{
            f"config/val_data/{key}": value for key, value in val_data.items()
        })
        row.update(**{f"config/sae/{key}": value for key, value in sae_cfg.items()})
        row.update(**{
            f"config/objective/{key}": value for key, value in obj_cfg.items()
        })

        row.update(**{f"config/{key}": value for key, value in config.items()})

        metadata = row.get("config/train_data/metadata")
        assert metadata is not None, f"Run {run.id} missing metadata"
        row["model_key"] = get_model_key(metadata)
        row["data_key"] = get_data_key(metadata)
        row["config/d_model"] = metadata["d_model"]
        return row

    @beartype.beartype
    def _finalize_df(rows: list[dict[str, object]]):
        df = pl.DataFrame(rows, infer_schema_length=None)

        activation_schema = pl.Struct([
            pl.Field("sparsity", pl.Struct([pl.Field("coeff", pl.Float64)])),
            pl.Field("top_k", pl.Int64),
        ])

        df = df.with_columns(
            pl.col("config/sae/activation").cast(
                activation_schema, strict=False
            )  # add missing fields as null
        )

        df = df.with_columns(
            pl.when(
                pl.col("config/sae/activation").struct.field("sparsity").is_not_null()
            )
            .then(pl.lit("relu"))
            .otherwise(pl.lit("topk"))
            .alias("config/sae/activation_kind"),
        )

        group_cols = (
            "model_key",
            "config/val_data/layer",
            "data_key",
            "config/sae/activation_kind",
        )
        lr_col = "config/lr"
        # lambda_col = "config/objective/sparsity_coeff"

        bounds_df = df.group_by(group_cols, maintain_order=False).agg(
            pl.col(lr_col).min().alias("lr_min"),
            pl.col(lr_col).max().alias("lr_max"),
        )

        df = (
            df.join(bounds_df, on=group_cols, how="left")
            .with_columns(
                (pl.col(lr_col) == pl.col("lr_min")).alias("is_lr_min"),
                (pl.col(lr_col) == pl.col("lr_max")).alias("is_lr_max"),
                # (pl.col(lambda_col) == pl.col("lambda_min")).alias("is_lambda_min"),
                # (pl.col(lambda_col) == pl.col("lambda_max")).alias("is_lambda_max"),
            )
            .sort(group_cols + ("summary/eval/l0", "summary/eval/mse"))
            .with_columns(
                (
                    pl.col("summary/eval/mse")
                    == pl.col("summary/eval/mse").cum_min().over(group_cols)
                ).alias("is_pareto"),
            )
        )

        return df

    @beartype.beartype
    def make_df_parallel(n_workers: int = 16):
        filters = {}

        filters["config.tag"] = WANDB_TAG
        runs = list(
            wandb.Api().runs(path=f"{WANDB_USERNAME}/{WANDB_PROJECT}", filters=filters)
        )
        if not runs:
            raise ValueError("No runs found.")

        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            fut_to_run_id = {pool.submit(_row_from_run, run): run.id for run in runs}
            for fut in mo.status.progress_bar(
                concurrent.futures.as_completed(fut_to_run_id),
                total=len(fut_to_run_id),
                remove_on_exit=True,
            ):
                try:
                    row = fut.result()
                except Exception as err:
                    print(f"Run {fut_to_run_id[fut]} blew up: {err}")
                    continue
                if row is None:
                    continue
                rows.append(row)

        assert rows, "No valid runs loaded in parallel."
        return _finalize_df(rows)

    df = make_df_parallel()
    return (df,)


@app.cell
def _(df, pl, plt):
    def _():
        fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300, layout="constrained")
        ks, ys, ids = (
            df.filter(pl.col("config/sae/activation_kind") == "topk")
            .group_by(pl.col("config/sae/activation").struct.field("top_k"))
            .agg(pl.col("summary/eval/l0"), pl.col("id"))
            .sort(by="top_k")
        )

        # ax.boxplot(ys)
        ax.violinplot(ys)
        for k, y in zip([32, 128, 512], ys):
            ax.axhline(k, color="tab:red", alpha=0.5)
            print(y.argmax())
        ax.set_xticks([1, 2, 3], ["$k$=32", "$k$=128", "$k$=512"])
        ax.set_yscale("log")
        ax.set_ylabel("Observed L$_0$")
        ax.spines[["top", "right"]].set_visible(False)

        return fig

    _()
    return


@app.cell
def _(df, pl):
    df.filter(pl.col("config/sae/activation_kind") == "topk").select(
        pl.col("config/sae/activation").struct.field("top_k"), "summary/eval/l0"
    )
    return


@app.cell
def _(df, pl, plt):
    def _(df: pl.DataFrame):
        # mse_col = "ade20k_val_nmse"
        l0_col = "summary/eval/l0"
        mse_col = "summary/eval/mse"
        layer_col = "config/val_data/layer"

        fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300, layout="constrained")
        texts = []

        point_alpha = 0.8

        activation_fns = [("relu", "tab:blue", "o"), ("topk", "tab:orange", "^")]

        for kind, color, marker in activation_fns:
            group = df.filter((pl.col("config/sae/activation_kind") == kind))

            pareto = group.filter(pl.col("is_pareto"))
            if pareto.height == 0:
                continue

            ids = pareto.get_column("id").to_list()
            xs = pareto.get_column(l0_col).to_numpy()
            ys = pareto.get_column(mse_col).to_numpy()

            line, *_ = ax.plot(
                xs, ys, alpha=0.8, label=f"{kind}", color=color, marker=marker
            )

            edge_mask = pl.col("is_lr_min") | pl.col("is_lr_max")
            edge_df = pareto.filter(edge_mask)

            if edge_df.height > 0:
                edge_xs = edge_df.get_column(l0_col).to_numpy()
                edge_ys = edge_df.get_column(mse_col).to_numpy()
                ax.scatter(
                    edge_xs,
                    edge_ys,
                    facecolors="none",
                    edgecolors="tab:red",
                    marker=marker,
                    s=60,
                    linewidths=1.2,
                    zorder=line.get_zorder() + 1,
                )

            lr_min = pareto.get_column("is_lr_min").to_list()
            lr_max = pareto.get_column("is_lr_max").to_list()

            for x, y, rid, is_lr_min, is_lr_max in zip(xs, ys, ids, lr_min, lr_max):
                edge_parts = []
                if is_lr_min:
                    edge_parts.append("LR min")
                if is_lr_max:
                    edge_parts.append("LR max")

                label = rid if not edge_parts else f"{rid} ({', '.join(edge_parts)})"
                color_text = "tab:red" if edge_parts else "black"
                texts.append(
                    ax.text(
                        x,
                        y,
                        label,
                        fontsize=4,
                        color=color_text,
                        ha="left",
                        va="bottom",
                    )
                )

        ax.set_xlabel("L$_0$ ($\\downarrow$)")
        ax.set_ylabel("MSE ($\\downarrow$)")
        ax.grid(True, linewidth=0.3, alpha=0.7)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()

        return fig

    _(df.filter((pl.col("config/val_data/layer") == 23)))
    return


@app.cell
def _(Float, beartype, jaxtyped, json, np, os):
    @jaxtyped(typechecker=beartype.beartype)
    def load_freqs(run) -> Float[np.ndarray, " d_sae"]:
        try:
            for artifact in run.logged_artifacts():
                if "evalfreqs" not in artifact.name:
                    continue

                dpath = artifact.download()
                fpath = os.path.join(dpath, "eval", "freqs.table.json")
                print(fpath)
                with open(fpath) as fd:
                    raw = json.load(fd)
                return np.array(raw["data"], dtype=float).reshape(-1)
        except Exception as err:
            raise RuntimeError(f"Wandb sucks: {err}") from err

        raise ValueError(f"freqs not found in run '{run.id}'")

    @jaxtyped(typechecker=beartype.beartype)
    def load_mean_values(run) -> Float[np.ndarray, " d_sae"]:
        try:
            for artifact in run.logged_artifacts():
                if "evalmean_values" not in artifact.name:
                    continue

                dpath = artifact.download()
                fpath = os.path.join(dpath, "eval", "mean_values.table.json")
                print(fpath)
                with open(fpath) as fd:
                    raw = json.load(fd)
                return np.array(raw["data"], dtype=float).reshape(-1)
        except Exception as err:
            raise RuntimeError(f"Wandb sucks: {err}") from err

        raise ValueError(f"mean_values not found in run '{run.id}'")

    return load_freqs, load_mean_values


@app.cell
def _(base64, beartype, pickle, saev):
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

    return get_data_key, get_model_key


if __name__ == "__main__":
    app.run()
