import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import collections
    import concurrent.futures
    import dataclasses
    import itertools
    import json
    import os.path
    import pathlib
    import pickle

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import scipy.sparse
    import wandb
    from jaxtyping import Bool, Float, jaxtyped

    import saev.colors
    import saev.data.datasets

    return (
        Bool,
        Float,
        base64,
        beartype,
        collections,
        concurrent,
        dataclasses,
        itertools,
        jaxtyped,
        json,
        mo,
        np,
        os,
        pathlib,
        pickle,
        pl,
        plt,
        saev,
        scipy,
        wandb,
    )


@app.cell
def _(pathlib):
    WANDB_USERNAME = "samuelstevens"
    WANDB_PROJECT = "saev"
    WANDB_TAG = "fishbase-v0.1"

    runs_root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
    shards_root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")
    return WANDB_PROJECT, WANDB_TAG, WANDB_USERNAME, runs_root, shards_root


@app.cell
def _(
    WANDB_PROJECT,
    WANDB_TAG,
    WANDB_USERNAME,
    beartype,
    concurrent,
    get_cls_results,
    get_data_key,
    get_model_key,
    load_freqs,
    load_mean_values,
    mo,
    pl,
    runs_root,
    saev,
    wandb,
):
    @beartype.beartype
    def _row_from_run(wandb_run) -> dict[str, object] | None:
        saev_run = saev.disk.Run(runs_root / wandb_run.id)
        row = {"id": wandb_run.id}

        row.update(**{
            f"summary/{key}": value for key, value in wandb_run.summary.items()
        })

        try:
            row["summary/eval/freqs"] = load_freqs(wandb_run)
        except Exception as err:
            print(f"Run {wandb_run.id} failed loading freqs: {err}")
            return None

        try:
            row["summary/eval/mean_values"] = load_mean_values(wandb_run)
        except Exception as err:
            print(f"Run {wandb_run.id} failed loading mean values: {err}")
            return None

        config = dict(wandb_run.config)

        try:
            train_data = config.pop("train_data")
            val_data = config.pop("val_data")
        except KeyError as err:
            print(f"Run {wandb_run.id} missing config section: {err}.")
            return None

        row.update(**{
            f"config/train_data/{key}": value for key, value in train_data.items()
        })
        row.update(**{
            f"config/val_data/{key}": value for key, value in val_data.items()
        })
        row.update(**{f"config/{key}": value for key, value in config.items()})

        metadata = row.get("config/train_data/metadata")
        assert metadata is not None, f"Run {wandb_run.id} missing metadata"

        row["model_key"] = get_model_key(metadata)
        row["data_key"] = get_data_key(metadata)

        row.update(**get_cls_results(saev_run))

        return row

    @beartype.beartype
    def _finalize_df(rows: list[dict[str, object]]):
        df = pl.DataFrame(rows, infer_schema_length=None)

        # activation_schema = pl.Struct([
        #     pl.Field("sparsity", pl.Struct([pl.Field("coeff", pl.Float64)])),
        #     pl.Field("top_k", pl.Int64),
        # ])

        # df = df.with_columns(
        #     pl.col("config/sae/activation").cast(
        #         activation_schema, strict=False
        #     )  # add missing fields as null
        # )

        # df = df.with_columns(
        #     pl.when(pl.col("config/sae/activation").struct.field("top_k").is_not_null())
        #     .then(pl.lit("topk"))
        #     .otherwise(pl.lit("relu"))
        #     .alias("config/sae/activation_kind"),
        # )

        group_cols = (
            "model_key",
            "config/val_data/layer",
            "data_key",
            "config/sae/activation/key",
            "config/sae/activation/aux/key",
            "config/sae/reinit_blend",
            "config/optim",
        )

        df = (
            df
            .unnest("config/sae", "config/train_data/metadata", separator="/")
            .unnest("config/sae/activation", separator="/")
            .unnest(
                "config/sae/activation/aux",
                "config/sae/activation/sparsity",
                separator="/",
            )
        )

        # Compute Pareto explicitly
        x_col = "summary/eval/l0"
        y_col = "summary/eval/normalized_mse"
        pareto_ids = set()
        for keys, group_df in df.group_by(group_cols):
            group_df = group_df.filter(
                pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null()
            ).sort(x_col, y_col)

            if group_df.height == 0:
                continue

            ids = group_df.get_column("id").to_list()
            ys = group_df.get_column(y_col).to_list()

            min_y = float("inf")
            for rid, y in zip(ids, ys):
                if y < min_y:
                    pareto_ids.add(rid)
                    min_y = y

        df = df.with_columns(pl.col("id").is_in(pareto_ids).alias("is_pareto"))

        return df

    @beartype.beartype
    def make_df_parallel(n_workers: int = 16):
        filters = {}

        filters["config.tag"] = WANDB_TAG

        runs = list(
            wandb.Api().runs(path=f"{WANDB_USERNAME}/{WANDB_PROJECT}", filters=filters)
        )
        # runs = runs[:8]

        runs = [wandb.Api().run(path=f"{WANDB_USERNAME}/{WANDB_PROJECT}/pdikj9bl")]
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

        assert rows, "No valid runs."
        return _finalize_df(rows)

    df = make_df_parallel()
    return (df,)


@app.cell
def _(df):
    df.select("^downstream/*")
    return


@app.cell
def _(collections, df, mo, pl, plt, saev):
    def _(df: pl.DataFrame):
        # mse_col = "ade20k_val_nmse"
        x_col = "config/sae/activation/top_k"
        y_col = "summary/eval/normalized_mse"

        layer_col = "config/val_data/layer"

        point_alpha = 0.5

        layers = [13, 15, 17, 19, 21, 23]

        fig, axes = plt.subplots(
            figsize=(8, 5),
            nrows=2,
            ncols=3,
            dpi=300,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes = axes.reshape(-1)

        pareto_ckpts = collections.defaultdict(list)

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            texts = []

            group = df.filter(
                (pl.col("config/sae/activation/key") == "top-k")
                & (pl.col("config/sae/reinit_blend") == 0.8)
                & (pl.col("config/val_data/layer") == layer)
                & (pl.col("config/sae/activation/aux/key") == "auxk")
                & (pl.col("data_key") == "FishVista (Img)")
            )
            group = group.sort(by=x_col)

            pareto = group.filter(pl.col("is_pareto"))
            if pareto.height == 0:
                continue

            ids = pareto.get_column("id").to_list()
            xs = pareto.get_column(x_col).to_numpy()
            ys = pareto.get_column(y_col).to_numpy()

            line, *_ = ax.plot(
                xs,
                ys,
                alpha=0.5,
                label="DINOv3",
                color=saev.colors.BLUE_RGB01,
                marker="o",
            )

            for x_true, id, (x_obs, n) in zip(
                xs, ids, group.group_by(x_col).agg(pl.len()).sort(by=x_col).iter_rows()
            ):
                assert x_true == x_obs
                if n == 6:
                    pareto_ckpts[layer].append(id)

            if i in (9, 10, 11):
                ax.set_xlabel("L$_0$ ($\\downarrow$)")

            if i in (0, 3, 6, 9):
                ax.set_ylabel("Normalized MSE ($\\downarrow$)")

            # if i in (0,):
            #     ax.legend()

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"Layer {layer + 1}/24")

        fig.savefig("contrib/trait_discovery/docs/assets/fishbase-pareto-mse.pdf")
        # return fig

        return mo.vstack([fig, dict(pareto_ckpts)])

    _(df)
    return


@app.cell
def _(df, pl):
    df.filter(
        (pl.col("config/sae/activation/key") == "top-k")
        & (pl.col("config/sae/reinit_blend") == 0.8)
        & (pl.col("config/val_data/layer") == 17)
        & (pl.col("data_key") == "FishVista (Img)")
        & (pl.col("config/sae/activation/top_k") == 128)
        & (pl.col("downstream/val/cls_results").is_not_null())
    )
    return


@app.cell
def _(df, pl, plt, saev):
    def _(df: pl.DataFrame):
        # mse_col = "ade20k_val_nmse"
        x_col = "config/lr"
        y_col = "summary/eval/normalized_mse"

        layer_col = "config/val_data/layer"

        point_alpha = 0.5

        layers = [13, 15, 17, 19, 21, 23]
        ks = (16, 32, 64, 128, 256)

        colors = [
            saev.colors.BLUE_RGB01,
            saev.colors.SEA_RGB01,
            saev.colors.ORANGE_RGB01,
            saev.colors.SCARLET_RGB01,
        ]

        fig, axes = plt.subplots(
            figsize=(8, 5),
            nrows=2,
            ncols=3,
            dpi=300,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes = axes.reshape(-1)

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            for k, color in zip(ks, saev.colors.ALL_RGB01[1:]):
                group = df.filter(
                    (pl.col("config/sae/activation/key") == "top-k")
                    & (pl.col("config/sae/reinit_blend") == 0.8)
                    & (pl.col("config/val_data/layer") == layer)
                    & (pl.col("data_key") == "FishVista (Img)")
                    & (pl.col("config/sae/activation/top_k") == k)
                )
                group = group.sort(by=x_col)

                if group.height == 0:
                    continue

                ids = group.get_column("id").to_list()
                xs = group.get_column(x_col).to_numpy()
                ys = group.get_column(y_col).to_numpy()

                line, *_ = ax.plot(
                    xs,
                    ys,
                    alpha=0.5,
                    color=color,
                    label=f"$k={k}$",
                    marker="o",
                )

            if i in (3, 4, 5):
                ax.set_xlabel("Learning Rate")

            if i in (0, 3):
                ax.set_ylabel("Normalized MSE ($\\downarrow$)")

            if i in (2,):
                ax.legend()

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.set_title(f"Layer {layer + 1}/24")

        fig.savefig("contrib/trait_discovery/docs/assets/birdclef-lr-mse.pdf")
        # return fig

        return fig

    _(df)
    return


@app.cell
def _(df, mo, pl, plt, saev):
    def _(df: pl.DataFrame):
        # mse_col = "ade20k_val_nmse"
        x_col = "config/sae/activation/top_k"
        y_col = "downstream/val/probe_r"

        layer_col = "config/val_data/layer"

        point_alpha = 0.5

        layers = [13, 15, 17, 19, 21, 23]

        fig, axes = plt.subplots(
            figsize=(8, 5),
            nrows=2,
            ncols=3,
            dpi=300,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes = axes.reshape(-1)

        for i, (layer, ax) in enumerate(zip(layers, axes)):
            texts = []

            group = df.filter(
                (pl.col("config/sae/activation/key") == "top-k")
                & (pl.col("config/sae/reinit_blend") == 0.8)
                & (pl.col("config/val_data/layer") == layer)
                & (pl.col("config/sae/activation/aux/key") == "auxk")
                & (pl.col("data_key") == "FishVista (Img)")
            )
            group = group.sort(by=x_col)

            pareto = group.filter(pl.col("is_pareto"))
            if pareto.height == 0:
                continue

            ids = pareto.get_column("id").to_list()
            xs = pareto.get_column(x_col).to_numpy()
            ys = pareto.get_column(y_col).to_numpy()

            line, *_ = ax.plot(
                xs,
                ys,
                alpha=0.5,
                label="DINOv3",
                color=saev.colors.BLUE_RGB01,
                marker="o",
            )

            if i in (9, 10, 11):
                ax.set_xlabel("L$_0$ ($\\downarrow$)")

            if i in (0, 3, 6, 9):
                ax.set_ylabel("Probe $R$ ($\\uparrow$)")

            # if i in (0,):
            #     ax.legend()

            ax.grid(True, linewidth=0.3, alpha=0.7)
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xscale("log")
            # ax.set_yscale("log")
            ax.set_title(f"Layer {layer + 1}/24")

        fig.savefig("contrib/trait_discovery/docs/assets/fishbase-pareto-mse.pdf")
        # return fig

        return mo.vstack([fig])

    _(df)
    return


@app.cell
def _(pl):
    habitat_cols = [
        "reef-associated",
        "pelagic-oceanic",
        "pelagic-neritic",
        "bathypelagic",
        "bathydemersal",
        "benthopelagic",
        "pelagic",
        "epipelagic",
        "mesopelagic",
        "abyssopelagic",
        "demersal",
    ]

    HabitatEnum = pl.Enum(habitat_cols)

    migration_cols = [
        "amphidromous",
        "anadromous",
        "catadromous",
        "limnodromous",
        "non-migratory",
        "oceanodromous",
        "potamodromous",
    ]

    MigrationEnum = pl.Enum(migration_cols)

    fishbase_df = (
        pl
        .read_csv(
            "contrib/trait_discovery/data/fishvista_fishbase.csv",
            null_values=["?"],
            # Order matters
            schema=pl.Schema({
                "genus": pl.String,
                "species": pl.String,
                "family": pl.String,
                "demersal": pl.Float64,
                "benthopelagic": pl.Float64,
                "bathydemersal": pl.Float64,
                "pelagic": pl.Float64,
                "pelagic-neritic": pl.Float64,
                "pelagic-oceanic": pl.Float64,
                "reef-associated": pl.Float64,
                "epipelagic": pl.Float64,
                "mesopelagic": pl.Float64,
                "bathypelagic": pl.Float64,
                "abyssopelagic": pl.Float64,
                "marine": pl.Float64,
                "freshwater": pl.Float64,
                "brackish": pl.Float64,
                "anadromous": pl.Float64,
                "catadromous": pl.Float64,
                "amphidromous": pl.Float64,
                "potamodromous": pl.Float64,
                "limnodromous": pl.Float64,
                "oceanodromous": pl.Float64,
                "non-migratory": pl.Float64,
                "min_depth_m": pl.Float64,
                "max_depth_m": pl.Float64,
                "usual_min_depth_m": pl.Float64,
                "usual_max_depth_m": pl.Float64,
                "min_ph": pl.Float64,
                "max_ph": pl.Float64,
                "min_dh": pl.Float64,
                "max_dh": pl.Float64,
                "url": pl.String,
            }),
        )
        .with_columns(
            pl
            .coalesce([
                pl.when(pl.col(col) == 1.0).then(pl.lit(col)) for col in habitat_cols
            ])
            .cast(HabitatEnum)
            .alias("habitat")
        )
        .drop(habitat_cols)
        .with_columns(
            pl
            .coalesce([
                pl.when(pl.col(col) == 1.0).then(pl.lit(col)) for col in migration_cols
            ])
            .cast(MigrationEnum)
            .alias("migration")
        )
        .drop(migration_cols)
    )
    return (fishbase_df,)


@app.cell
def _(fishbase_df):
    fishbase_df  # .filter(pl.col('habitat').is_null())
    return


@app.cell
def _():
    dinov3_fishvista_val = "8692dfa9"
    dinov3_fishvista_train = "5dcb2f75"
    return (dinov3_fishvista_val,)


@app.cell
def _(base64, dataclasses, pickle, pl, saev, shards_root):
    def load_fishvista_df(shards):
        md = saev.data.Metadata.load(shards_root / shards)
        data_cfg = pickle.loads(base64.b64decode(md.data.encode("utf8")))
        ds = saev.data.datasets.get_dataset(data_cfg)

        rows = []
        for sample in ds.samples:
            dct = dataclasses.asdict(sample)
            dct["label"] = dct["label"].strip()
            names = dct["label"].split("_")
            if len(names) == 2:
                dct["family"], dct["genus"] = names
            elif len(names) == 3:
                dct["family"], dct["genus"], dct["species"] = names

            rows.append(dct)

        return pl.DataFrame(rows)

    return (load_fishvista_df,)


@app.cell
def _(
    dinov3_fishvista_val,
    fishbase_df,
    load_fishvista_df,
    np,
    runs_root,
    saev,
    scipy,
    shards_root,
):
    def _():
        shards = dinov3_fishvista_val
        run_id = "hfpct5ae"
        # run_id = "s465wgg4"

        token_acts_csr = scipy.sparse.load_npz(
            saev.disk.Run(runs_root / run_id).inference / shards / "token_acts.npz"
        )

        md = saev.data.Metadata.load(shards_root / shards)
        labels = np.memmap(
            shards_root / shards / "labels.bin",
            mode="r",
            dtype=np.uint8,
            shape=(md.n_examples * md.content_tokens_per_example),
        )

        species_df = load_fishvista_df(shards).join(
            fishbase_df, on=("genus", "species"), how="left"
        )
        print(species_df.get_column("habitat").dtype.categories.to_list())
        habitats = (
            species_df
            .get_column("habitat")
            .to_physical()
            .to_numpy()
            .repeat(md.content_tokens_per_example)
        )
        migration = (
            species_df
            .get_column("migration")
            .to_physical()
            .to_numpy()
            .repeat(md.content_tokens_per_example)
        )

        assert (species_df.height * md.content_tokens_per_example,) == labels.shape

        return token_acts_csr, token_acts_csr.tocsc(), species_df, labels, habitats

    token_acts_csr, token_acts_csc, fishvista_df, body_parts, habitats = _()
    return body_parts, fishvista_df, habitats, token_acts_csr


@app.cell
def _(Bool, Float, beartype, jaxtyped, np, scipy):
    @jaxtyped(typechecker=beartype.beartype)
    def fast_auc(
        activations: Float[np.ndarray, "n d_sae"], labels: Bool[np.ndarray, " n"]
    ) -> Float[np.ndarray, " d_sae"]:
        """
        activations: (n_patches, n_features) dense or convert from sparse
        labels: (n_patches,) binary
        returns: (n_features,) AUC scores
        """
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos

        # Rank activations per feature (1 = lowest)
        ranks = scipy.stats.rankdata(activations, axis=0)  # (n_patches, n_features)

        # AUC = (mean rank of positives - expected under null) / n_neg
        mean_rank_pos = ranks[labels].mean(axis=0)
        auc = (mean_rank_pos - (n_pos + 1) / 2) / n_neg

        return auc

    return


@app.cell
def _(Bool, Float, beartype, jaxtyped, np):
    @jaxtyped(typechecker=beartype.beartype)
    def fast_pearson(
        activations: Float[np.ndarray, "n d_sae"], labels: Bool[np.ndarray, " n"]
    ) -> Float[np.ndarray, " d_sae"]:
        """
        Pearson correlation between each feature column and binary labels.
        """
        labels_float = labels.astype(np.float64)

        # Center both
        acts_centered = activations - activations.mean(axis=0)
        labels_centered = labels_float - labels_float.mean()

        # Covariance (numerator)
        cov = (acts_centered * labels_centered[:, None]).sum(axis=0)

        # Standard deviations (denominator)
        acts_std = np.sqrt((acts_centered**2).sum(axis=0))
        labels_std = np.sqrt((labels_centered**2).sum())

        return cov / (acts_std * labels_std + 1e-10)

    return (fast_pearson,)


@app.cell
def _(Bool, Float, beartype, jaxtyped, np):
    @jaxtyped(typechecker=beartype.beartype)
    def activation_freq_ratio(
        activations: Float[np.ndarray, "n d_sae"],
        labels: Bool[np.ndarray, " n"],
        threshold: float = 0.1,
    ) -> Float[np.ndarray, " d_sae"]:
        pos_mask = labels
        neg_mask = ~pos_mask

        active = activations > threshold

        freq_given_pos = active[pos_mask].mean(axis=0)  # P(active | label=1)
        freq_given_neg = active[neg_mask].mean(axis=0)  # P(active | label=0)

        # Log odds ratio
        eps = 1e-8
        log_or = np.log((freq_given_pos + eps) / (freq_given_neg + eps))

        return log_or

    return


@app.cell
def _(
    body_parts,
    fast_pearson,
    fishvista_df,
    habitats,
    itertools,
    mo,
    np,
    pl,
    token_acts_csr,
):
    comparisons = [
        {
            "cruisers": ["pelagic-oceanic", "pelagic-neritic", "pelagic"],
            "maneuverers": ["reef-associated"],
        },
        {
            "pelagic": ["pelagic-oceanic", "pelagic-neritic", "pelagic", "epipelagic"],
            "demersal": ["demersal", "bathydemersal", "benthopelagic"],
        },
        {
            "shallow": ["epipelagic", "reef-associated", "pelagic-neritic"],
            "deep": ["mesopelagic", "bathypelagic", "abyssopelagic", "bathydemersal"],
        },
    ]

    def _():
        lookup = {
            key: val
            for val, key in enumerate(
                fishvista_df.get_column("habitat").dtype.categories
            )
        }
        parts = [
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
        print(lookup)

        has_habitat = ~np.isnan(habitats)
        acts = token_acts_csr[has_habitat].toarray()

        n_features = 1024
        scores = np.zeros((n_features, 10, len(comparisons), 2))

        for part, (i, dct) in mo.status.progress_bar(
            list(itertools.product(range(10), enumerate(comparisons)))
        ):
            for j, (name, keys) in enumerate(sorted(dct.items())):
                vals = np.array([lookup[key] for key in keys])
                score = fast_pearson(
                    acts[:, :n_features],
                    (body_parts[has_habitat] == part)
                    & (np.isin(habitats[has_habitat], vals)),
                )

                scores[:, part, i, j] = score

                scores[np.isnan(scores)] = 0.0

        rows = []
        for p, part in enumerate(parts):
            for i, dct in enumerate(comparisons):
                for j, (name, keys) in enumerate(sorted(dct.items())):
                    best = np.abs(scores[:, p, i, j]).argmax().item()
                    score = scores[best, p, i, j]
                    row = {
                        "part": part,
                        "group": name,
                        "latent": best,
                        "score": np.abs(score),
                    }
                    rows.append(row)

        return pl.DataFrame(rows)

    _()
    return


@app.cell
def _():
    return


@app.cell
def _(body_parts, fast_pearson, habitats, itertools, mo, np, token_acts_csr):
    def _():
        has_habitat = ~np.isnan(habitats)
        acts = token_acts_csr[has_habitat].toarray()

        n_features = 1024
        scores = np.zeros((n_features, 10, 11))

        for part, habitat in mo.status.progress_bar(
            list(itertools.product(range(10), range(11)))
        ):
            score = fast_pearson(
                acts[:, :n_features],
                (body_parts[has_habitat] == part) & (habitats[has_habitat] == habitat),
            )

            scores[:, part, habitat] = score

            scores[np.isnan(scores)] = 0.0

        return scores

    scores = _()
    return (scores,)


@app.cell
def _(np, scores):
    latents = set()
    for i in range(10):
        latents.update(np.abs(scores[:, i, :]).argmax(axis=0).tolist())
    print(" ".join(str(i) for i in sorted(latents)))
    return


@app.cell
def _(np, pl, scores):
    def _():
        rows = []
        for p, part in enumerate([
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
        ]):
            for h, habitat in enumerate([
                "reef-associated",
                "pelagic-oceanic",
                "pelagic-neritic",
                "bathypelagic",
                "bathydemersal",
                "benthopelagic",
                "pelagic",
                "epipelagic",
                "mesopelagic",
                "abyssopelagic",
                "demersal",
            ]):
                best = np.abs(scores[:, p, h]).argmax().item()
                score = scores[best, p, h]
                row = {
                    "part": part,
                    "habitat": habitat,
                    "latent": best,
                    "score": np.abs(score),
                }
                rows.append(row)

        return pl.DataFrame(rows)

    _()
    return


@app.cell
def _(habitats, np, plt):
    def _():
        labels = [
            "reef-associated",
            "pelagic-oceanic",
            "pelagic-neritic",
            "bathypelagic",
            "bathydemersal",
            "benthopelagic",
            "pelagic",
            "epipelagic",
            "mesopelagic",
            "abyssopelagic",
            "demersal",
        ]
        fig, ax = plt.subplots(dpi=150, layout="constrained")
        vals, counts = np.unique_counts(habitats[~np.isnan(habitats)])
        ax.bar(vals, counts / 256)  # content tokens per example
        # ax.hist(, bins=10)
        ax.set_xticks(
            np.arange(11),
            [f"{label} ({i})" for i, label in enumerate(labels)],
            rotation=90,
        )
        ax.set_ylabel("Examples")
        # ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        return fig

    _()
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
def _(beartype, pathlib, saev):
    @beartype.beartype
    def get_shards_split_label(shards_dpath: pathlib.Path) -> str | None:
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

    return (get_shards_split_label,)


@app.cell
def _(
    beartype,
    get_baseline_ce,
    get_inference_probe_metric_fpaths,
    get_shards_split_label,
    json,
    mode,
    np,
    pathlib,
    row,
    saev,
    saev_run,
    shards_root,
    wandb_run,
):
    @beartype.beartype
    def get_probe1d_results(run: saev.disk.Run) -> dict[str, float]:
        split_map: dict[str, tuple[pathlib.Path, str, pathlib.Path]] = {}
        for metrics_fpath in get_inference_probe_metric_fpaths(saev_run.run_dir):
            shard_id = metrics_fpath.parent.name
            shards_dpath = shards_root / shard_id

            if not shards_dpath.exists():
                print(f"Skipping {wandb_run.id}: shards dir {shards_dpath} missing.")
                continue

            split_label = get_shards_split_label(shards_dpath)
            if split_label is None:
                print(
                    f"Skipping shards {shard_id}: unknown split (run {wandb_run.id})."
                )
                continue

            if split_label in split_map:
                print(f"Skipping {wandb_run.id}: duplicate {split_label} metrics.")
                split_map = {}
                break

            split_map[split_label] = (metrics_fpath, shard_id, shards_dpath)

        if not split_map:
            print(f"Skipping {wandb_run.id}: no splits.")
            return row

        if "train" not in split_map:
            print(f"Skipping {wandb_run.id}: missing train, got {split_map.keys()}.")
            return row

        if "val" not in split_map:
            print(f"Skipping {wandb_run.id}: missing val, got {split_map.keys()}.")
            return row

        if wandb_run.id == "pdikj9bl":
            print(wandb_run.id, split_map)

        train_probe_metrics_fpath, train_shards, train_shards_dpath = split_map["train"]
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

        row["downstream/frac_w_neg"] = (w < 0).mean().item()
        row["downstream/frac_best_w_neg"] = (
            (w[best_i, np.arange(n_classes)] < 0).mean().item()
        )

        train_base_ce = get_baseline_ce(train_shards_dpath).mean().item()
        val_base_ce = get_baseline_ce(val_shards_dpath).mean().item()

        row["downstream/train/probe_ce"] = train_ce
        row["downstream/train/baseline_ce"] = train_base_ce
        row["downstream/train/probe_r"] = 1 - train_ce / train_base_ce

        row["downstream/val/probe_ce"] = val_ce
        row["downstream/val/baseline_ce"] = val_base_ce
        row["downstream/val/probe_r"] = 1 - val_ce / val_base_ce

        path = saev_run.inference / train_shards / "metrics.json"
        if path.is_file():
            nmse = json.loads(path.read_text())["normalized_mse"]
            row["downstream/train/normalized_mse"] = nmse

        path = saev_run.inference / val_shards / "metrics.json"
        if path.is_file():
            nmse = json.loads(path.read_text())["normalized_mse"]
            row["downstream/val/normalized_mse"] = nmse

        # k = 16
        path = (
            saev_run.inference
            / val_shards
            / f"probe1d_metrics__train-{train_shards}.npz"
        )
        if path.is_file():
            with np.load(path) as fd:
                ap_c = fd["ap"]
                prec_c = fd["precision"]
                recall_c = fd["recall"]
                f1_c = fd["f1"]
                top_labels_dk = fd["top_labels"]

            row["downstream/val/mean_ap"] = ap_c.mean().item()
            row["downstream/val/mean_prec"] = prec_c.mean().item()
            row["downstream/val/mean_recall"] = recall_c.mean().item()
            row["downstream/val/mean_f1"] = f1_c.mean().item()
            for tau in [0.3, 0.5, 0.7]:
                row[f"downstream/val/cov_at_{tau}".replace(".", "_")] = (
                    (ap_c > tau).mean().item()
                )

            for k in [16, 64, 256]:
                _, count = mode(top_labels_dk[best_i, :k], axis=1)
                row[f"downstream/val/mean_purity_at_{k}"] = (count / k).mean().item()

    return


@app.cell
def _(
    beartype,
    cloudpickle,
    cls_fpath,
    cls_results,
    collections,
    get_shards_split_label,
    json,
    pathlib,
    saev,
    shards_root,
):
    @beartype.beartype
    def get_cls_results_fpaths(run: saev.disk.Run) -> dict[str, list[pathlib.Path]]:
        if not run.inference.is_dir():
            return {}

        globbed = list(run.inference.glob("**/cls_*.pkl"))
        if not globbed:
            print(f"Skipping {run.run_id}: no cls_*.pkl files.")
            return {}

        cls_results_fpaths = collections.defaultdict(list)
        for results_fpath in globbed:
            shard_id = results_fpath.parent.name
            shards_dpath = shards_root / shard_id

            if not shards_dpath.exists():
                print(f"Skipping {run.run_id}: shards dir {shards_dpath} missing.")
                continue

            split_label = get_shards_split_label(shards_dpath)
            if split_label is None:
                print(f"Skipping shards {shard_id}: unknown split (run {run.run_id}).")
                continue

            cls_results_fpaths[split_label].append(results_fpath)

        return cls_results_fpaths

    @beartype.beartype
    def get_cls_results(run: saev.disk.Run) -> dict[str, float]:
        row = {}
        for split, results_fpaths in get_cls_results_fpaths(run).items():
            for fpath in results_fpaths:
                try:
                    with open(fpath, "rb") as fd:
                        header_line = fd.readline()
                        header = json.loads(header_line.decode("utf8"))
                        ckpt = cloudpickle.load(fd)
                    k = header["cfg"]["cls"]["n_nonzero"]

                    # {'cfg': {'run': '/fs/ess/PAS2136/samuelstevens/saev/runs/pdikj9bl', 'train_shards': '/fs/scratch/PAS2136/samuelstevens/saev/shards/e65cf404', 'test_shards': '/fs/scratch/PAS2136/samuelstevens/saev/shards/b8a9ff56', 'target_col': 'habitat', 'patch_agg': 'mean', 'cls': {'n_nonzero': 10}, 'debug': False, 'mem_gb': 80, 'slurm_acct': 'PAS2136', 'slurm_partition': 'preemptible-nextgen', 'n_hours': 1.0, 'log_to': '/users/PAS1576/samuelstevens/projects/saev/logs'}, 'test_acc': 0.6633986928104575, 'n_classes': 10}
                except Exception as err:
                    print(f"Failed to load {cls_fpath}: {err}")
                    continue

        return row

        if cls_results:
            row["downstream/val/cls_results"] = cls_results
        # Load classification results from cls_*.pkl files
        # This a bit of an issue because the probe results are in different inference folders. TODO: fix this.
        cls_results = []

    return (get_cls_results,)


@app.cell
def _():
    {
        "cfg": {
            "run": "/fs/ess/PAS2136/samuelstevens/saev/runs/pdikj9bl",
            "train_shards": "/fs/scratch/PAS2136/samuelstevens/saev/shards/e65cf404",
            "test_shards": "/fs/scratch/PAS2136/samuelstevens/saev/shards/b8a9ff56",
            "target_col": "habitat",
            "patch_agg": "mean",
            "cls": {"n_nonzero": 10},
            "debug": False,
            "mem_gb": 80,
            "slurm_acct": "PAS2136",
            "slurm_partition": "preemptible-nextgen",
            "n_hours": 1.0,
            "log_to": "/users/PAS1576/samuelstevens/projects/saev/logs",
        },
        "test_acc": 0.6633986928104575,
        "n_classes": 10,
    }
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


if __name__ == "__main__":
    app.run()
