"""
Feature-table abstractions for Cambridge mimic-pair analysis.
"""

import beartype
import polars as pl


@beartype.beartype
def get_empty_raw_features_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "run_id": pl.String,
            "feature_id": pl.Int64,
            "weight": pl.Float64,
            "favors": pl.String,
            "abs_weight": pl.Float64,
            "source_C": pl.Float64,
            "source_balanced_acc": pl.Float64,
            "source_ckpt_fpath": pl.String,
        }
    )


@beartype.beartype
def get_empty_pooled_features_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "run_id": pl.String,
            "feature_id": pl.Int64,
            "weight": pl.Float64,
            "favors": pl.String,
            "abs_weight": pl.Float64,
            "source_C": pl.Float64,
            "source_balanced_acc": pl.Float64,
            "source_ckpt_fpath": pl.String,
            "n_ckpts_using_feature": pl.Int64,
            "n_pos": pl.Int64,
            "n_neg": pl.Int64,
            "sign_flips": pl.Boolean,
            "weight_min": pl.Float64,
            "weight_max": pl.Float64,
        }
    )


@beartype.beartype
def make_feature_tables(
    selected_ckpt_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not selected_ckpt_df.height:
        return get_empty_raw_features_df(), get_empty_pooled_features_df()

    required_cols = {
        "run_id",
        "C",
        "balanced_acc",
        "ckpt_fpath",
        "features",
        "weights",
    }
    missing = required_cols - set(selected_ckpt_df.columns)
    msg = f"Missing columns in selected checkpoint table: {sorted(missing)}."
    assert not missing, msg

    bad_rows = selected_ckpt_df.filter(
        pl.col("features").list.len() != pl.col("weights").list.len()
    )
    msg = "Found checkpoint rows with mismatched features/weights lengths."
    assert bad_rows.height == 0, msg

    raw_features_df = (
        selected_ckpt_df
        .select(
            "run_id",
            pl.col("C").alias("source_C"),
            pl.col("balanced_acc").alias("source_balanced_acc"),
            pl.col("ckpt_fpath").alias("source_ckpt_fpath"),
            "features",
            "weights",
        )
        .explode("features", "weights")
        .rename({"features": "feature_id", "weights": "weight"})
        .with_columns(
            pl.col("feature_id").cast(pl.Int64),
            pl.col("weight").cast(pl.Float64),
            pl
            .when(pl.col("weight") > 0)
            .then(pl.lit("melpomene"))
            .otherwise(pl.lit("erato"))
            .alias("favors"),
            pl.col("weight").abs().alias("abs_weight"),
        )
        .select(
            "run_id",
            "feature_id",
            "weight",
            "favors",
            "abs_weight",
            "source_C",
            "source_balanced_acc",
            "source_ckpt_fpath",
        )
        .sort(
            "source_balanced_acc",
            "abs_weight",
            "run_id",
            "feature_id",
            descending=[True, True, False, False],
        )
    )

    canonical_df = (
        raw_features_df
        .sort(
            "source_balanced_acc",
            "abs_weight",
            "source_C",
            descending=[True, True, False],
        )
        .group_by("run_id", "feature_id")
        .first()
    )

    agg_df = raw_features_df.group_by("run_id", "feature_id").agg(
        pl.len().cast(pl.Int64).alias("n_ckpts_using_feature"),
        (pl.col("weight") > 0).sum().cast(pl.Int64).alias("n_pos"),
        (pl.col("weight") < 0).sum().cast(pl.Int64).alias("n_neg"),
        pl.col("weight").min().alias("weight_min"),
        pl.col("weight").max().alias("weight_max"),
    )

    pooled_features_df = (
        canonical_df
        .join(agg_df, on=["run_id", "feature_id"], how="inner")
        .with_columns(
            ((pl.col("n_pos") > 0) & (pl.col("n_neg") > 0)).alias("sign_flips")
        )
        .select(
            "run_id",
            "feature_id",
            "weight",
            "favors",
            "abs_weight",
            "source_C",
            "source_balanced_acc",
            "source_ckpt_fpath",
            "n_ckpts_using_feature",
            "n_pos",
            "n_neg",
            "sign_flips",
            "weight_min",
            "weight_max",
        )
        .sort("abs_weight", "run_id", "feature_id", descending=[True, False, False])
    )
    return raw_features_df, pooled_features_df
