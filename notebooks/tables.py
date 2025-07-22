import marimo

__generated_with = "0.9.14"
app = marimo.App(width="full")


@app.cell
def __():
    import json
    import os.path

    import beartype
    import marimo as mo
    import polars as pl
    import wandb

    return beartype, json, mo, os, pl, wandb


@app.cell
def __():
    tag = "baseline-v3.0"
    return (tag,)


@app.cell
def __(df, pl):
    df.sort(
        by=(pl.col("model_key"), pl.col("data_key"), pl.col("summary/losses/loss")),
        descending=False,
    ).select(
        "id",
        "model_key",
        "data_key",
        "config/sae/remove_parallel_grads",
        "config/sae/normalize_w_dec",
        "config/sae/n_reinit_samples",
        "summary/losses/loss",
        pl.col("config/data/shard_root").str.strip_prefix(
            "/local/scratch/stevens.994/cache/saev/"
        ),
    )
    return


@app.cell
def __():
    # sorted(df.columns)
    return


@app.cell
def __(get_data_key, get_model_key, json, os, pl, tag, wandb):
    def make_df(runs):
        rows = []
        for run in runs:
            row = {}
            row["id"] = run.id
            row.update(**{
                f"summary/{key}": value for key, value in run.summary.items()
            })
            row.pop("summary/eval/hist", None)

            # config
            row.update(**{
                f"config/data/{key}": value
                for key, value in run.config.pop("data").items()
            })
            row.update(**{
                f"config/sae/{key}": value
                for key, value in run.config.pop("sae").items()
            })

            row.update(**{f"config/{key}": value for key, value in run.config.items()})

            with open(
                os.path.join(row["config/data/shard_root"], "metadata.json")
            ) as fd:
                metadata = json.load(fd)

            row["model_key"] = get_model_key(metadata)

            data_key = get_data_key(metadata)
            if data_key is None:
                print("Bad run: {run.id}")
                continue
            row["data_key"] = data_key

            row["config/d_vit"] = metadata["d_vit"]
            rows.append(row)

        df = pl.DataFrame(rows).with_columns(
            (pl.col("config/sae/d_vit") * pl.col("config/sae/exp_factor")).alias(
                "config/sae/d_sae"
            )
        )
        return df

    df = make_df(
        wandb.Api().runs(path="samuelstevens/saev", filters={"config.tag": tag})
    )
    return df, make_df


@app.cell
def __(beartype):
    @beartype.beartype
    def get_model_key(metadata: dict[str, object]) -> str | None:
        if (
            metadata["model_org"] == "dinov2"
            and metadata["model_ckpt"] == "dinov2_vitb14_reg"
        ):
            return "DINOv2 ViT-B/14"
        if (
            metadata["model_org"] == "open-clip"
            and metadata["model_ckpt"] == "ViT-B-16/openai"
        ):
            return "CLIP ViT-B/16"

        print(f"Unknown model: {(metadata['model_org'], metadata['model_ckpt'])}")
        return None

    @beartype.beartype
    def get_data_key(metadata: dict[str, object]) -> str | None:
        if "train_mini" in metadata["data"] and "Inat21Dataset" in metadata["data"]:
            return "iNat21"

        if "train" in metadata["data"] and "Imagenet" in metadata["data"]:
            return "ImageNet-1K"

        print(f"Unknown data: {metadata['data']}")
        return None

    return get_data_key, get_model_key


@app.cell
def __(beartype):
    @beartype.beartype
    def to_human(num: float | int) -> str:
        prefix = "-" if num < 0 else ""

        num = abs(num)

        for i, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
            if num > i:
                return f"{prefix}{num / i:.1f}{suffix}"

        if num < 1:
            return f"{prefix}{num:.3g}"

        return f"{prefix}{num}"

    return (to_human,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
