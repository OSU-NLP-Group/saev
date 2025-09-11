import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import bisect
    import dataclasses
    import json
    import pathlib

    import beartype
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    from jaxtyping import Float, jaxtyped
    return Float, beartype, bisect, jaxtyped, mo, np, pathlib, pl, torch


@app.cell
def _(pathlib):
    root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/acts/butterflies/")
    return (root,)


@app.cell
def _(mo, root):
    def make_ckpt_dropdown():
        try:
            choices = sorted(path.name for path in root.iterdir())

        except FileNotFoundError:
            choices = []

        return mo.ui.dropdown(choices, label="Checkpoint")


    ckpt_dropdown = make_ckpt_dropdown()
    return (ckpt_dropdown,)


@app.cell
def _(ckpt_dropdown):
    ckpt_dropdown
    return


@app.cell
def _(pl):
    def add_target(obs: pl.DataFrame, fields: list[str]) -> pl.DataFrame:
        # If fields can be nulls, give them a bucket so factorization is well-defined
        obs = obs.with_columns([pl.col(field).fill_null("unknown") for field in fields])

        combos = (
            obs.select(fields)
            .unique(maintain_order=True)  # first-seen ordering
            .with_columns(pl.arange(0, pl.len(), dtype=pl.Int32).alias("target"))
        )

        obs = obs.join(combos, on=fields, how="inner")

        target2fields = {
            target: tuple(rest)
            for target, *rest in obs.unique(pl.col("target"))
            .select("target", *fields)
            .iter_rows()
        }

        return obs, target2fields
    return (add_target,)


@app.cell
def _(add_target, ckpt_dropdown, pl, root):
    img_obs, target2fields = add_target(
        pl.read_parquet(root / ckpt_dropdown.value / "img_obs.parquet"), ["Taxonomic_Name", "View"]
    )
    sae_var = pl.read_parquet(root / ckpt_dropdown.value / "sae_var.parquet")
    return img_obs, sae_var, target2fields


@app.cell
def _(ckpt_dropdown, mo):
    mo.stop(ckpt_dropdown.value is None, mo.md("You must select a checkpoint."))

    get_i, set_i = mo.state(0)
    return get_i, set_i


@app.cell
def _(bisect, ckpt_dropdown, mo, root):
    features = sorted(
        [
            int(path.name)
            for path in (root / ckpt_dropdown.value / "features").iterdir()
            if path.name.isdigit()
        ]
    )


    def find_i(f: int):
        return bisect.bisect_left(features, f)


    mo.md(f"Found {len(features)} saved features.")
    return features, find_i


@app.cell
def _(features, mo, set_i):
    next_button = mo.ui.button(
        label="Next", on_change=lambda _: set_i(lambda v: (v + 1) % len(features))
    )

    prev_button = mo.ui.button(
        label="Previous", on_change=lambda _: set_i(lambda v: (v - 1) % len(features))
    )
    return next_button, prev_button


@app.cell
def _(features, get_i, mo, set_i):
    feature_picker_slider = mo.ui.slider(
        0, len(features), value=get_i(), on_change=lambda i: set_i(i), full_width=True
    )
    return (feature_picker_slider,)


@app.cell
def _(features, get_i, mo):
    feature_picker_text = mo.ui.text(value=str(features[get_i()]))

    feature_picker_button = mo.ui.run_button(label="Search")
    return feature_picker_button, feature_picker_text


@app.cell
def _(feature_picker_button, feature_picker_text, find_i, mo, set_i):
    mo.stop(not feature_picker_button.value)

    set_i(find_i(int(feature_picker_text.value)))
    return


@app.cell
def _(features, get_i, mo, sae_var):
    def display_info(f: int):
        feature = sae_var.row(f, named=True)
        return mo.md(
            f"Feature {f} ({get_i()}/{len(features)}; {get_i() / len(features) * 100:.1f}%) | Frequency: {10 ** feature['log10_freq'] * 100:.5f}% of inputs | Mean Value: {10 ** feature['log10_value']:.3f}"
        )
    return (display_info,)


@app.cell
def _(
    display_info,
    feature_picker_button,
    feature_picker_slider,
    feature_picker_text,
    features,
    get_i,
    mo,
    next_button,
    prev_button,
):
    mo.md(
        f"""
    {mo.hstack([prev_button, next_button, display_info(features[get_i()])], justify="start")}
    {mo.hstack([feature_picker_slider, feature_picker_text, feature_picker_button], align="center")}
    """
    )
    return


@app.cell
def _(ckpt_dropdown, img_obs, mo, root, sae_var):
    def show_img(feature: int, i: int):
        neuron_dir = root / ckpt_dropdown.value / "features" / str(feature)

        sae_path = neuron_dir / f"{i}_sae.png"
        if not sae_path.exists():
            return mo.md(f"*Missing image {i}*")

        feature = sae_var.row(feature, named=True)
        md = img_obs.row(feature["top_i_im"][i], named=True)

        return mo.vstack(
            [mo.image(sae_path), mo.md(md["Taxonomic_Name"]), mo.md(md["View"])],
            align="center",
        )
    return (show_img,)


@app.cell
def _(mo):
    cols_slider = mo.ui.slider(1, 10, value=5, full_width=False)
    return (cols_slider,)


@app.cell
def _(cols_slider, mo):
    mo.hstack([mo.md(f"{cols_slider.value} column(s):"), cols_slider], justify="start")
    return


@app.cell
def _(cols_slider, features, get_i, mo, show_img):
    feature = features[get_i()]
    n_cols = cols_slider.value
    n_images = 25  # Always show 25 images

    # Create rows of images based on the number of columns
    rows = []
    for row_start in range(0, n_images, n_cols):
        row_end = min(row_start + n_cols, n_images)
        row_images = [show_img(feature, i) for i in range(row_start, row_end)]
        if row_images:  # Only add non-empty rows
            rows.append(mo.hstack(row_images, widths="equal"))

    mo.vstack(rows)
    return


@app.cell
def _(ckpt_dropdown, np, root, torch):
    x = torch.load(root / ckpt_dropdown.value / "img_acts.pt").numpy()

    percentiles = np.quantile(x, 0.95, axis=0)
    return percentiles, x


@app.cell
def _(Float, beartype, img_obs, jaxtyped, mo, np, percentiles, pl, x):
    @jaxtyped(typechecker=beartype.beartype)
    def get_f1(x_ns: Float[np.ndarray, "n_imgs d_sae"], obs: pl.DataFrame):
        n_imgs, d_sae = x_ns.shape
        n_classes = obs.select("target").unique().height

        preds_ns = (x_ns > percentiles[None, :]).astype(np.int32)
        n_pos_s = (preds_ns == 1).sum(axis=0)

        labels_n = obs.get_column("target").to_numpy()

        prec_cs = np.ones((n_classes, d_sae), dtype=np.float32)
        recall_cs = np.ones((n_classes, d_sae), dtype=np.float32)

        for c_idx in mo.status.progress_bar(range(n_classes)):
            mask = labels_n == c_idx
            assert mask.sum() > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                prec_cs[c_idx, :] = preds_ns[mask].sum(axis=0) / n_pos_s

            recall_cs[c_idx, :] = preds_ns[mask].sum(axis=0) / mask.sum()

        prec_cs[:, n_pos_s == 0] = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = (2 * prec_cs * recall_cs) / (prec_cs + recall_cs)

        f1 = np.nan_to_num(f1, 0.0)
        return f1, prec_cs, recall_cs


    f1, prec, recall = get_f1(x, img_obs)
    return f1, prec, recall


@app.cell
def _(f1, img_obs, pl, prec, recall, target2fields):
    class_counts = dict(img_obs.group_by("target").len().iter_rows())

    df = (
        pl.DataFrame(
            [
                {
                    "species": target2fields[i][0],
                    "view": target2fields[i][1],
                    "f1": f1[i, feature],
                    "prec": prec[i, feature],
                    "recall": recall[i, feature],
                    "feature": feature,
                    "n_imgs": class_counts[i],
                    "target": i,
                }
                for i, feature in set(
                    []
                    + list(enumerate(f1.argmax(axis=1)))
                    + list(enumerate(prec.argmax(axis=1)))
                    + list(enumerate(recall.argmax(axis=1)))
                )
            ]
        )
        .unique()
        .sort(by="f1", descending=True)
        .filter(pl.col("n_imgs") >= 5)
    )
    return class_counts, df


@app.cell
def _(class_counts, img_obs, pl):
    for key, value in sorted(class_counts.items()):
        if value == 2640:
            print(
                key,
                img_obs.filter(pl.col("target") == key).item(
                    row=0, column="Taxonomic_Name"
                ),
                value,
            )
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    pairs = [
        ("lativitta", "malleti"),
        ("cyrbia", "cythera"),
        ("notabilis", "plesseni"),
        ("hydara", "melpomene"),
        ("venus", "vulcanus"),
        ("demophoon", "rosina"),
        ("phyllis", "nanna"),
        ("erato", "thelxiopeia"),
        ("phyllis", "amandus"),
        ("erato", "thelxiopeia"),
        ("amalfreda", "meriana"),
        ("dignus", "bellula"),
    ]
    return (pairs,)


@app.cell
def _(df, mo, pairs, pl):
    mo.vstack(
        [
            df.filter(
                ~pl.col("species").str.contains(" x ")
                & (
                    (pl.col("species").str.contains(f"ssp. {a}"))
                    | (pl.col("species").str.contains(f"ssp. {b}"))
                )
            ).sort(by="f1", descending=True)
            for a, b in pairs
        ]
    )
    return


@app.cell
def _(df, pairs, pl):
    for a, b in pairs:
        for ssp in [a, b]:
            count_series = (
                df.filter(
                    ~pl.col("species").str.contains(" x ")
                    & (pl.col("species").str.contains(f"ssp. {ssp}"))
                )
                .get_column("n_imgs")
                .unique()
            )
            if count_series.len() == 1:
                count = count_series.item()
            elif count_series.len() == 0:
                count = 0
            else:
                raise ValueError(f"Found {count_series.len()} rows for '{ssp}'.")
            print(f"| {ssp} | {count}", end=" ")
        print()
    return


@app.cell
def _(df):
    df.select("species", "n").unique().sort(by="species")
    return


@app.cell
def _(df, mo):
    latents_str = " ".join(map(str, df.get_column("feature").to_list()))

    mo.md(f"`--include-latents {latents_str}`")
    return


if __name__ == "__main__":
    app.run()
