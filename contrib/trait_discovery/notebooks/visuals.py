import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import bisect
    import pathlib

    import beartype
    import marimo as mo
    import numpy as np
    import polars as pl
    from jaxtyping import Float, jaxtyped

    import saev.disk
    return Float, beartype, bisect, jaxtyped, mo, np, pathlib, pl, saev


@app.cell
def _(pathlib):
    root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs/")
    return (root,)


@app.cell
def _(mo, root):
    def make_ckpt_dropdown():
        try:
            choices = sorted(path.name for path in root.iterdir())

        except FileNotFoundError:
            choices = []

        return mo.ui.dropdown(choices, label="Checkpoint", searchable=True)


    ckpt_dropdown = make_ckpt_dropdown()
    return (ckpt_dropdown,)


@app.cell
def _(ckpt_dropdown):
    ckpt_dropdown
    return


@app.cell
def _(ckpt_dropdown, root, saev):
    run = saev.disk.Run(root / ckpt_dropdown.value)
    return (run,)


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
    return


@app.cell
def _(pl, run):
    # img_obs, target2fields = add_target(
    #     pl.read_parquet(root / ckpt_dropdown.value / "obs.parquet"),
    #     ["Taxonomic_Name", "View"],
    # )
    sae_var = pl.read_parquet(run.inference / "781f8739" / "var.parquet")
    return (sae_var,)


@app.cell
def _(ckpt_dropdown, mo):
    mo.stop(ckpt_dropdown.value is None, mo.md("You must select a checkpoint."))

    get_i, set_i = mo.state(0)
    return get_i, set_i


@app.cell
def _(bisect, mo, run):
    features = sorted(
        [
            int(path.name)
            for path in (run.inference / "781f8739" / "images").iterdir()
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
def _(mo):
    cols_slider = mo.ui.slider(1, 10, value=3, full_width=False)
    show_img_switch = mo.ui.switch(value=False, label="Original Image")
    show_sae_img_switch = mo.ui.switch(value=True, label="Highlighted Image")
    show_seg_switch = mo.ui.switch(value=False, label="Original Segmentation")
    show_sae_seg_switch = mo.ui.switch(value=True, label="Highlighted Segmentation")
    return (
        cols_slider,
        show_img_switch,
        show_sae_img_switch,
        show_sae_seg_switch,
        show_seg_switch,
    )


@app.cell
def _(
    cols_slider,
    mo,
    show_img_switch,
    show_sae_img_switch,
    show_sae_seg_switch,
    show_seg_switch,
):
    mo.hstack(
        [
            mo.hstack(
                [mo.md(f"{cols_slider.value} column(s):"), cols_slider], justify="start"
            ),
            show_img_switch,
            show_sae_img_switch,
            show_seg_switch,
            show_sae_seg_switch,
        ],
        justify="space-between",
    )
    return


@app.cell
def _(
    mo,
    run,
    sae_var,
    show_img_switch,
    show_sae_img_switch,
    show_sae_seg_switch,
    show_seg_switch,
):
    def show_img(feature: int, i: int):
        neuron_dir = run.inference / "781f8739" / "images" / str(feature)

        imgs = []

        n_imgs = sum(
            [
                show_img_switch.value,
                show_sae_img_switch.value,
                show_seg_switch.value,
                show_sae_seg_switch.value,
            ]
        )
        width = 100 / n_imgs

        if show_img_switch.value:
            path = neuron_dir / f"{i}_img.png"
            imgs.append(
                mo.image(path, width=f"{width}%")
                if path.exists()
                else mo.md(f"*Missing image {i}*")
            )

        if show_sae_img_switch.value:
            path = neuron_dir / f"{i}_sae_img.png"
            imgs.append(
                mo.image(path, width=f"{width}%")
                if path.exists()
                else mo.md(f"*Missing SAE image {i}*")
            )

        if show_seg_switch.value:
            path = neuron_dir / f"{i}_seg.png"
            imgs.append(
                mo.image(path, width=f"{width}%")
                if path.exists()
                else mo.md(f"*Missing segmentation {i}*")
            )

        if show_sae_seg_switch.value:
            path = neuron_dir / f"{i}_sae_seg.png"
            imgs.append(
                mo.image(path, width=f"{width}%")
                if path.exists()
                else mo.md(f"*Missing SAE segmentation {i}*")
            )

        feature = sae_var.row(feature, named=True)
        # metadata = img_obs.row(feature["top_img_i"][i], named=True)

        return mo.vstack(
            [
                mo.hstack(imgs),
                # mo.md(metadata["Taxonomic_Name"]),
                # mo.md(metadata["Image_name"]),
            ],
            align="center",
        )
    return (show_img,)


@app.cell
def _(cols_slider, features, get_i, mo, show_img):
    n_cols = cols_slider.value
    n_images = 8

    # Create rows of images based on the number of columns
    rows = []
    for row_start in range(0, n_images, n_cols):
        row_end = min(row_start + n_cols, n_images)
        row_images = [show_img(features[get_i()], i) for i in range(row_start, row_end)]
        if row_images:  # Only add non-empty rows
            rows.append(mo.hstack(row_images, widths="equal"))

    mo.vstack(rows)
    return


@app.cell
def _():
    # x = torch.load(root / ckpt_dropdown.value / "img_acts.pt").numpy()

    # percentiles = np.quantile(x, 0.95, axis=0)
    return


@app.cell
def _(Float, beartype, img_obs, jaxtyped, mo, np, percentiles, pl, x):
    @jaxtyped(typechecker=beartype.beartype)
    def get_f1(x_ns: Float[np.ndarray, "n_imgs d_sae"], obs: pl.DataFrame):
        obs = obs.filter(pl.col("hybrid_stat") == "non-hybrid")
        x_ns = x_ns[obs.get_column("index").to_numpy()]
        n_imgs, d_sae = x_ns.shape
        n_classes = obs.select("target").unique().height

        preds_ns = (x_ns > percentiles[None, :]).astype(np.int32)
        n_pos_s = (preds_ns == 1).sum(axis=0)

        labels_n = obs.get_column("target").to_numpy()

        prec_cs = np.ones((n_classes, d_sae), dtype=np.float32)
        recall_cs = np.ones((n_classes, d_sae), dtype=np.float32)

        for i, c_idx in enumerate(mo.status.progress_bar(sorted(np.unique(labels_n)))):
            mask = labels_n == c_idx
            assert mask.sum() > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                prec_cs[i, :] = preds_ns[mask].sum(axis=0) / n_pos_s

            recall_cs[i, :] = preds_ns[mask].sum(axis=0) / mask.sum()

        prec_cs[:, n_pos_s == 0] = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = (2 * prec_cs * recall_cs) / (prec_cs + recall_cs)

        f1 = np.nan_to_num(f1, 0.0)
        return f1, prec_cs, recall_cs


    f1, prec, recall = get_f1(x, img_obs)
    return f1, prec, recall


@app.cell
def _(img_obs, pl):
    img_obs.filter(pl.col("hybrid_stat") == "non-hybrid").select("target").unique().max()
    return


@app.cell
def _(f1, img_obs, np, pl, prec, recall, sae_var, target2fields):
    class_counts = dict(img_obs.group_by("target").len().iter_rows())

    i2c = sorted(
        np.unique(
            img_obs.filter(pl.col("hybrid_stat") == "non-hybrid")
            .get_column("target")
            .to_numpy()
        ).tolist()
    )
    print(i2c)

    df = (
        pl.DataFrame(
            [
                {
                    "species": target2fields[i2c[i]][0],
                    "view": target2fields[i2c[i]][1],
                    "f1": f1[i, feature],
                    "prec": prec[i, feature],
                    "recall": recall[i, feature],
                    "feature": feature,
                    "n_imgs": class_counts[i2c[i]],
                    "target": i2c[i],
                    "log10_freq": sae_var.row(feature, named=True)["log10_freq"],
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
