import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import bisect
    import pathlib

    import marimo as mo
    import polars as pl

    import saev.disk

    return bisect, mo, pathlib, pl, saev


@app.cell
def _(pathlib):
    root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs/")
    return (root,)


@app.cell
def _(mo, root):
    def has_clips(run_dir):
        inference_dir = run_dir / "inference"

        if not inference_dir.is_dir():
            return False

        for shards in inference_dir.iterdir():
            if (inference_dir / shards / "clips").is_dir():
                return True

        return False

    def make_ckpt_dropdown():
        try:
            choices = sorted(
                path.name
                for path in mo.status.progress_bar(list(root.iterdir()))
                if has_clips(path)
            )

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
def _(mo, run):
    def _():
        options = [
            dir.name for dir in run.inference.iterdir() if (dir / "clips").is_dir()
        ]
        default = options[0] if options else None
        return mo.ui.dropdown(options, value=default, label="Shards", searchable=True)

    shards_dropdown = _()
    return (shards_dropdown,)


@app.cell
def _(shards_dropdown):
    shards_dropdown
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

    return


@app.cell
def _(pl, run, shards_dropdown):
    sae_var = pl.read_parquet(run.inference / shards_dropdown.value / "var.parquet")
    return (sae_var,)


@app.cell
def _(ckpt_dropdown, mo):
    mo.stop(ckpt_dropdown.value is None, mo.md("You must select a checkpoint."))

    get_i, set_i = mo.state(0)
    return get_i, set_i


@app.cell
def _(bisect, mo, run, shards_dropdown):
    features = sorted([
        int(path.name)
        for path in (run.inference / shards_dropdown.value / "clips").iterdir()
        if path.name.isdigit()
    ])

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
        0,
        len(features) - 1,
        value=get_i(),
        on_change=lambda i: set_i(i),
        full_width=True,
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
        i = get_i()
        return mo.md(
            f"Feature {f} ({i + 1}/{len(features)}; {(i + 1) / len(features) * 100:.1f}%) | Frequency: {10 ** feature['log10_freq'] * 100:.5f}% of inputs | Mean Value: {10 ** feature['log10_value']:.3f}"
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
    show_spectrogram_switch = mo.ui.switch(value=False, label="Original Spectrogram")
    show_sae_spectrogram_switch = mo.ui.switch(
        value=True, label="Highlighted Spectrogram"
    )
    show_time_clip_switch = mo.ui.switch(value=False, label="Time-Clipped Audio")
    show_time_freq_clip_switch = mo.ui.switch(
        value=True, label="Time-Freq-Clipped Audio"
    )
    return (
        cols_slider,
        show_sae_spectrogram_switch,
        show_spectrogram_switch,
        show_time_clip_switch,
        show_time_freq_clip_switch,
    )


@app.cell
def _(
    cols_slider,
    mo,
    show_sae_spectrogram_switch,
    show_spectrogram_switch,
    show_time_clip_switch,
    show_time_freq_clip_switch,
):
    mo.hstack(
        [
            mo.hstack(
                [mo.md(f"{cols_slider.value} column(s):"), cols_slider], justify="start"
            ),
            show_spectrogram_switch,
            show_sae_spectrogram_switch,
            show_time_clip_switch,
            show_time_freq_clip_switch,
        ],
        justify="space-between",
    )
    return


@app.cell
def _(
    mo,
    run,
    sae_var,
    shards_dropdown,
    show_sae_spectrogram_switch,
    show_spectrogram_switch,
    show_time_clip_switch,
    show_time_freq_clip_switch,
):
    def show_img(feature: int, i: int):
        neuron_dir = run.inference / shards_dropdown.value / "clips" / str(feature)

        imgs = []
        audios = []

        n_imgs = sum([
            show_spectrogram_switch.value,
            show_sae_spectrogram_switch.value,
        ])
        width = 100 / n_imgs

        if show_spectrogram_switch.value:
            path = neuron_dir / f"{i}_spectrogram.png"
            print(path)
            imgs.append(
                mo.image(path, width=f"{width}%")
                if path.exists()
                else mo.md(f"*Missing image {i}*")
            )

        if show_sae_spectrogram_switch.value:
            path = neuron_dir / f"{i}_sae_spectrogram.png"
            imgs.append(
                mo.image(path, width=f"{width}%")
                if path.exists()
                else mo.md(f"*Missing SAE image {i}*")
            )

        if show_time_clip_switch.value:
            path = neuron_dir / f"{i}_time_clip.ogg"
            audios.append(
                mo.audio(path)
                if path.exists()
                else mo.md(f"*Missing time-clipped audio {i}*")
            )

        if show_time_freq_clip_switch.value:
            path = neuron_dir / f"{i}_time_freq_clip.ogg"
            audios.append(
                mo.audio(path)
                if path.exists()
                else mo.md(f"*Missing time-freq-clipped audio {i}*")
            )

        feature = sae_var.row(feature, named=True)
        # metadata = img_obs.row(feature["top_img_i"][i], named=True)

        return mo.vstack(
            [
                mo.hstack([*imgs, mo.vstack(audios)]),
                # mo.md(metadata["Taxonomic_Name"]),
                # mo.md(metadata["Image_name"]),
            ],
            align="center",
        )

    return (show_img,)


@app.cell
def _(cols_slider, features, get_i, mo, show_img):
    n_cols = cols_slider.value
    n_images = 4

    # Create rows of images based on the number of columns
    rows = []
    for row_start in range(0, n_images, n_cols):
        row_end = min(row_start + n_cols, n_images)
        row_images = [show_img(features[get_i()], i) for i in range(row_start, row_end)]
        if row_images:  # Only add non-empty rows
            rows.append(mo.hstack(row_images, widths="equal"))

    mo.vstack(rows)
    return


if __name__ == "__main__":
    app.run()
