import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import tqdm

    return json, mo, np, os, plt, torch, tqdm


@app.cell
def _(mo, os):
    def make_ckpt_dropdown():
        try:
            choices = sorted(
                os.listdir("/fs/scratch/PAS2136/samuelstevens/saev/visuals/")
            )

        except FileNotFoundError:
            choices = []

        return mo.ui.dropdown(choices, label="Checkpoint:")

    ckpt_dropdown = make_ckpt_dropdown()
    return (ckpt_dropdown,)


@app.cell
def _(ckpt_dropdown, mo):
    mo.hstack([ckpt_dropdown], justify="start")
    return


@app.cell
def _(ckpt_dropdown, mo):
    mo.stop(
        ckpt_dropdown.value is None,
        mo.md(
            "Run `uv run main.py webapp --help` to fill out at least one checkpoint."
        ),
    )

    webapp_dir = f"/fs/scratch/PAS2136/samuelstevens/saev/visuals/{ckpt_dropdown.value}/sort_by_patch"

    get_i, set_i = mo.state(0)
    return get_i, set_i, webapp_dir


@app.cell
def _(mo):
    sort_by_freq_btn = mo.ui.run_button(label="Sort by frequency")

    sort_by_value_btn = mo.ui.run_button(label="Sort by value")

    sort_by_latent_btn = mo.ui.run_button(label="Sort by latent")
    return sort_by_freq_btn, sort_by_latent_btn, sort_by_value_btn


@app.cell
def _(mo, sort_by_freq_btn, sort_by_latent_btn, sort_by_value_btn):
    mo.hstack(
        [sort_by_freq_btn, sort_by_value_btn, sort_by_latent_btn], justify="start"
    )
    return


@app.cell
def _(
    json,
    mo,
    os,
    sort_by_freq_btn,
    sort_by_latent_btn,
    sort_by_value_btn,
    tqdm,
    webapp_dir,
):
    def get_neurons() -> list[dict]:
        rows = []
        for name in tqdm.tqdm(list(os.listdir(f"{webapp_dir}/neurons"))):
            if not name.isdigit():
                continue
            try:
                with open(f"{webapp_dir}/neurons/{name}/metadata.json") as fd:
                    rows.append(json.load(fd))
            except FileNotFoundError:
                print(f"Missing metadata.json for neuron {name}.")
                continue
            # rows.append({"neuron": int(name)})
        return rows

    neurons = get_neurons()

    if sort_by_latent_btn.value:
        neurons = sorted(neurons, key=lambda dct: dct["neuron"])
    elif sort_by_freq_btn.value:
        neurons = sorted(neurons, key=lambda dct: dct["log10_freq"])
    elif sort_by_value_btn.value:
        neurons = sorted(neurons, key=lambda dct: dct["log10_value"], reverse=True)

    mo.md(f"Found {len(neurons)} saved neurons.")
    return (neurons,)


@app.cell
def _(mo, neurons, set_i):
    next_button = mo.ui.button(
        label="Next",
        on_change=lambda _: set_i(lambda v: (v + 1) % len(neurons)),
    )

    prev_button = mo.ui.button(
        label="Previous",
        on_change=lambda _: set_i(lambda v: (v - 1) % len(neurons)),
    )
    return next_button, prev_button


@app.cell
def _(get_i, mo, neurons, set_i):
    neuron_slider = mo.ui.slider(
        0,
        len(neurons),
        value=get_i(),
        on_change=lambda i: set_i(i),
        full_width=True,
    )
    return (neuron_slider,)


@app.cell
def _():
    return


@app.cell
def _(
    display_info,
    get_i,
    mo,
    neuron_slider,
    neurons,
    next_button,
    prev_button,
):
    # label = f"Neuron {neurons[get_i()]} ({get_i()}/{len(neurons)}; {get_i() / len(neurons) * 100:.2f}%)"
    # , display_info(**neurons[get_i()])
    mo.md(f"""
    {mo.hstack([prev_button, next_button, display_info(**neurons[get_i()])], justify="start")}
    {neuron_slider}
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(get_i, mo, neurons):
    def display_info(log10_freq: float, log10_value: float, neuron: int):
        return mo.md(
            f"Neuron {neuron} ({get_i()}/{len(neurons)}; {get_i() / len(neurons) * 100:.1f}%) | Frequency: {10**log10_freq * 100:.5f}% of inputs | Mean Value: {10**log10_value:.3f}"
        )

    return (display_info,)


@app.cell
def _(json, mo, os, webapp_dir):
    def show_img(n: int, i: int):
        neuron_dir = f"{webapp_dir}/neurons/{n}"

        # Try to load metadata from JSON first
        metadata = {}
        label = "No label found."

        # Check for new JSON format
        if os.path.exists(f"{neuron_dir}/{i}.json"):
            try:
                with open(f"{neuron_dir}/{i}.json") as fd:
                    metadata = json.load(fd)
                    label = metadata.get("label", "No label found.")
                    label = " ".join(label.split("_"))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        # Fall back to old text format
        elif os.path.exists(f"{neuron_dir}/{i}.txt"):
            try:
                label = open(f"{neuron_dir}/{i}.txt").read().strip()
                label = " ".join(label.split("_"))
            except FileNotFoundError:
                pass

        # Check which image files exist
        sae_path = f"{neuron_dir}/{i}_sae.png"
        seg_path = f"{neuron_dir}/{i}_seg.png"
        old_path = f"{neuron_dir}/{i}.png"  # For backwards compatibility

        # Determine which images to show
        images_to_show = []

        # Try new SAE image first, fall back to old format
        if os.path.exists(sae_path):
            images_to_show.append(
                mo.vstack([mo.image(sae_path), mo.md("SAE Activations")])
            )
        elif os.path.exists(old_path):
            images_to_show.append(mo.vstack([mo.image(old_path), mo.md(label)]))
        else:
            return mo.md(f"*Missing image {i}*")

        # Add segmentation if it exists
        if os.path.exists(seg_path):
            images_to_show.append(
                mo.vstack([mo.image(seg_path), mo.md("Segmentation")])
            )

        # If we have both images, show side-by-side; otherwise just show what we have
        if len(images_to_show) == 2:
            return mo.vstack([mo.md(label), mo.hstack(images_to_show, widths="equal")])
        else:
            return images_to_show[0]

    return (show_img,)


@app.cell
def _(mo):
    # Add a slider to control number of columns
    cols_slider = mo.ui.slider(
        1, 10, value=5, label="Number of columns:", full_width=False
    )
    return (cols_slider,)


@app.cell
def _(cols_slider, mo):
    mo.md(f"""{cols_slider}""")
    return


@app.cell
def _(cols_slider, get_i, mo, neurons, show_img):
    n = neurons[get_i()]["neuron"]
    n_cols = cols_slider.value
    n_images = 25  # Always show 25 images

    # Create rows of images based on the number of columns
    rows = []
    for row_start in range(0, n_images, n_cols):
        row_end = min(row_start + n_cols, n_images)
        row_images = [show_img(n, i) for i in range(row_start, row_end)]
        if row_images:  # Only add non-empty rows
            rows.append(mo.hstack(row_images, widths="equal"))

    mo.vstack(rows)
    return


@app.cell
def _(os, torch, webapp_dir):
    sparsity_fpath = os.path.join(webapp_dir, "sparsity.pt")
    sparsity = torch.load(sparsity_fpath, weights_only=True, map_location="cpu")

    values_fpath = os.path.join(webapp_dir, "mean_values.pt")
    values = torch.load(values_fpath, weights_only=True, map_location="cpu")
    return sparsity, values


@app.cell
def _(mo, np, plt, sparsity):
    def plot_hist(counts):
        fig, ax = plt.subplots()
        ax.hist(np.log10(counts.numpy() + 1e-9), bins=100)
        return fig

    mo.md(f"""
    Sparsity Log10

    {mo.as_html(plot_hist(sparsity))}
    """)
    return (plot_hist,)


@app.cell
def _(mo, plot_hist, values):
    mo.md(
        f"""
    Mean Value Log10

    {mo.as_html(plot_hist(values))}
    """
    )
    return


@app.cell
def _(np, plt, sparsity, values):
    def plot_dist(
        min_log_sparsity: float,
        max_log_sparsity: float,
        min_log_value: float,
        max_log_value: float,
    ):
        fig, ax = plt.subplots()

        log_sparsity = np.log10(sparsity.numpy() + 1e-9)
        log_values = np.log10(values.numpy() + 1e-9)

        mask = np.ones(len(log_sparsity)).astype(bool)
        mask[log_sparsity < min_log_sparsity] = False
        mask[log_sparsity > max_log_sparsity] = False
        mask[log_values < min_log_value] = False
        mask[log_values > max_log_value] = False

        n_shown = mask.sum()
        ax.scatter(
            log_sparsity[mask],
            log_values[mask],
            marker=".",
            alpha=0.1,
            color="tab:blue",
            label=f"Shown ({n_shown})",
        )
        n_filtered = (~mask).sum()
        ax.scatter(
            log_sparsity[~mask],
            log_values[~mask],
            marker=".",
            alpha=0.1,
            color="tab:red",
            label=f"Filtered ({n_filtered})",
        )

        ax.axvline(min_log_sparsity, linewidth=0.5, color="tab:red")
        ax.axvline(max_log_sparsity, linewidth=0.5, color="tab:red")
        ax.axhline(min_log_value, linewidth=0.5, color="tab:red")
        ax.axhline(max_log_value, linewidth=0.5, color="tab:red")

        ax.set_xlabel("Feature Frequency (log10)")
        ax.set_ylabel("Mean Activation Value (log10)")
        ax.legend(loc="upper right")

        return fig

    return (plot_dist,)


@app.cell
def _(mo, plot_dist, sparsity_slider, value_slider):
    mo.md(
        f"""
    Log Sparsity Range: {sparsity_slider}
    {sparsity_slider.value}

    Log Value Range: {value_slider}
    {value_slider.value}

    {mo.as_html(plot_dist(sparsity_slider.value[0], sparsity_slider.value[1], value_slider.value[0], value_slider.value[1]))}
    """
    )
    return


@app.cell
def _(mo):
    sparsity_slider = mo.ui.range_slider(start=-8, stop=0, step=0.1, value=[-6, -1])
    return (sparsity_slider,)


@app.cell
def _(mo):
    value_slider = mo.ui.range_slider(start=-3, stop=1, step=0.1, value=[-0.75, 1.0])
    return (value_slider,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
