import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import random

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import torch
    import tqdm

    return json, mo, np, os, plt, torch, tqdm


@app.cell
def _(mo, os):
    def make_ckpt_dropdown():
        try:
            choices = sorted(
                os.listdir("/users/PZS1151/igwilson99/local/src/saev/checkpoints/")
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

    webapp_dir = f"/users/PZS1151/igwilson99/local/src/saev/checkpoints/"

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
            f"Neuron {neuron} ({get_i()}/{len(neurons)}; {get_i() / len(neurons) * 100:.1f}%) | Frequency: {10**log10_freq * 100:.3f}% of inputs | Mean Value: {10**log10_value:.3f}"
        )

    return (display_info,)


@app.cell
def _(mo, webapp_dir):
    def show_img(n: int, i: int):
        label = "No label found."
        try:
            label = open(f"{webapp_dir}/neurons/{n}/{i}.txt").read().strip()
            label = " ".join(label.split("_"))
        except FileNotFoundError:
            return mo.md(f"*Missing image {i + 1}*")

        return mo.vstack([mo.image(f"{webapp_dir}/neurons/{n}/{i}.png"), mo.md(label)])

    return (show_img,)


@app.cell
def _(get_i, mo, neurons, show_img):
    n = neurons[get_i()]["neuron"]

    mo.vstack([
        mo.hstack(
            [
                show_img(n, 0),
                show_img(n, 1),
                show_img(n, 2),
                show_img(n, 3),
                show_img(n, 4),
            ],
            widths="equal",
        ),
        mo.hstack(
            [
                show_img(n, 5),
                show_img(n, 6),
                show_img(n, 7),
                show_img(n, 8),
                show_img(n, 9),
            ],
            widths="equal",
        ),
        mo.hstack(
            [
                show_img(n, 10),
                show_img(n, 11),
                show_img(n, 12),
                show_img(n, 13),
                show_img(n, 14),
            ],
            widths="equal",
        ),
        mo.hstack(
            [
                show_img(n, 15),
                show_img(n, 16),
                show_img(n, 17),
                show_img(n, 18),
                show_img(n, 19),
            ],
            widths="equal",
        ),
        mo.hstack(
            [
                show_img(n, 20),
                show_img(n, 21),
                show_img(n, 22),
                show_img(n, 23),
                show_img(n, 24),
            ],
            widths="equal",
        ),
    ])
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
