import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import pathlib

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import torch
    from jaxtyping import Float, jaxtyped

    return Float, beartype, jaxtyped, json, mo, np, os, pathlib, pl, plt, torch


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

    webapp_dir = f"/fs/scratch/PAS2136/samuelstevens/saev/visuals/{ckpt_dropdown.value}"

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
    webapp_dir,
):
    def get_neurons() -> list[dict]:
        rows = []
        for name in mo.status.progress_bar(
            list(os.listdir(f"{webapp_dir}/sort_by_patch/neurons"))
        ):
            if not name.isdigit():
                continue
            try:
                with open(
                    f"{webapp_dir}/sort_by_patch/neurons/{name}/metadata.json"
                ) as fd:
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
        neuron_dir = f"{webapp_dir}/sort_by_patch/neurons/{n}"

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
            images_to_show.append(mo.vstack([mo.image(sae_path), mo.md(label)]))
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
def _(ckpt_dropdown, pathlib):
    root = pathlib.Path(
        f"/fs/scratch/PAS2136/samuelstevens/saev/acts/butterflies/{ckpt_dropdown.value}"
    )
    return (root,)


@app.cell
def _(np, root, torch):
    x = torch.load(root / "img_acts.pt").numpy()

    percentiles = np.quantile(x, 0.95, axis=0)
    return percentiles, x


@app.cell
def _(pl, root):
    obs = pl.read_parquet(root / "img_obs.parquet").with_columns(
        i=pl.int_range(pl.len()).alias("index")
    )

    target_map = {
        target: label
        for target, label in obs.select("target", "label").unique().iter_rows()
    }
    return obs, target_map


@app.cell
def _(obs):
    obs
    return


@app.cell
def _(Float, beartype, jaxtyped, mo, np, obs, percentiles, pl, x):
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

    f1, prec, recall = get_f1(x, obs)
    return f1, prec, recall


@app.cell
def _(obs, pl):
    target_counts = {
        i: count
        for i, count in obs.group_by("target").agg(pl.len().alias("count")).iter_rows()
    }
    return (target_counts,)


@app.cell
def _(obs, pl, target_counts):
    for key, value in sorted(target_counts.items()):
        if value == 2640:
            print(
                key,
                obs.filter(pl.col("target") == key).item(row=0, column="label"),
                value,
            )
    return


@app.cell
def _(obs, pl):
    obs.filter(pl.col("target") == 50)
    return


@app.cell
def _(f1, pl, prec, recall, target_counts, target_map):
    df = pl.DataFrame([
        {
            "species": target_map[i],
            "f1": f1[i, feature],
            "prec": prec[i, feature],
            "recall": recall[i, feature],
            "feature": feature,
            "n": target_counts[i],
        }
        for i, feature in set(
            []
            + list(enumerate(f1.argmax(axis=1)))
            + list(enumerate(prec.argmax(axis=1)))
            + list(enumerate(recall.argmax(axis=1)))
        )
    ]).unique()

    df.filter(
        ~pl.col("species").str.contains(" x ")
        & (
            (pl.col("species").str.contains("lativitta"))
            | (pl.col("species").str.contains("malleti"))
        )
    ).sort(by="f1", descending=True)
    return (df,)


@app.cell
def _(df, pl):
    df.filter(
        ~pl.col("species").str.contains(" x ")
        & (
            (pl.col("species").str.contains("cyrbia"))
            | (pl.col("species").str.contains("cythera"))
        )
    ).sort(by="f1", descending=True)
    return


@app.cell
def _(df, pl):
    df.filter(
        ~pl.col("species").str.contains(" x ")
        & (
            (pl.col("species").str.contains("notabilis"))
            | (pl.col("species").str.contains("plesseni"))
        )
    ).sort(by="f1", descending=True)
    return


@app.cell
def _(df, pl):
    df.filter(
        ~pl.col("species").str.contains(" x ")
        & (
            (pl.col("species").str.contains("hydara"))
            | (pl.col("species").str.contains("ssp. melpomene"))
        )
    ).sort(by="f1", descending=True)
    return


@app.cell
def _(df, pl):
    df.filter(
        ~pl.col("species").str.contains(" x ")
        & (
            (pl.col("species").str.contains("venus"))
            | (pl.col("species").str.contains("ssp. vulcanus"))
        )
    ).sort(by="f1", descending=True)
    return


@app.cell
def _(df, pl):
    df.filter(
        ~pl.col("species").str.contains(" x ")
        & (
            (pl.col("species").str.contains("ssp. phyllis"))
            | (pl.col("species").str.contains("ssp. nanna"))
        )
    ).sort(by="f1", descending=True)
    return


@app.cell
def _(df, pl):
    df.filter(
        ~pl.col("species").str.contains(" x ")
        & (
            (pl.col("species").str.contains("ssp. demophoon"))
            | (pl.col("species").str.contains("ssp. rosina"))
        )
    ).sort(by="f1", descending=True)
    return


@app.cell
def _(df):
    print("--include-latents", " ".join(map(str, df.get_column("latent").to_list())))
    return


@app.cell
def _(Float, beartype, jaxtyped, np, obs, percentiles, pl, x):
    @jaxtyped(typechecker=beartype.beartype)
    def get_entropy(x_ns: Float[np.ndarray, "n_imgs d_sae"], obs: pl.DataFrame):
        """
        Entropy over labels among images where each latent is ON (x > τ_q),
        plus normalized entropy and concentration (1 - normalized entropy).

        Returns:
          H_d:          Float[np.ndarray, "d_sae"]        # entropy per latent
          H_norm_d:     Float[np.ndarray, "d_sae"]        # H / ln(C)
          concentration_d: Float[np.ndarray, "d_sae"]     # 1 - H_norm
          p_cd:         Float[np.ndarray, "n_classes d_sae"]  # label distribution within ON set
          support_d:    Int[np.ndarray,   "d_sae"]        # #(images ON) per latent
        """
        n_imgs, d_sae = x_ns.shape
        n_classes = obs.select("target").unique().height

        labels_n = obs.get_column("target").to_numpy()

        # Thresholds τ_f per latent and ON mask
        on_ns = x_ns > percentiles[None, :]  # (n,d) bool
        support_s = on_ns.sum(axis=0)

        counts_cs = np.zeros((n_classes, d_sae), dtype=np.int64)
        for c_idx in range(n_classes):
            mask = labels_n == c_idx
            if not np.any(mask):
                continue
            counts_cs[c_idx, :] = on_ns[mask, :].sum(axis=0)

        # Convert to probabilities within the ON set: p[c,d] = counts[c,d] / support[d]
        with np.errstate(divide="ignore", invalid="ignore"):
            p_cs = counts_cs / np.clip(support_s[None, :], 1, None)
        p_cs[:, support_s == 0] = 0.0  # undefined ON → set to 0; we’ll mark NaN later

        # Entropy per latent: H_d = -Σ_c p * ln p, treating 0 * ln 0 = 0
        H_terms = np.where(p_cs > 0, p_cs * np.log(p_cs), 0.0)
        H_s = -H_terms.sum(axis=0).astype(np.float32)  # (d,)

        # Normalization by ln(C) and concentration
        if n_classes > 1:
            H_norm_d = H_s / np.log(n_classes)
            concentration_d = 1.0 - H_norm_d
        else:
            H_norm_d = np.zeros_like(H_s)
            concentration_d = np.ones_like(H_s)

        # For latents with no ON images, mark as NaN
        H_s[support_s == 0] = np.nan
        H_norm_d[support_s == 0] = np.nan
        concentration_d[support_s == 0] = np.nan

        return H_s, H_norm_d, concentration_d, p_cs, support_s

    H_s, *_ = get_entropy(x, obs)
    return (H_s,)


@app.cell
def _(root, torch):
    top_img_i = torch.load(root / "top_img_i.pt", map_location="cpu").numpy()
    return (top_img_i,)


@app.cell
def _(H_s, np, pl, top_img_i):
    var = pl.DataFrame([
        {"i": i, "entropy": entropy, "n_imgs": np.unique(top_img_i[i]).size}
        for i, entropy in enumerate(H_s.tolist())
    ])
    return (var,)


@app.cell
def _(pl, var):
    low_entropy_latents = (
        var.filter(~pl.col("entropy").is_nan() & (pl.col("n_imgs") >= 6))
        .sort(by="entropy")
        .head(100)
        .get_column("i")
        .to_list()
    )
    print("--include-latents", " ".join(map(str, low_entropy_latents)))
    var.filter(~pl.col("entropy").is_nan() & (pl.col("n_imgs") >= 5)).sort(by="entropy")
    return


if __name__ == "__main__":
    app.run()
