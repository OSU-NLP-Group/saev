import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import saev.data
    import saev.disk

    return beartype, mo, np, pathlib, plt, saev


@app.cell
def _(pathlib, saev):
    run = saev.disk.Run(
        pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs/8atno5xa/")
    )
    shards_root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")
    return run, shards_root


@app.cell
def _(beartype, np, pathlib, saev, shards_root):
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

    baseline_ce = get_baseline_ce(shards_root / "e967c008")
    baseline_ce.mean().item()
    return (baseline_ce,)


@app.cell
def _(np, run):
    with np.load(run.inference / "e967c008" / "probe1d/probe1d_metrics.npz") as fd:
        loss = fd["loss"]
        biases = fd["biases"]
        weights = fd["weights"]
        tp = fd["tp"]
        tn = fd["tn"]
        fp = fd["fp"]
        fn = fd["fn"]
    return biases, fn, fp, loss, tn, tp, weights


@app.cell
def _():
    return


@app.cell
def _(fn, fp, mo, tn, tp):
    acc = (tp + tn) / (tp + tn + fp + fn)

    mo.md(f"Accuracy: {acc.max(axis=0).mean().item() * 100:.3f}%")
    return


@app.cell
def _(baseline_ce, loss, np, plt):
    def _():
        fig, ax = plt.subplots(dpi=200, layout="constrained")
        ax.hist(
            1 - loss.min(axis=0) / baseline_ce,
            bins=np.linspace(0, 1, 51),
            label=f"8atno5xa ({(1 - loss.min(axis=0) / baseline_ce).mean().item():.3f})",
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Variance Explained")
        ax.set_ylabel("Count of 151 Classes")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        return fig

    _()
    return


@app.cell
def _(plt, weights):
    def _():
        fig, ax = plt.subplots(dpi=200, layout="constrained")
        ax.hist(weights.ravel(), bins=100)
        ax.set_yscale("log")

        ax.spines[["top", "right"]].set_visible(False)
        return fig

    _()
    return


@app.cell
def _(biases, np):
    np.histogram(biases.ravel(), bins=20)
    return


@app.cell
def _(biases, plt):
    def _():
        fig, ax = plt.subplots(dpi=200, layout="constrained")
        ax.hist(biases.ravel(), bins=100)
        ax.set_yscale("log")

        ax.spines[["top", "right"]].set_visible(False)
        return fig

    _()
    return


if __name__ == "__main__":
    app.run()
