import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import beartype
    import matplotlib.pyplot as plt
    import numpy as np

    import saev.data
    import saev.disk

    return beartype, np, pathlib, plt, saev


@app.cell
def _(pathlib):
    runs_root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
    shards_root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")
    return runs_root, shards_root


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
def _(baseline_ce, np, plt, runs_root, saev):
    def _():
        runs = [
            "8atno5xa",
            "3kfmgk35",
            "pwnr5ewh",
            "po7lsp9a",
        ]

        fig, (ax1, ax2, ax3) = plt.subplots(
            dpi=200, layout="constrained", nrows=3, figsize=(12, 12)
        )

        for run in runs:
            run = saev.disk.Run(runs_root / run)

            probe_metrics = run.inference / "e967c008" / "probe1d_metrics.npz"
            if not probe_metrics.exists():
                continue

            with np.load(run.inference / "e967c008" / "probe1d_metrics.npz") as fd:
                loss = fd["loss"]
                biases = fd["biases"]
                weights = fd["weights"]
                tp = fd["tp"]
                tn = fd["tn"]
                fp = fd["fp"]
                fn = fd["fn"]

            ax1.hist(
                1 - loss.min(axis=0) / baseline_ce,
                bins=np.linspace(0, 1, 101),
                label=f"{run.run_id} ({(1 - loss.min(axis=0) / baseline_ce).mean().item():.3f})",
                alpha=0.3,
            )

            ax2.hist(
                biases.ravel(),
                alpha=0.3,
                label=run.run_id,
                # bins=30,
                bins=np.linspace(-100, 100, 101),
            )
            ax3.hist(
                weights.ravel(),
                alpha=0.3,
                label=run.run_id,
                # bins=30,
                bins=np.linspace(-1500, 1500, 101),
            )

        ax1.set_xlim(0, 1)
        ax1.set_xlabel("Variance Explained")
        ax1.set_ylabel("Count of 151 Classes")
        ax1.legend()
        ax1.spines[["top", "right"]].set_visible(False)

        ax2.set_title("Bias Term Distribution")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.spines[["top", "right"]].set_visible(False)

        ax3.set_title("Weight Term Distribution")
        ax3.set_yscale("log")
        ax3.legend()
        ax3.spines[["top", "right"]].set_visible(False)

        return fig

    _()
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
