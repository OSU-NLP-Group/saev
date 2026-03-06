import marimo

__generated_with = "0.9.32"
app = marimo.App(width="full")


@app.cell
def _():
    import os.path
    import sys

    if os.path.dirname(sys.path[0]) not in sys.path:
        sys.path.append(os.path.dirname(sys.path[0]))

    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import saev.colors

    return mo, mpl, os, plt, saev, sys


@app.cell
def __(mpl):
    mpl.font_manager.fontManager.addfont(
        "/System/Library/Fonts/Supplemental/GillSans.ttc"
    )

    mpl.rcParams["font.family"] = "Gill Sans"
    return


@app.cell
def _(plt):
    def barchart(probs: list[tuple[str, float, str]], ylim_max: float):
        labels = [label for label, value, color in probs]
        values = [value for label, value, color in probs]
        colors = [color for label, value, color in probs]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values, color=colors, width=0.7)

        # Customize the plot
        ax.set_ylabel("Probability (%)", fontsize=13)
        ax.set_ylim(0, ylim_max)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_axisbelow(True)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Adjust layout
        fig.tight_layout()

        return fig, ax

    return (barchart,)


@app.cell
def _(barchart, saev):
    def bluejay_before():
        probs_before = [
            ("Blue Jay", 49.0, saev.colors.CYAN_RGB01),
            ("Clark\nNutcracker", 15.0, saev.colors.SEA_RGB01),
            ("White-Breasted\nNuthatch", 11.0, saev.colors.CREAM_RGB01),
            ("Florida Jay", 7.0, saev.colors.GOLD_RGB01),
        ]

        fig, _ = barchart(probs_before, ylim_max=55.0)
        return fig

    bluejay_before()
    return (bluejay_before,)


@app.cell
def __(barchart, saev):
    def bluejay_after():
        probs_after = [
            ("Clark\nNutcracker", 32.0, saev.colors.SEA_RGB01),
            ("White-Breasted\nNuthatch", 21.0, saev.colors.CREAM_RGB01),
            ("Great Gray\nShrike", 7.0, saev.colors.RUST_RGB01),
            ("Blue Jay", 4.0, saev.colors.BLUE_RGB01),
        ]

        fig, _ = barchart(probs_after, ylim_max=55.0)
        return fig

    bluejay_after()
    return (bluejay_after,)


@app.cell
def __(barchart, saev):
    def kingbird_before():
        probs_before = [
            ("Tropical\nKingbird", 93.0, saev.colors.GOLD_RGB01),
            ("Gray\nKingbird", 4.0, saev.colors.CREAM_RGB01),
            ("Great Crested\nFlycatcher", 1.0, saev.colors.SEA_RGB01),
            ("Sayornis", 1.0, saev.colors.CYAN_RGB01),
        ]

        fig, _ = barchart(probs_before, ylim_max=100.0)
        return fig

    kingbird_before()
    return (kingbird_before,)


@app.cell
def __(barchart, saev):
    def kingbird_after():
        probs_before = [
            ("Gray\nKingbird", 73.0, saev.colors.CREAM_RGB01),
            ("Tropical\nKingbird", 12.0, saev.colors.GOLD_RGB01),
            ("Western\nWood Peewee", 5.0, saev.colors.BLUE_RGB01),
            ("Sayornis", 2.0, saev.colors.CYAN_RGB01),
        ]

        fig, _ = barchart(probs_before, ylim_max=100.0)
        return fig

    kingbird_after()
    return (kingbird_after,)


@app.cell
def __(barchart, saev):
    def warbler_before():
        probs_before = [
            ("Canada\nWarbler", 59.0, saev.colors.SEA_RGB01),
            ("Magnolia\nWarbler", 17.0, saev.colors.CREAM_RGB01),
            ("Wilson\nWarbler", 8.0, saev.colors.GOLD_RGB01),
            ("Kentucky\nWarbler", 3.0, saev.colors.ORANGE_RGB01),
        ]

        fig, _ = barchart(probs_before, ylim_max=100.0)
        return fig

    warbler_before()
    return (warbler_before,)


@app.cell
def __(barchart, saev):
    def warbler_after():
        probs = [
            ("Wilson\nWarbler", 36.0, saev.colors.GOLD_RGB01),
            ("Canada\nWarbler", 32.0, saev.colors.SEA_RGB01),
            ("Magnolia\nWarbler", 9.0, saev.colors.CREAM_RGB01),
            ("Kentucky\nWarbler", 3.0, saev.colors.ORANGE_RGB01),
        ]

        fig, _ = barchart(probs, ylim_max=100.0)
        return fig

    warbler_after()
    return (warbler_after,)


@app.cell
def __(barchart, saev):
    def finch_before():
        probs = [
            ("Purple\nFinch", 83.0, saev.colors.RED_RGB01),
            ("Pine\nGrosbeak", 4.0, saev.colors.RUST_RGB01),
            ("Summer\nTanager", 2.0, saev.colors.ORANGE_RGB01),
            ("Bay-Breasted\nWarbler", 2.0, saev.colors.GOLD_RGB01),
        ]

        fig, _ = barchart(probs, ylim_max=100.0)
        return fig

    finch_before()
    return (finch_before,)


@app.cell
def __(barchart, saev):
    def finch_after():
        probs = [
            ("Field\nSparrow", 14.0, saev.colors.BLUE_RGB01),
            ("Bay-Breasted\nWarbler", 11.0, saev.colors.GOLD_RGB01),
            ("Tree\nSparrow", 5.0, saev.colors.CYAN_RGB01),
            ("Chipping\nSparrow", 4.0, saev.colors.CREAM_RGB01),
        ]

        fig, _ = barchart(probs, ylim_max=100.0)
        return fig

    finch_after()
    return (finch_after,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
