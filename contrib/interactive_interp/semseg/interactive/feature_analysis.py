import marimo

__generated_with = "0.9.20"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""Use this dashboard to reproduce step 2 of the feature suppression experiment (see the README.md for complete details)."""
    )
    return


if __name__ == "__main__":
    app.run()
