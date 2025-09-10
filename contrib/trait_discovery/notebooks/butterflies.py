import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import datasets
    import marimo as mo

    import saev.helpers

    return datasets, mo, saev


@app.cell
def __(datasets, saev):
    ds = datasets.load_dataset(
        "imageomics/Heliconius-Collection_Cambridge-Butterfly",
        "full_master",
        cache_dir=saev.helpers.get_cache_dir(),
    )
    return (ds,)


@app.cell
def __(ds):
    ds
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
