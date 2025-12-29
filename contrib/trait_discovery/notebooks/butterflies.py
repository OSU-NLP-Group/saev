import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    import pathlib

    import beartype
    import glasbey
    import numpy as np
    from PIL import Image

    import marimo as mo


    import saev.data.datasets
    return Image, beartype, glasbey, itertools, mo, np, pathlib, saev


@app.cell
def _(glasbey):
    palette = [
        [int(c * 255) for c in rgb]
        for rgb in glasbey.create_palette(palette_size=256, as_hex=False)
    ]
    return (palette,)


@app.cell
def _(Image, beartype, np, palette):
    @beartype.beartype
    def make_seg(seg: Image.Image) -> Image.Image:
        """Create a colored visualization of segmentation patches."""

        seg = seg.resize((256, 256), resample=Image.NEAREST)
        seg = np.asarray(seg)

        img = np.zeros((256, 256, 3), dtype=np.uint8)

        print(np.unique(seg))

        for i in np.arange(seg.max()):
            img[seg == i] = palette[i]

        return Image.fromarray(img)
    return (make_seg,)


@app.cell
def _(make_seg, pathlib, saev):
    ds = saev.data.datasets.get_dataset(
        saev.data.datasets.ImgSegFolder(
            root=pathlib.Path(
                "/fs/scratch/PAS2136/samuelstevens/derived-datasets/butterflies-segfolder"
            ),
            img_label_fname="image_labels.txt",
        ),
        data_transform=lambda img: img.resize((256, 256)),
        mask_transform=make_seg,
    )
    return (ds,)


@app.cell
def _(ds):
    len(ds)
    return


@app.cell
def _(ds, itertools, mo):
    mo.vstack(
        [
            mo.hstack(
                sample.values(), justify="start"
            )
            for sample in itertools.islice(ds, 20)
        ]
    )
    return


if __name__ == "__main__":
    app.run()
