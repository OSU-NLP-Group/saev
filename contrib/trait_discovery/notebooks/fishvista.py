import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import fractions
    import math
    import pathlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from PIL import Image

    return Image, fractions, math, mo, np, pathlib, pl, plt


@app.cell
def _(pathlib):
    root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/datasets/fish-vista")
    assert root.is_dir()
    return (root,)


@app.cell
def _(pl, root):
    df = pl.read_csv(root / "segmentation_train.csv")
    return (df,)


@app.cell
def _(Image, df, mo, np, root):
    # How big are these images?
    sizes = []

    for (img_fname,) in mo.status.progress_bar(
        df.select("filename").iter_rows(), total=df.height
    ):
        img_fpath = root / "Images" / img_fname
        img = Image.open(img_fpath)
        sizes.append(img.size)

    sizes = np.array(sizes)
    return (sizes,)


@app.cell
def _(plt):
    def hist(dist):
        fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
        ax.hist(dist, bins=50)
        return fig

    return (hist,)


@app.cell
def _(hist, sizes):
    ratios = sizes[:, 0] / sizes[:, 1]
    hist(ratios)
    return (ratios,)


@app.cell
def _(hist, sizes):
    hist(sizes[:, 0])
    return


@app.cell
def _(hist, sizes):
    hist(sizes[:, 1])
    return


@app.cell
def _(np, ratios):
    np.median(ratios)
    return


@app.cell
def _(np, sizes):
    np.median(sizes[:, 1])
    return


@app.cell
def _(np, sizes):
    np.median(sizes[:, 0])
    return


@app.cell
def _():
    # 3984 / 16 for width
    # 1408 / 16 for height
    # 249 x 88 patches
    return


@app.cell
def _():
    return


@app.cell
def _(Image, df, mo, resize_to_patch_grid, root):
    def sample_resized():
        content = []

        for i, (img_fname,) in enumerate(
            mo.status.progress_bar(df.select("filename").iter_rows(), total=df.height)
        ):
            if i % 600 != 1:
                continue

            img_fpath = root / "Images" / img_fname
            img = Image.open(img_fpath)
            w, h = img.size

            content.append(img.resize((w // 4, h // 4)))
            resized = resize_to_patch_grid(img, p=16, n=640)
            content.append(resized)

            n_patches = resized.size[0] // 16, resized.size[1] // 16
            ratio = resized.size[0] / resized.size[1]
            content.append(mo.md(f"`Ratio: {w / h:.3f} -> {ratio:.3f}`"))
            content.append(mo.md(f"`Sizes: {resized.size} px {n_patches} patches`"))

        return mo.vstack(content)

    sample_resized()
    return


@app.cell
def _(Image, math):
    def resize_to_patch_grid(
        img: Image.Image, p: int, n: int, resample=Image.LANCZOS
    ) -> Image.Image:
        """
        Resize image to (w, h) so that:
          - w % p == 0, h % p == 0
          - (h/p) * (w/p) == N
        Uses the math: choose c* = argmin_{c|N} |c - sqrt(A*N)| with A = W0/H0, then r* = N // c*.
        Final size: w = p*c*, h = p*r*.
        """
        if p <= 0 or n <= 0:
            raise ValueError("p and n must be positive integers")

        w0, h0 = img.size
        a0 = w0 / h0

        # Find the divisor of N nearest to target (no lists; just scan factor pairs)
        best_c = 0
        best_dist = float("inf")
        for i in range(1, int(math.sqrt(n) + 1)):
            if n % i != 0:
                continue

            for d in (i, n // i):
                c, r = d, n // d
                aspect = c / r
                dist = abs(aspect - a0)

                if dist < best_dist:
                    best_c = d
                    best_dist = dist

        c = best_c
        r = n // c
        w, h = c * p, r * p
        return img.resize((w, h), resample=resample)

    # Example:
    # img = Image.open("input.jpg")
    # out = resize_to_patch_grid(img, p=16, N=192)
    # out.save("resized.jpg")
    return (resize_to_patch_grid,)


@app.cell
def _(fractions):
    def aspect_ratios(n: int):
        """
        Return sorted aspect ratios (w/h) achievable with r*c = n patches.
        - If landscape_only=True, only ratios >= 1 are returned (unique).
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")

        ratios = set()
        # loop to sqrt(n), add factor pairs once
        r = 1
        while r * r <= n:
            if n % r == 0:
                c = n // r
                ratios.add(fractions.Fraction(c, r))  # landscape
                ratios.add(fractions.Fraction(r, c))  # portrait
            r += 1

        out = sorted(ratios, key=float)
        return [float(x) for x in out]

    def list_ratios():
        for n in range(128, 1280, 128):
            ratios = [n for n in aspect_ratios(n) if n > 0.1 and n <= 8]
            print(n, ratios)

    list_ratios()
    return


if __name__ == "__main__":
    app.run()
