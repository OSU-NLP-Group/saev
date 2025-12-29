import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import colorsys
    import fractions
    import math
    import pathlib

    import beartype
    import einops
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import torch
    from PIL import Image
    from torch import nn
    from torchvision.transforms import v2

    import saev.data.transforms

    return (
        Image,
        beartype,
        colorsys,
        einops,
        fractions,
        math,
        mo,
        nn,
        np,
        pathlib,
        pl,
        plt,
        saev,
        torch,
        v2,
    )


@app.cell
def _(pathlib):
    root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/datasets/fish-vista")
    assert root.is_dir()
    return (root,)


@app.cell
def _(pl, root):
    df = pl.read_csv(root / "segmentation_train.csv")
    return (df,)


@app.cell
def _(plt):
    def hist(dist):
        fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
        ax.hist(dist, bins=50)
        return fig

    return


@app.cell
def _(Image, df, mo, resize_to_patch_grid, root):
    def sample_resized():
        content = []

        for i, (img_fname,) in enumerate(
            mo.status.progress_bar(df.select("filename").iter_rows(), total=df.height)
        ):
            if i % 1200 != 1:
                continue

            img_fpath = root / "Images" / img_fname
            img = Image.open(img_fpath)
            w, h = img.size

            content.append(img.resize((w // 4, h // 4)))
            resized = resize_to_patch_grid(img, p=16, n=1920)
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


@app.cell
def _(beartype, einops, nn, torch):
    @beartype.beartype
    class Patchify(nn.Module):
        def __init__(self, patch_size: int, n_patches: int, key: str = "image"):
            super().__init__()
            self.patch_size = patch_size
            self.n_patches = n_patches
            self.key = key

        def forward(self, sample: dict[str, object]) -> dict[str, object]:
            msg = f"{self.key} not in {sorted(sample.keys())}."
            assert self.key in sample, msg
            img = sample[self.key]
            c, h, w = img.shape
            p = self.patch_size
            assert (h % p == 0) and (w % p == 0), f"Got {h}x{w}, patch={p}"

            patches_nd = einops.rearrange(
                img, "c (hp p1) (wp p2) -> (hp wp) (c p1 p2)", p1=p, p2=p
            )
            n, d = patches_nd.shape
            assert n == self.n_patches, f"Expected n={self.n_patches}, got {n}"
            assert d == c * p * p, f"d mismatch: {d} != {c}*{p}*{p}"

            sample[self.key] = patches_nd.contiguous()
            sample["grid"] = torch.tensor([h // p, w // p], dtype=torch.int16)
            return sample

    return (Patchify,)


@app.cell
def _(Image, Patchify, pathlib, saev, torch, v2):
    cfg = saev.data.datasets.ImgSegFolder(
        root=pathlib.Path(
            "/fs/scratch/PAS2136/samuelstevens/derived-datasets/fish-vista-segfolder"
        ),
        img_label_fname="image_labels.txt",
    )

    img_transform = v2.Compose([
        saev.data.transforms.FlexResize(patch_size=16, n_patches=640),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
    ])

    mask_transform = v2.Compose([
        saev.data.transforms.FlexResize(
            patch_size=16, n_patches=640, resample=Image.NEAREST
        ),
        v2.ToImage(),
    ])
    sample_transform = v2.Compose([
        Patchify(patch_size=16, n_patches=640),
        # Patchify(patch_size=16, n_patches=640, key=""),
    ])

    dataset = saev.data.datasets.ImgSegFolderDataset(
        cfg,
        img_transform=img_transform,
        mask_transform=mask_transform,
        sample_transform=sample_transform,
    )
    return (dataset,)


@app.cell
def _(Image, colorsys, np):
    def make_palette(n: int, *, sat=0.75, val=1.0, seed=0) -> list[int]:
        # evenly spaced hues → bright distinct colors
        hues = np.linspace(0, 1, n, endpoint=False)
        rgb = []
        for h in hues:
            r, g, b = colorsys.hsv_to_rgb(h, sat, val)
            rgb += [int(255 * r), int(255 * g), int(255 * b)]
        # pad to 256 entries for PNG 'P' mode
        rgb += [0, 0, 0] * (256 - n)
        return rgb

    def colorize_with_palette(
        label_hw, n_classes: int, ignore_index: int | None = None
    ) -> Image.Image:
        lab = np.asarray(label_hw).astype(np.uint8)
        pal = make_palette(n_classes)
        if ignore_index is not None and 0 <= ignore_index < 256:
            # optional: make ignore transparent-ish (e.g., black)
            start = 3 * ignore_index
            pal[start : start + 3] = [0, 0, 0]
        im = Image.fromarray(lab, mode="P")
        im.putpalette(pal)
        return im

    return (colorize_with_palette,)


@app.cell
def _(torch):
    def patch_label_ignore_bg_bincount(
        pixel_labels_nd: torch.Tensor,  # [N, P*P], int
        *,
        background_idx: int = 0,
        num_classes: int | None = None,
    ) -> torch.Tensor:  # [N]
        x = pixel_labels_nd.to(torch.long)
        print(x.shape)
        N, _ = x.shape
        if num_classes is None:
            num_classes = int(x.max().item()) + 1

        # counts[i, c] = number of times class c appears in patch i
        offsets = torch.arange(N, device=x.device).unsqueeze(1) * num_classes
        flat = (x + offsets).reshape(-1)
        counts = torch.bincount(flat, minlength=N * num_classes).reshape(N, num_classes)

        nonbg = counts.clone()
        nonbg[:, background_idx] = 0
        has_nonbg = nonbg.sum(dim=1) > 0
        nonbg_arg = nonbg.argmax(dim=1)
        bg = torch.full_like(nonbg_arg, background_idx)
        return torch.where(has_nonbg, nonbg_arg, bg)

    return (patch_label_ignore_bg_bincount,)


@app.cell
def _(
    colorize_with_palette,
    dataset,
    einops,
    mo,
    patch_label_ignore_bg_bincount,
):
    def show_sample(sample: dict[str, object]):
        pixel_labels_nd = sample["patch_labels"]
        mode_labels_n = pixel_labels_nd.mode(axis=1).values
        patch_labels_n = patch_label_ignore_bg_bincount(
            pixel_labels_nd, background_idx=0, num_classes=10
        )

        p = 16
        hp, wp = sample["grid"].tolist()
        print(hp, wp)
        print(hp * p, wp * p)

        pixel_labels_hw = einops.rearrange(
            pixel_labels_nd,
            "(hp wp) (p1 p2) -> (hp p1) (wp p2)",
            p1=p,
            p2=p,
            hp=hp,
            wp=wp,
        )

        patch_labels_hw = einops.rearrange(
            patch_labels_n, "(hp wp) -> hp wp", hp=hp, wp=wp
        )
        patch_labels_hw = einops.repeat(
            patch_labels_hw, "h w -> (h p1) (w p2)", p1=p, p2=p
        )

        mode_labels_hw = einops.rearrange(
            mode_labels_n, "(hp wp) -> hp wp", hp=hp, wp=wp
        )
        mode_labels_hw = einops.repeat(
            mode_labels_hw, "h w -> (h p1) (w p2)", p1=p, p2=p
        )

        return mo.vstack([
            colorize_with_palette(pixel_labels_hw.numpy(), 10),
            colorize_with_palette(mode_labels_hw.numpy(), 10),
            colorize_with_palette(patch_labels_hw.numpy(), 10),
        ])

    show_sample(dataset[100])
    return


@app.cell
def _(dataset):
    dataset[100]
    return


@app.cell
def _(dataset):
    dataset.sample_transform
    return


if __name__ == "__main__":
    app.run()
