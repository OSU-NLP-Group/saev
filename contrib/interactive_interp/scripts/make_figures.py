import math
import os

import beartype
import tyro
from PIL import Image, ImageDraw

default_highlighted_i_semseg = [
    202,
    203,
    204,
    205,
    206,
    207,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
]

default_highlighted_i_classification = [63, 64, 77, 78, 79, 93, 94]


@beartype.beartype
def add_highlights(img: Image.Image, patches: list[bool]) -> Image.Image:
    if not len(patches):
        return img

    iw_np, ih_np = int(math.sqrt(len(patches))), int(math.sqrt(len(patches)))
    iw_px, ih_px = img.size
    pw_px, ph_px = iw_px // iw_np, ih_px // ih_np
    assert iw_np * ih_np == len(patches)

    # Create a transparent overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for p, highlight in enumerate(patches):
        if not highlight:
            continue
        x_np, y_np = p % iw_np, p // ih_np

        draw.rectangle(
            [
                (x_np * pw_px, y_np * ph_px),
                (x_np * pw_px + pw_px, y_np * ph_px + ph_px),
            ],
            fill=(225, 29, 72, 128),
        )

    # Composite the original image and the overlay
    return Image.alpha_composite(img.convert("RGBA"), overlay)


@beartype.beartype
def make_figure_overview(
    starfish_in: str = os.path.join(".", "docs", "assets", "overview", "starfish.jpg"),
    highlighted_i: list[int] = [103, 104, 105, 106, 107, 119, 120, 121, 122, 123],
    out: str = os.path.join(".", "docs", "assets", "overview"),
    patchified_i: list[int] = [0, 1, 2, 3, 192, 193, 194, 195],
):
    """
    Make all the assets for the overview figure.
    """

    os.makedirs(out, exist_ok=True)

    import einops.layers.torch
    from torchvision.transforms import v2

    img = Image.open(starfish_in)

    to_array = v2.Compose([
        v2.Resize(512, interpolation=v2.InterpolationMode.NEAREST),
        v2.CenterCrop((448, 448)),
        v2.ToImage(),
        einops.layers.torch.Rearrange("channels width height -> width height channels"),
    ])
    img_arr = to_array(img)
    patch_size = 32
    n_patch_per_side = 14

    for p in patchified_i + highlighted_i:
        row = (p // n_patch_per_side) * patch_size
        col = (p % n_patch_per_side) * patch_size
        patch = img_arr[row : row + patch_size, col : col + patch_size]
        patch_img = Image.fromarray(patch.numpy())

        if p in highlighted_i:
            overlay = Image.new("RGBA", patch_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([(0, 0), patch_img.size], fill=(225, 29, 72, 128))
            patch_img = Image.alpha_composite(patch_img.convert("RGBA"), overlay)

        patch_img.save(os.path.join(out, f"starfish_patch{p}.png"))

    bool_patches = [i in highlighted_i for i in range(256)]
    highlighted_img = add_highlights(Image.fromarray(img_arr.numpy()), bool_patches)
    highlighted_img.save(os.path.join(out, "starfish_highlighted.png"))


@beartype.beartype
def make_figure_semseg(
    example_i: int = 3122,
    patchified_i: list[int] = [0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255],
    highlighted_i: list[int] = default_highlighted_i_semseg,
    out: str = os.path.join(".", "logs", "figures"),
    ade20k: str = "/research/nfs_su_809/workspace/stevens.994/datasets/ade20k/",
    split: str = "training",
):
    """
    Parts of the figure that need to be programmatically generated:

    * Split image into image patches. They should be saved as patch1.png, patch2.png, etc. Only patches 1, 2, 3, and 254, 255 are required. If at all possible, they should be larger than 16x16 (resize to 512x512, crop to 448x448, then patchify to 32x32).

    * Image with specific patches highlighted. Use highlighted_i to choose the patches. Again, should be saved at a 448x448 resolution.

    * Semantic segmentation prediction before modification. Classes should be specific colors rather than random colors.
    """
    os.makedirs(out, exist_ok=True)

    import einops.layers.torch
    import saev.activations
    import saev.config
    from torchvision.transforms import v2

    to_array = v2.Compose([
        v2.Resize((512, 512), interpolation=v2.InterpolationMode.NEAREST),
        v2.CenterCrop((448, 448)),
        v2.ToImage(),
        einops.layers.torch.Rearrange("channels width height -> width height channels"),
    ])

    ade20k_dataset = saev.activations.Ade20k(
        saev.config.Ade20kDataset(root=ade20k, split=split),
        img_transform=to_array,
        seg_transform=to_array,
    )

    img_arr = ade20k_dataset[example_i]["image"]

    # Split into 28x28 pixel image patches and save patches `patchified_i` as ade20k_example{i}_patch{p} to out
    patch_size = 28
    n_patch_per_side = 16
    for p in patchified_i:
        row = (p // n_patch_per_side) * patch_size
        col = (p % n_patch_per_side) * patch_size
        patch = img_arr[row : row + patch_size, col : col + patch_size]
        patch_img = Image.fromarray(patch.numpy())

        # Highlight some patches
        if p in highlighted_i:
            overlay = Image.new("RGBA", patch_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([(0, 0), patch_img.size], fill=(225, 29, 72, 128))
            patch_img = Image.alpha_composite(patch_img.convert("RGBA"), overlay)

        patch_img.save(os.path.join(out, f"ade20k_example{example_i}_patch{p}.png"))

    # Save image with highlighted_i patches highlighted.
    bool_patches = [i in highlighted_i for i in range(256)]
    highlighted_img = add_highlights(Image.fromarray(img_arr.numpy()), bool_patches)
    highlighted_img.save(os.path.join(out, f"ade20k_highlighted_img{example_i}.png"))


@beartype.beartype
def barchart(data: dict[str, float], colors: list[str], ylim_max: int = 100):
    import matplotlib.pyplot as plt

    data = sorted(data.items(), key=lambda pair: pair[1], reverse=True)
    categories = [label for label, value in data]
    values = [value * 100 for label, value in data]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(categories, values, color=colors[: len(values)])

    # Customize the plot
    ax.set_ylabel("Probability (%)", fontsize=13)
    ax.set_ylim(0, ylim_max)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Adjust layout
    fig.tight_layout()

    return fig, ax


@beartype.beartype
def make_figure_classification(
    example_i: int = 680,
    highlighted_i: list[int] = default_highlighted_i_classification,
    patchified_i: list[int] = [0, 1, 2, 3, 190, 191, 192, 193, 194, 195],
    out: str = os.path.join(".", "logs", "figures", "classification"),
    ylim_max: int = -1,
    probs_before: dict[str, float] = {},
    probs_after: dict[str, float] = {},
):
    import contrib.classification.transforms
    import saev.activations
    import saev.config

    transform = contrib.classification.transforms.for_figures()

    dataset = saev.activations.ImageFolder(
        "/research/nfs_su_809/workspace/stevens.994/datasets/cub2011/test",
        transform=transform,
    )
    img_arr = dataset[example_i]["image"]

    patch_size = 32
    n_patch_per_side = 14
    for p in patchified_i + highlighted_i:
        row = (p // n_patch_per_side) * patch_size
        col = (p % n_patch_per_side) * patch_size
        patch = img_arr[row : row + patch_size, col : col + patch_size]
        patch_img = Image.fromarray(patch.numpy())

        # Highlight some patches
        if p in highlighted_i:
            overlay = Image.new("RGBA", patch_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([(0, 0), patch_img.size], fill=(225, 29, 72, 128))
            patch_img = Image.alpha_composite(patch_img.convert("RGBA"), overlay)

        patch_img.save(os.path.join(out, f"cub200_img{example_i}_patch{p}.png"))

    bool_patches = [i in highlighted_i for i in range(196)]
    highlighted_img = add_highlights(Image.fromarray(img_arr.numpy()), bool_patches)
    highlighted_img.save(os.path.join(out, f"cub200_highlighted_img{example_i}.png"))

    if not probs_before:
        return

    probs_before = {
        key.replace("\\n", "\n"): value for key, value in probs_before.items()
    }
    probs_after = {
        key.replace("\\n", "\n"): value for key, value in probs_after.items()
    }
    os.makedirs(out, exist_ok=True)

    bar_colors = [
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
    ]

    # Create color mapping for all unique categories
    all_categories = sorted(set(probs_before.keys()) | set(probs_after.keys()))
    color_map = {
        cat: bar_colors[i % len(bar_colors)] for i, cat in enumerate(all_categories)
    }

    # Map colors to categories in each dict
    colors_before = [
        color_map[cat]
        for cat in sorted(probs_before.keys(), key=probs_before.get, reverse=True)
    ]
    colors_after = [
        color_map[cat]
        for cat in sorted(probs_after.keys(), key=probs_after.get, reverse=True)
    ]

    if ylim_max < 0:
        # Find max probability across both dicts and convert to percentage
        max_prob = (
            max(max(probs_before.values()) * 100, max(probs_after.values()) * 100) + 5
        )
        # Round up to next multiple of 10
        ylim_max = math.ceil(max_prob / 10) * 10

    fig, ax = barchart(probs_before, colors_before, ylim_max)
    fig.savefig(os.path.join(out, "probs_before.png"))
    fig, ax = barchart(probs_after, colors_after, ylim_max)
    fig.savefig(os.path.join(out, "probs_after.png"))


@beartype.beartype
def make_colorbar_legend(
    colormap: str = "plasma", out: str = os.path.join(".", "logs", "figures")
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    cmap = mpl.colormaps.get_cmap(colormap)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=(1, 8), dpi=300)
    fig.colorbar(sm, cax=ax)
    ax.set_yticks([])
    ax.set_ylabel("Normalized Activation Strength", fontsize=16)
    fig.tight_layout()

    plt.savefig(os.path.join(out, "legend.png"), dpi=300)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "overview": make_figure_overview,
        "semseg": make_figure_semseg,
        "classification": make_figure_classification,
        "legend": make_colorbar_legend,
    })
