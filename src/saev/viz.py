import colorsys

import beartype
import matplotlib
import numpy as np
import torch
from jaxtyping import Float, Int, jaxtyped
from PIL import Image, ImageDraw

colormap = matplotlib.colormaps.get_cmap("plasma")


@jaxtyped(typechecker=beartype.beartype)
def add_highlights(
    img: Image.Image,
    patches: Float[np.ndarray, " n_patches"],
    *,
    patch_size: int,
    upper: float | None = None,
    opacity: float = 0.9,
) -> Image.Image:
    if not len(patches):
        return img
    iw_px, ih_px = img.size
    assert ih_px % patch_size == 0
    assert iw_px % patch_size == 0
    ih_np, iw_np = ih_px // patch_size, iw_px // patch_size
    assert iw_np * ih_np == len(patches)

    # Create a transparent overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    colors = (colormap(patches / (upper + 1e-9))[:, :3] * 255).astype(np.uint8)

    for p, (val, color) in enumerate(zip(patches, colors)):
        assert upper is not None
        val /= upper + 1e-9
        x_np, y_np = p % iw_np, p // iw_np
        draw.rectangle(
            [
                (x_np * patch_size, y_np * patch_size),
                (x_np * patch_size + patch_size, y_np * patch_size + patch_size),
            ],
            fill=(*color, int(opacity * val * 255)),
        )

    # Composite the original image and the overlay
    return Image.alpha_composite(img.convert("RGBA"), overlay)


def make_palette(n: int, *, sat: float = 0.75, val: float = 1.0) -> list[int]:
    """Create a color palette with n distinct colors using HSV color space."""
    # Evenly spaced hues → bright distinct colors
    hues = np.linspace(0, 1, n, endpoint=False)
    rgb = []
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        rgb += [int(255 * r), int(255 * g), int(255 * b)]
    # Pad to 256 entries for PNG 'P' mode
    rgb += [0, 0, 0] * (256 - n)
    return rgb


@jaxtyped(typechecker=beartype.beartype)
def colorize_segmentation_patches(
    patch_labels: Int[torch.Tensor, " n_patches"],
    *,
    patch_size: int,
    img_width: int,
    img_height: int,
    n_classes: int | None = None,
    background_idx: int = 0,
) -> Image.Image:
    """
    Create a colored visualization of patch-level segmentation labels.

    Args:
        patch_labels: Patch-level segmentation labels
        patch_size: Size of each patch in pixels
        img_width: Width of the output image in pixels
        img_height: Height of the output image in pixels
        n_classes: Number of segmentation classes (auto-detected if None)
        background_idx: Index of the background class (for special coloring)

    Returns:
        PIL Image with colored patches representing segmentation classes
    """
    # Calculate patch grid dimensions
    w_patches = img_width // patch_size
    h_patches = img_height // patch_size

    # Convert to numpy and reshape to 2D grid
    labels_np = patch_labels.cpu().numpy().reshape(h_patches, w_patches)

    # Repeat each patch to fill the pixel space
    labels_full = np.repeat(
        np.repeat(labels_np, patch_size, axis=0), patch_size, axis=1
    )

    # Determine number of classes if not provided
    if n_classes is None:
        n_classes = int(labels_np.max()) + 1

    # Create palette
    palette = make_palette(n_classes)

    # Make background black or very dark
    if 0 <= background_idx < n_classes:
        start = 3 * background_idx
        palette[start : start + 3] = [0, 0, 0]

    # Create indexed image
    img = Image.fromarray(labels_full.astype(np.uint8), mode="P")
    img.putpalette(palette)

    # Convert to RGBA for consistency with other visualizations
    return img.convert("RGBA")
