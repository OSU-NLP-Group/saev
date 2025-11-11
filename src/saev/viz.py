import pathlib
import re

import beartype
import matplotlib
import numpy as np
from jaxtyping import Float, jaxtyped
from PIL import Image, ImageDraw

colormap = matplotlib.colormaps.get_cmap("plasma")


@jaxtyped(typechecker=beartype.beartype)
def add_highlights(
    img: Image.Image,
    patches: Float[np.ndarray, " n_patches"],
    patch_size: int,
    *,
    upper: float | None = None,
    opacity: float = 0.9,
) -> Image.Image:
    if not len(patches):
        return img
    assert upper is not None
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


HEX_RE = re.compile(r"^#([0-9a-fA-F]{6})$")
RGB_RE = re.compile(r"^rgb\s*\((.+)\)$", re.IGNORECASE)


@beartype.beartype
def parse_color(line: str) -> tuple[float, float, float]:
    msg = f"Invalid color: '{line}'"
    stripped = line.strip()
    assert stripped, msg

    hex_match = HEX_RE.match(stripped)
    if hex_match is not None:
        hex_val = hex_match.group(1)
        color = tuple(int(hex_val[i : i + 2], 16) / 255.0 for i in range(0, 6, 2))
    else:
        rgb_match = RGB_RE.match(stripped)
        assert rgb_match is not None, msg
        channels = tuple(
            float(part.strip())
            for part in rgb_match.group(1).split(",")
            if part.strip()
        )
        assert len(channels) == 3, msg
        max_chan = max(channels)
        min_chan = min(channels)
        assert min_chan >= 0.0, msg
        if max_chan <= 1.0:
            color = channels
        else:
            assert max_chan <= 255.0, msg
            color = tuple(chan / 255.0 for chan in channels)

    msg = f"Invalid color: {color}"
    assert all(0 <= chan <= 1 for chan in color), msg
    return tuple(float(chan) for chan in color)


@beartype.beartype
def load_palette(path: pathlib.Path) -> list[tuple[float, float, float]]:
    """TODO: docstring."""
    import glasbey

    palette = []

    for i, line in enumerate(path.read_text().split("\n")):
        line = line.strip()
        if not line:
            palette.append(None)
            continue

        palette.append(parse_color(line))

    # Extend the palette using https://glasbey.readthedocs.io/en/latest/extending_palettes.html
    n_missing = sum(color is None for color in palette)
    if n_missing:
        seed_palette = [color for color in palette if color is not None]
        if seed_palette:
            extended = glasbey.extend_palette(
                seed_palette, palette_size=len(seed_palette) + n_missing, as_hex=False
            )
            fill_colors = extended[len(seed_palette) :]
        else:
            fill_colors = glasbey.create_palette(palette_size=n_missing, as_hex=False)

        fill_iter = iter(fill_colors)
        for i, color in enumerate(palette):
            if color is not None:
                continue
            next_color = tuple(float(chan) for chan in next(fill_iter))
            palette[i] = next_color

    for i, color in enumerate(palette):
        assert color is not None
        msg = f"Color {i} is invalid: {color}"
        assert all(0 <= chan <= 1 and isinstance(chan, float) for chan in color), msg

    return palette
