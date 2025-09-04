import math

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
    *,
    upper: float | None = None,
    opacity: float = 0.9,
) -> Image.Image:
    if not len(patches):
        return img

    iw_np, ih_np = int(math.sqrt(len(patches))), int(math.sqrt(len(patches)))
    iw_px, ih_px = img.size
    pw_px, ph_px = iw_px // iw_np, ih_px // ih_np
    assert iw_np * ih_np == len(patches)

    # Create a transparent overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    colors = (colormap(patches / (upper + 1e-9))[:, :3] * 256).astype(np.uint8)

    for p, (val, color) in enumerate(zip(patches, colors)):
        assert upper is not None
        val /= upper + 1e-9
        x_np, y_np = p % iw_np, p // ih_np
        draw.rectangle(
            [
                (x_np * pw_px, y_np * ph_px),
                (x_np * pw_px + pw_px, y_np * ph_px + ph_px),
            ],
            fill=(*color, int(opacity * val * 256)),
        )

    # Composite the original image and the overlay
    return Image.alpha_composite(img.convert("RGBA"), overlay)
