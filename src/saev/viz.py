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
