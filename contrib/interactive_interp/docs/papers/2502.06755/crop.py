import numpy as np
from PIL import Image


def trim_transparent(image_path, output_path):
    # Open the image
    img = Image.open(image_path)

    # Convert to numpy array with alpha channel
    img_array = np.array(img)

    # Get alpha channel
    alpha = (
        img_array[:, :, 3] if img_array.shape[2] == 4 else np.ones(img_array.shape[:2])
    )

    # Find non-transparent pixels
    coords = np.argwhere(alpha > 0)

    # Find bounding box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Crop image
    cropped = img.crop((x0, y0, x1, y1))

    # Save
    cropped.save(output_path)


for filename in ("dna-knockout.png", "dna-basic.png", "dna-identified.png"):
    output_name = f"trimmed_{filename}"
    trim_transparent(filename, output_name)
