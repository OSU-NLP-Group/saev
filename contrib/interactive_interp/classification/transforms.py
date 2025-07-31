"""
Contains the transforms used in every spot:

* Training
* Making figures
* Web app

For both figures and the webapp, the transform is:

1. Resize the image so that the shortest size is 512 pixels.
2. Take the middle 448x448 as a crop

This gives us object-centric images that are 448x448 (fixed pixel size is important for the web app) and not distorted.
"""

from PIL import Image


def for_training(vit_ckpt: str):
    import saev.activations

    return saev.activations.make_img_transform("clip", vit_ckpt)


def for_figures():
    import einops.layers.torch
    from torchvision.transforms import v2

    return v2.Compose([
        v2.Resize(512, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop((448, 448)),
        v2.ToImage(),
        einops.layers.torch.Rearrange("channels width height -> width height channels"),
    ])


def for_webapp(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w > h:
        resize_w = w * 512 / h
        resize_px = (resize_w, 512)

        margin_x = (resize_w - 448) / 2
        crop_px = (margin_x, 32, 448 + margin_x, 480)
    else:
        resize_h = h * 512 / w
        resize_px = (512, resize_h)
        margin_y = (resize_h - 448) / 2
        crop_px = (32, margin_y, 480, 448 + margin_y)

    return img.resize(resize_px, resample=Image.Resampling.BICUBIC).crop(crop_px)
