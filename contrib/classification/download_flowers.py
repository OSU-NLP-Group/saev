# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "tqdm",
#     "tyro",
#     "scipy",
# ]
# ///
"""
A script to download the Flowers102 dataset.

```sh
uv run contrib/classification/download_flowers.py --help
```
"""

import dataclasses
import os
import os.path
import shutil
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import scipy.io
import tqdm
import tyro

images_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
splits_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"


@dataclasses.dataclass(frozen=True)
class Args:
    """Configure download options."""

    dir: str = "."
    """Where to save data."""

    chunk_size_kb: int = 1
    """How many KB to download at a time before writing to file."""


def main(args: Args):
    """Download NeWT."""
    os.makedirs(args.dir, exist_ok=True)
    chunk_size = int(args.chunk_size_kb * 1024)
    labels_mat_path = os.path.join(args.dir, "imagelabels.mat")
    images_tgz_path = os.path.join(args.dir, "102flowers.tgz")
    images_dir_path = os.path.join(args.dir, "jpg")

    # Download labels
    # r = requests.get(labels_url, stream=True)
    # r.raise_for_status()

    # with open(labels_mat_path, "wb") as fd:
    #     for chunk in r.iter_content(chunk_size=chunk_size):
    #         fd.write(chunk)
    # print(f"Downloaded labels: {labels_mat_path}.")

    # # Download images.
    # r = requests.get(images_url, stream=True)
    # r.raise_for_status()

    # n_bytes = int(r.headers["content-length"])

    # with open(images_tgz_path, "wb") as fd:
    #     for chunk in tqdm.tqdm(
    #         r.iter_content(chunk_size=chunk_size),
    #         total=n_bytes / chunk_size,
    #         unit="b",
    #         unit_scale=1,
    #         unit_divisor=1024,
    #         desc="Downloading images",
    #     ):
    #         fd.write(chunk)
    # print(f"Downloaded images: {images_tgz_path}.")

    mat = scipy.io.loadmat(labels_mat_path)
    labels = mat["labels"].reshape(-1).tolist()

    with tarfile.open(images_tgz_path, "r") as tar:
        for member in tqdm.tqdm(tar, desc="Extracting images", total=len(labels)):
            tar.extract(member, path=args.dir, filter="data")
    print(f"Extracted images: {images_dir_path}.")

    # In images_dir_path are files labeled image_00001.jpg, image_00002.jpg, etc. There is one label for each image. The structure for torchvision.datasets.ImageFolder is:
    #
    # root/dog/xxx.png
    # root/dog/xxy.png
    # root/dog/[...]/xxz.png
    #
    # root/cat/123.png
    # root/cat/nsdf3.png
    # root/cat/[...]/asd932_.png
    #
    # We can replicate this structure for the flowers102 dataset by making directories for each label and moving images. This can be done efficiently using the python `threading` module because we are IO bound by `shutil.move()`
    # Create directories for each unique label
    unique_labels = set(labels)
    for label in unique_labels:
        label_dir = os.path.join(args.dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

    def move_image(idx: int):
        """Move a single image to its label directory."""
        img_num = str(idx + 1).zfill(5)
        src = os.path.join(images_dir_path, f"image_{img_num}.jpg")
        dst = os.path.join(args.dir, str(labels[idx]), f"image_{img_num}.jpg")
        shutil.move(src, dst)

    # Move files in parallel using a thread pool
    print("Organizing images into class folders...")
    with ThreadPoolExecutor(max_workers=min(32, len(labels))) as executor:
        list(tqdm.tqdm(
            executor.map(move_image, range(len(labels))),
            total=len(labels),
            desc="Moving images"
        ))
    
    # Clean up empty source directory
    try:
        os.rmdir(images_dir_path)
    except OSError:
        pass

    print(f"Organized {len(labels)} images into {len(unique_labels)} class folders.")


if __name__ == "__main__":
    main(tyro.cli(Args))
