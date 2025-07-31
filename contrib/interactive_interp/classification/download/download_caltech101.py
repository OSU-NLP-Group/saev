# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "beartype",
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
A script to download the Caltech101 dataset for use as an saev.activations.ImageFolderDataset.

```sh
uv run contrib/classification/download_flowers.py --help
```
"""

import dataclasses
import os
import random
import shutil
import tarfile
import zipfile

import beartype
import requests
import tqdm
import tyro

url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"


IMG_EXTS = (".jpg", ".jpeg", ".png")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Configure download options."""

    dir: str = "."
    """Where to save data."""

    chunk_size_kb: int = 1
    """How many KB to download at a time before writing to file."""

    seed: int = 42
    """Random seed used to generate split."""


def main(args: Args):
    """Download Caltech 101."""
    os.makedirs(args.dir, exist_ok=True)
    chunk_size = int(args.chunk_size_kb * 1024)
    zip_path = os.path.join(args.dir, "caltech-101.zip")
    tgz_path = os.path.join(args.dir, "caltech-101", "101_ObjectCategories.tar.gz")

    # Download dataset.
    r = requests.get(url, stream=True)
    r.raise_for_status()

    n_bytes = int(r.headers["content-length"])

    with open(zip_path, "wb") as fd:
        for chunk in tqdm.tqdm(
            r.iter_content(chunk_size=chunk_size),
            total=n_bytes / chunk_size,
            unit="b",
            unit_scale=1,
            unit_divisor=1024,
            desc="Downloading dataset",
        ):
            fd.write(chunk)
    print(f"Downloaded dataset: {zip_path}.")

    zip = zipfile.ZipFile(zip_path)
    zip.extract("caltech-101/101_ObjectCategories.tar.gz", args.dir)
    print("Unzipped file.")

    with tarfile.open(tgz_path, "r") as tar:
        for member in tqdm.tqdm(tar, desc="Extracting images"):
            tar.extract(member, path=args.dir, filter="data")
    print("Extracted images.")

    # Clean up and organize files

    # Remove the temporary caltech-101 directory
    dpath = os.path.join(args.dir, "caltech-101")
    shutil.rmtree(dpath)
    print(f"Removed temporary directory: {dpath}")

    # Move 101_ObjectCategories to caltech-101
    os.rename(os.path.join(args.dir, "101_ObjectCategories"), dpath)
    print(f"Moved dataset to: {dpath}")

    # Remove the BACKGROUND_Google folder
    shutil.rmtree(os.path.join(dpath, "BACKGROUND_Google"))
    print("Removed BACKGROUND_Google folder")

    # Create train/test split
    random.seed(args.seed)

    # Create output directories
    train_dir = os.path.join(args.dir, "train")
    test_dir = os.path.join(args.dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Process each class directory
    for class_name in sorted(os.listdir(dpath)):
        class_dpath = os.path.join(dpath, class_name)
        if not os.path.isdir(class_dpath):
            print(f"Skippping {class_dpath} because it is not a directory.")
            continue

        # Get all image files
        image_files = [
            f for f in sorted(os.listdir(class_dpath)) if f.endswith(IMG_EXTS)
        ]
        random.shuffle(image_files)

        # Take first 30 for training
        train_images = image_files[:30]
        # Take up to next 50 for testing
        test_images = image_files[30:80]

        # Create class directories in train and test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Move training images
        for img in train_images:
            src = os.path.join(class_dpath, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)

        # Move test images
        for img in test_images:
            src = os.path.join(class_dpath, img)
            dst = os.path.join(test_dir, class_name, img)
            shutil.copy2(src, dst)

    # Remove the original directory
    shutil.rmtree(dpath)
    print(f"Created train/test split with {len(os.listdir(train_dir))} classes")
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
