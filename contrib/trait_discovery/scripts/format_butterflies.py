# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beartype",
#     "tyro",
#     "datasets",
#     "polars",
#     "pillow",
#     "tqdm",
# ]
# ///
"""
Converts Heliconius butterfly images from ImageFolder format to SegFolder format by combining them with segmentation masks from HuggingFace.

This script:
1. Downloads segmentation masks from https://huggingface.co/datasets/samuelstevens/cambridge-butterflies-sst (or uses cached version)
2. Reads the Heliconius_img_master.csv to map between image files and metadata
3. Copies butterfly images from species-organized folders to a flat training directory
4. Extracts segmentation masks from the parquet data and saves them as PNG files
5. Generates an image_labels.txt file mapping image stems to species names

Input structure:
- Images: /fs/scratch/PAS2136/samuelstevens/datasets/butterflies-imgfolder/images/{species}/{X}_{CAMID}_{view}.JPG
- CSV: /fs/scratch/PAS2136/samuelstevens/datasets/butterflies-imgfolder/Heliconius_img_master.csv
- Masks: Downloaded from HuggingFace to /fs/scratch/PAS2136/samuelstevens/datasets/cambridge-butterflies-sst

Output structure (SegFolder format):
- /fs/scratch/PAS2136/samuelstevens/datasets/butterflies-segfolder/
  - images/training/{CAMID}_{view}.JPG
  - annotations/training/{CAMID}_{view}.png
  - image_labels.txt

The script uses ThreadPoolExecutor for parallel processing of image copying and mask writing operations.
All images are placed in a single "training" split with no train/val/test division.
"""

import concurrent.futures
import dataclasses
import io
import logging
import pathlib
import shutil

import beartype
import polars as pl
import tqdm
import tyro
from PIL import Image

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("format_butterflies")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    imgfolder_path: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/datasets/butterflies-imgfolder"
    )
    """Path to butterfly images in ImageFolder format."""
    hf_dataset_path: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/datasets/cambridge-butterflies-sst"
    )
    """Path to download HuggingFace parquet dataset."""
    segfolder_path: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/datasets/butterflies-segfolder"
    )
    """Output path for SegFolder format."""
    hf_dataset_name: str = "samuelstevens/cambridge-butterflies-sst"
    """HuggingFace dataset name."""
    csv_fname: str = "Heliconius_img_master.csv"
    """CSV filename with image metadata."""
    img_label_fname: str = "image_labels.txt"
    """Output label filename."""
    n_threads: int = 16
    """Number of concurrent threads for copying files."""
    job_size: int = 256
    """Number of images to process per job."""


@beartype.beartype
def download_parquet(cfg: Config) -> pl.DataFrame:
    """Download the HuggingFace dataset and return as a polars DataFrame."""
    import datasets

    logger.info("Downloading dataset from HuggingFace: %s", cfg.hf_dataset_name)

    # Download dataset
    dataset = datasets.load_dataset(cfg.hf_dataset_name, split="train")

    # Convert to pandas then polars for easier processing
    df = pl.from_pandas(dataset.to_pandas())

    # Save locally for future use
    cfg.hf_dataset_path.mkdir(parents=True, exist_ok=True)
    parquet_path = cfg.hf_dataset_path / "data.parquet"
    df.write_parquet(parquet_path)
    logger.info("Saved parquet to %s", parquet_path)

    return df


@beartype.beartype
def copy_images_batch(
    cfg: Config, csv_df: pl.DataFrame, parquet_df: pl.DataFrame, start: int, end: int
):
    """Copy a batch of images from imgfolder to segfolder."""
    for row in csv_df.slice(start, end - start).iter_rows(named=True):
        camid = row["CAMID"]
        image_name = row["Image_name"]
        x_id = row["X"]

        # Find corresponding row in parquet
        parquet_rows = parquet_df.filter(pl.col("CAMID") == camid)
        if parquet_rows.is_empty():
            logger.warning("No parquet data for CAMID %s", camid)
            continue

        # Construct source path - the actual filename is X_Image_name
        actual_fname = f"{x_id}_{image_name}"
        found = False

        for species_dir in (cfg.imgfolder_path / "images").iterdir():
            if not species_dir.is_dir():
                continue
            src_path = species_dir / actual_fname
            if src_path.exists():
                found = True
                break

        if not found:
            logger.warning("Image not found: %s", actual_fname)
            continue

        # Copy to segfolder using original image_name (without X prefix)
        dst_path = cfg.segfolder_path / "images" / "training" / image_name
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)


@beartype.beartype
def write_masks_batch(cfg: Config, parquet_df: pl.DataFrame, start: int, end: int):
    """Write a batch of segmentation masks from parquet to disk."""
    for row in parquet_df.slice(start, end - start).iter_rows(named=True):
        image_name = row["Image_name"]
        mask_data = row["mask"]

        if mask_data is None:
            logger.warning("No mask for image %s", image_name)
            continue

        # Extract mask bytes - the mask is stored as a dict with 'bytes' key
        if isinstance(mask_data, dict) and "bytes" in mask_data:
            mask_bytes = mask_data["bytes"]
        else:
            logger.warning("Unexpected mask format for %s", image_name)
            continue

        # Convert JPG filename to PNG for mask
        mask_fname = pathlib.Path(image_name).stem + ".png"
        mask_path = cfg.segfolder_path / "annotations" / "training" / mask_fname

        if mask_path.exists():
            continue

        # Write mask image
        try:
            img = Image.open(io.BytesIO(mask_bytes))
            img.save(mask_path)
        except Exception as e:
            logger.error("Failed to write mask for %s: %s", image_name, e)


@beartype.beartype
def write_labels(cfg: Config, csv_df: pl.DataFrame, parquet_df: pl.DataFrame):
    """Write the image_labels.txt file."""
    label_path = cfg.segfolder_path / cfg.img_label_fname

    with open(label_path, "w") as fd:
        for row in csv_df.iter_rows(named=True):
            camid = row["CAMID"]
            image_name = row["Image_name"]

            # Find species from parquet
            parquet_rows = parquet_df.filter(pl.col("CAMID") == camid)
            if parquet_rows.is_empty():
                continue

            parquet_row = parquet_rows.row(0, named=True)
            species = parquet_row["Taxonomic_Name"]

            # Format: filename_stem species_label
            img_stem = pathlib.Path(image_name).stem
            # Replace spaces with underscores in species name
            species_label = species.replace(" ", "_") if species else "unknown"

            fd.write(f"{img_stem} {species_label}\n")

    logger.info("Wrote labels to %s", label_path)


@beartype.beartype
def batched_idx(total: int, batch_size: int):
    """Generate batch indices for parallel processing."""
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end


@beartype.beartype
def main(cfg: Config):
    """Main function to orchestrate the conversion."""
    logger.info("Starting butterfly dataset conversion")

    # Create output directories
    (cfg.segfolder_path / "images" / "training").mkdir(parents=True, exist_ok=True)
    (cfg.segfolder_path / "annotations" / "training").mkdir(parents=True, exist_ok=True)

    # Download or load parquet data
    parquet_path = cfg.hf_dataset_path / "data.parquet"
    if parquet_path.exists():
        logger.info("Loading existing parquet from %s", parquet_path)
        parquet_df = pl.read_parquet(parquet_path)
    else:
        parquet_df = download_parquet(cfg)

    # Load CSV data
    csv_path = cfg.imgfolder_path / cfg.csv_fname
    csv_df = pl.read_csv(csv_path, infer_schema_length=None)
    logger.info("Loaded CSV with %d rows", len(csv_df))

    # Process with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.n_threads) as pool:
        futures = []

        # Submit image copying jobs
        for start, end in batched_idx(len(csv_df), cfg.job_size):
            futures.append(
                pool.submit(copy_images_batch, cfg, csv_df, parquet_df, start, end)
            )

        # Submit mask writing jobs
        for start, end in batched_idx(len(parquet_df), cfg.job_size):
            futures.append(pool.submit(write_masks_batch, cfg, parquet_df, start, end))

        # Submit label writing job
        futures.append(pool.submit(write_labels, cfg, csv_df, parquet_df))

        # Wait for completion with progress bar
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            if exc := future.exception():
                logger.warning("Job failed: %s", exc)

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main(tyro.cli(Config))
