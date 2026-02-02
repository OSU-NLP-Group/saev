# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beartype",
#     "tyro",
#     "datasets",
#     "pillow",
#     "tqdm",
# ]
# ///
"""
Download HuggingFace dataset and convert to ImgSegFolder format.

ImgSegFolder format:
- root/images/{split}/ - RGB images
- root/annotations/{split}/ - segmentation masks (PNG, same stem as image)
- root/labels.csv - CSV with columns: stem,label1,label2,...

Example usage:
    uv run python contrib/trait_discovery/scripts/download_butterflies.py
    uv run python contrib/trait_discovery/scripts/download_butterflies.py --help
"""

import concurrent.futures
import csv
import dataclasses
import io
import logging
import pathlib

import beartype
import tqdm
import tyro
from PIL import Image

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("download_butterflies")

IMAGE_COL_ALIASES = ("image", "img", "photo", "picture")
MASK_COL_ALIASES = ("mask", "segmentation", "seg", "annotation")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    hf_dataset_name: str = "samuelstevens/cambridge-segfolder"
    """HuggingFace dataset name."""
    revision: str = "v1.2"
    """HuggingFace dataset revision (branch, tag, or commit hash)"""
    output_dpath: pathlib.Path = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/derived-datasets/cambridge-segfolder"
    )
    """Output path for ImgSegFolder format."""
    split: str = "train"
    """HuggingFace dataset split to download."""
    target_split: str = "training"
    """Target split name for ImgSegFolder (training or validation)."""
    image_col: str = "image"
    """Column name for images in HF dataset."""
    mask_col: str = "mask"
    """Column name for masks in HF dataset."""
    label_cols: tuple[str, ...] = ("subspecies",)
    """Label column names (become columns in labels.csv after stem)."""
    stem_col: str | None = "stem"
    """Column for stem names. If None, uses index-based naming."""
    n_threads: int = 16
    """Number of concurrent threads."""
    job_size: int = 256
    """Batch size for parallel processing."""


@beartype.beartype
def setup_directories(cfg: Config):
    """Create output directory structure."""
    (cfg.output_dpath / "images" / cfg.target_split).mkdir(parents=True, exist_ok=True)
    (cfg.output_dpath / "annotations" / cfg.target_split).mkdir(
        parents=True, exist_ok=True
    )


@beartype.beartype
def download_dataset(cfg: Config):
    """Download dataset from HuggingFace."""
    import datasets

    logger.info(
        "Downloading dataset from HuggingFace: %s (revision=%s)",
        cfg.hf_dataset_name,
        cfg.revision,
    )
    return datasets.load_dataset(
        cfg.hf_dataset_name, split=cfg.split, revision=cfg.revision
    )


@beartype.beartype
def find_column(dataset_cols: set[str], primary: str, aliases: tuple[str, ...]) -> str:
    """Find a column by name, trying aliases if primary not found."""
    if primary in dataset_cols:
        return primary
    for alias in aliases:
        if alias in dataset_cols:
            logger.info("Using '%s' instead of '%s' for column", alias, primary)
            return alias
    available = ", ".join(sorted(dataset_cols))
    raise ValueError(f"Column '{primary}' not found. Available: {available}")


@beartype.beartype
def extract_pil_image(data) -> Image.Image:
    """Extract PIL Image from various HF dataset formats."""
    # Case 1: Already a PIL Image
    if isinstance(data, Image.Image):
        return data

    # Case 2: Dict with 'bytes' key (common in parquet storage)
    if isinstance(data, dict) and "bytes" in data:
        return Image.open(io.BytesIO(data["bytes"]))

    # Case 3: Dict with 'path' key (file reference)
    if isinstance(data, dict) and "path" in data:
        return Image.open(data["path"])

    # Case 4: Raw bytes
    if isinstance(data, bytes):
        return Image.open(io.BytesIO(data))

    raise ValueError(f"Unknown image format: {type(data)}")


@beartype.beartype
def batched_idx(total: int, batch_size: int):
    """Generate batch indices for parallel processing."""
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end


@beartype.beartype
def write_labels_csv(cfg: Config, dataset) -> dict[int, str]:
    """Write labels.csv and return index-to-stem mapping."""
    labels_fpath = cfg.output_dpath / "labels.csv"

    # Determine actual column names
    dataset_cols = set(dataset.column_names)

    # Build stem mapping and collect labels (skip duplicate stems)
    idx_to_stem: dict[int, str] = {}
    seen_stems: set[str] = set()
    rows: list[list[str]] = []

    for i, row in enumerate(tqdm.tqdm(dataset, desc="Building labels")):
        # Generate stem (either from column or index-based)
        if cfg.stem_col and cfg.stem_col in row:
            stem = pathlib.Path(str(row[cfg.stem_col])).stem
        else:
            stem = f"{i:08d}"

        idx_to_stem[i] = stem

        # Skip duplicate stems in labels.csv (first one wins)
        if stem in seen_stems:
            continue
        seen_stems.add(stem)

        # Collect labels for this stem
        label_values = [stem]
        for col in cfg.label_cols:
            assert col in dataset_cols, f"Label column '{col}' not in dataset."
            label_values.append(str(row[col]))
        rows.append(label_values)

    # Write CSV
    header = ["stem"] + list(cfg.label_cols)
    with open(labels_fpath, "w", newline="") as fd:
        writer = csv.writer(fd)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info(
        "Wrote %d labels to %s (%d duplicates skipped)",
        len(rows),
        labels_fpath,
        len(dataset) - len(rows),
    )
    return idx_to_stem


@beartype.beartype
def write_batch(
    cfg: Config,
    dataset,
    idx_to_stem: dict[int, str],
    image_col: str,
    mask_col: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    """Write a batch of images and masks. Returns (n_success, n_skip)."""
    n_success = 0
    n_skip = 0

    for i in range(start, end):
        row = dataset[i]
        stem = idx_to_stem[i]

        img_fpath = cfg.output_dpath / "images" / cfg.target_split / f"{stem}.jpg"
        mask_fpath = cfg.output_dpath / "annotations" / cfg.target_split / f"{stem}.png"

        # Skip if both exist (resumability)
        if img_fpath.exists() and mask_fpath.exists():
            n_skip += 1
            continue

        try:
            # Extract and save image
            if not img_fpath.exists():
                img = extract_pil_image(row[image_col])
                img = img.convert("RGB")
                img.save(img_fpath)

            # Extract and save mask
            if not mask_fpath.exists():
                mask = extract_pil_image(row[mask_col])
                mask.save(mask_fpath)

            n_success += 1
        except Exception as e:
            logger.warning("Failed to process %s: %s", stem, e)

    return n_success, n_skip


@beartype.beartype
def main(cfg: Config):
    """Main function to download and convert HF dataset to ImgSegFolder format."""
    logger.info("Starting dataset conversion from %s", cfg.hf_dataset_name)
    assert cfg.target_split in ("training", "validation"), (
        f"Invalid target_split: {cfg.target_split}"
    )

    # 1. Create output directory structure
    setup_directories(cfg)

    # 2. Download dataset from HuggingFace
    dataset = download_dataset(cfg)
    logger.info("Downloaded %d examples", len(dataset))

    # 3. Validate columns
    dataset_cols = set(dataset.column_names)
    image_col = find_column(dataset_cols, cfg.image_col, IMAGE_COL_ALIASES)
    mask_col = find_column(dataset_cols, cfg.mask_col, MASK_COL_ALIASES)
    logger.info("Using image_col='%s', mask_col='%s'", image_col, mask_col)

    # 4. Write labels.csv and get stem mapping
    idx_to_stem = write_labels_csv(cfg, dataset)

    # 5. Write images and masks in parallel
    total_success = 0
    total_skip = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.n_threads) as pool:
        futures = []
        for start, end in batched_idx(len(dataset), cfg.job_size):
            futures.append(
                pool.submit(
                    write_batch,
                    cfg,
                    dataset,
                    idx_to_stem,
                    image_col,
                    mask_col,
                    start,
                    end,
                )
            )

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Writing"
        ):
            if exc := future.exception():
                logger.warning("Job failed: %s", exc)
            else:
                n_success, n_skip = future.result()
                total_success += n_success
                total_skip += n_skip

    logger.info(
        "Conversion complete! %d written, %d skipped. Output at %s",
        total_success,
        total_skip,
        cfg.output_dpath,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
