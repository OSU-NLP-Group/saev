# contrib/trait_discovery/scripts/format_fishvista.py

import concurrent.futures
import dataclasses
import logging
import pathlib
import shutil
import typing as tp

import beartype
import polars as pl
import tyro

from saev import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("format")

seg_split_mapping = {"train": "training", "val": "validation", "test": "test"}


HABITAT_COLS = (
    "reef-associated",
    "pelagic-oceanic",
    "pelagic-neritic",
    "bathypelagic",
    "bathydemersal",
    "benthopelagic",
    "pelagic",
    "epipelagic",
    "mesopelagic",
    "abyssopelagic",
    "demersal",
)

MIGRATION_COLS = (
    "amphidromous",
    "anadromous",
    "catadromous",
    "limnodromous",
    "non-migratory",
    "oceanodromous",
    "potamodromous",
)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    fv_root: pathlib.Path = pathlib.Path("./data/fish-vista")
    """Where you downloaded FishVista from HuggingFace."""
    dump_to: pathlib.Path = pathlib.Path("./data/segfolder")
    """Where to write the new directories."""
    fishbase_csv: pathlib.Path | None = None
    """Optional FishBase CSV (e.g., fishvista_fishbase.csv) to merge habitat/environment/migration labels."""
    n_threads: int = 16
    """Number of concurrent threads for copying files on disk."""
    job_size: int = 256
    """Number of images to copy per job."""


@beartype.beartype
def _cp_seg(
    cfg: Config,
    valid_stems: set[str],
    fv_split: str,
    tgt_split: str,
    start: int,
    end: int,
):
    seg_df = pl.read_csv(cfg.fv_root / f"segmentation_{fv_split}.csv")

    for (fname,) in seg_df.select("filename").slice(start, end - start).iter_rows():
        stem = pathlib.Path(fname).stem
        if stem not in valid_stems:
            continue

        # Original Image
        src_fpath = cfg.fv_root / "Images" / fname
        if not src_fpath.exists():
            logger.warning("Missing image '%s'", src_fpath)
            continue

        dst_fpath = cfg.dump_to / "images" / tgt_split / fname
        if dst_fpath.exists():
            continue

        shutil.copy2(src_fpath, dst_fpath)

        # Segmentations
        seg_fname = f"{stem}.png"
        src_fpath = cfg.fv_root / "segmentation_masks" / "images" / seg_fname
        if not src_fpath.exists():
            continue

        dst_fpath = cfg.dump_to / "annotations" / tgt_split / seg_fname
        if dst_fpath.exists():
            continue

        shutil.copy2(src_fpath, dst_fpath)


@beartype.beartype
def _get_valid_stems(cfg: Config) -> set[str]:
    """Compute the set of stems that have valid FishBase habitat data."""
    # Load and concat all FishVista segmentation splits
    seg_dfs = []
    for fv_split in seg_split_mapping:
        fpath = cfg.fv_root / f"segmentation_{fv_split}.csv"
        assert fpath.is_file(), f"FishVista segmentation CSV not found: {fpath}"
        seg_dfs.append(pl.read_csv(fpath))

    seg_df = pl.concat(seg_dfs)

    # Validate FishVista columns
    fv_cols = set(seg_df.columns)
    for required in ("filename", "family", "standardized_species"):
        assert required in fv_cols, f"FishVista CSV missing '{required}' column"

    # Extract stem, family, genus, species_epithet
    # FishVista: "Genus species" -> genus="genus", species="species"
    # FishBase has separate genus/species columns with just the epithet
    seg_df = seg_df.with_columns([
        pl.col("filename")
        .map_elements(lambda f: pathlib.Path(f).stem, return_dtype=pl.Utf8)
        .alias("stem"),
        pl.col("standardized_species")
        .str.split(" ")
        .list.first()
        .str.to_lowercase()
        .alias("genus"),
        pl.col("standardized_species")
        .str.split(" ")
        .list.last()
        .str.to_lowercase()
        .alias("species"),
    ]).select(["stem", "family", "genus", "species"])

    # Load and validate FishBase data
    assert cfg.fishbase_csv is not None, "fishbase_csv is required"
    assert cfg.fishbase_csv.is_file(), f"FishBase CSV not found: {cfg.fishbase_csv}"

    fb_df = pl.read_csv(cfg.fishbase_csv, null_values=["?"])
    fb_cols = set(fb_df.columns)

    # Validate required columns
    assert "genus" in fb_cols, "FishBase CSV missing 'genus' column"
    assert "species" in fb_cols, "FishBase CSV missing 'species' column"

    missing_habitat = set(HABITAT_COLS) - fb_cols
    assert not missing_habitat, (
        f"FishBase CSV missing habitat columns: {missing_habitat}"
    )

    missing_migration = set(MIGRATION_COLS) - fb_cols
    assert not missing_migration, (
        f"FishBase CSV missing migration columns: {missing_migration}"
    )

    for env_col in ("marine", "freshwater", "brackish"):
        assert env_col in fb_cols, f"FishBase CSV missing '{env_col}' column"

    fb_df = fb_df.with_columns([
        pl.col("genus").str.to_lowercase(),
        pl.col("species").str.to_lowercase(),
    ])

    # Cast habitat/migration columns to Float64 (some may be inferred as String)
    for col in HABITAT_COLS + MIGRATION_COLS:
        fb_df = fb_df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # Collapse habitat columns into single categorical
    fb_df = fb_df.with_columns(
        pl.coalesce([
            pl.when(pl.col(col) == 1.0).then(pl.lit(col)) for col in HABITAT_COLS
        ]).alias("habitat")
    )

    # Collapse migration columns into single categorical
    fb_df = fb_df.with_columns(
        pl.coalesce([
            pl.when(pl.col(col) == 1.0).then(pl.lit(col)) for col in MIGRATION_COLS
        ]).alias("migration")
    )

    # Add environment columns
    for env_col in ("marine", "freshwater", "brackish"):
        fb_df = fb_df.with_columns(
            pl.when(pl.col(env_col) == 1.0)
            .then(pl.lit("yes"))
            .otherwise(pl.lit("no"))
            .alias(env_col)
        )

    extra_cols = ["habitat", "migration", "marine", "freshwater", "brackish"]
    fb_df = fb_df.select(["genus", "species"] + extra_cols)
    seg_df = seg_df.join(fb_df, on=["genus", "species"], how="left")

    # Filter to only images with habitat data
    n_before = seg_df.height
    seg_df = seg_df.filter(pl.col("habitat").is_not_null())
    n_after = seg_df.height
    n_dropped = n_before - n_after

    match_pct = 100 * n_after / n_before
    logger.info(
        "FishBase join: %d/%d matched (%.1f%%), dropped %d without habitat",
        n_after,
        n_before,
        match_pct,
        n_dropped,
    )
    assert match_pct > 50, f"FishBase join matched only {match_pct:.1f}%, expected >50%"
    assert n_after > 0, "No images left after filtering for habitat data"

    # Write CSV
    import csv

    header = ["stem", "family", "species"] + extra_cols
    with open(cfg.dump_to / "labels.csv", "w", newline="") as fd:
        writer = csv.writer(fd)
        writer.writerow(header)
        for row in seg_df.select(header).iter_rows():
            writer.writerow(row)

    return set(seg_df.get_column("stem").to_list())


@beartype.beartype
def segfolder(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")],
) -> int:
    """
    Convert FishVista to a format useable with SegFolderDataset in src/saev/data/images.py.
    """
    # Create output directories
    for tgt_split in seg_split_mapping.values():
        (cfg.dump_to / "images" / tgt_split).mkdir(exist_ok=True, parents=True)
        (cfg.dump_to / "annotations" / tgt_split).mkdir(exist_ok=True, parents=True)

    # First compute valid stems and write labels.csv (must happen before copying)
    logger.info("Computing valid stems and writing labels.csv...")
    valid_stems = _get_valid_stems(cfg)
    logger.info("Found %d valid stems with habitat data", len(valid_stems))

    # Then copy images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.n_threads) as pool:
        futs = []
        for fv_split, tgt_split in seg_split_mapping.items():
            seg_df = pl.read_csv(cfg.fv_root / f"segmentation_{fv_split}.csv")
            futs.extend([
                pool.submit(_cp_seg, cfg, valid_stems, fv_split, tgt_split, start, end)
                for start, end in helpers.batched_idx(seg_df.height, cfg.job_size)
            ])

        for fut in helpers.progress(
            concurrent.futures.as_completed(futs), total=len(futs), desc="copying"
        ):
            if err := fut.exception():
                logger.warning("Exception: %s", err)

    return 0


@beartype.beartype
def _cp_img(cfg: Config, split: str, start: int, end: int):
    seg_df = pl.read_csv(cfg.fv_root / f"classification_{split}.csv")

    for fname, clsname in (
        seg_df.select("filename", "standardized_species")
        .slice(start, end - start)
        .iter_rows()
    ):
        src_fpath = cfg.fv_root / "Images" / fname
        if not src_fpath.exists():
            logger.warning("Missing image '%s'", src_fpath)
            continue

        dst_fpath = cfg.dump_to / split / clsname / fname
        if dst_fpath.exists():
            continue

        dst_fpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_fpath, dst_fpath)


@beartype.beartype
def imgfolder(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")],
) -> int:
    """
    Convert FishVista to a format useable with ImageFolderDataset in src/saev/data/images.py.
    """

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.n_threads) as pool:
        futs = []

        for split in ["train", "val", "test"]:
            (cfg.dump_to / split).mkdir(exist_ok=True, parents=True)

            img_df = pl.read_csv(cfg.fv_root / f"classification_{split}.csv")

            futs.extend([
                pool.submit(_cp_img, cfg, split, start, end)
                for start, end in helpers.batched_idx(img_df.height, cfg.job_size)
            ])

        for fut in helpers.progress(
            concurrent.futures.as_completed(futs), total=len(futs), desc="copying"
        ):
            if err := fut.exception():
                logger.warning("Exception: %s", err)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            tyro.extras.subcommand_cli_from_dict({
                "imgfolder": imgfolder,
                "segfolder": segfolder,
            })
        )
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
