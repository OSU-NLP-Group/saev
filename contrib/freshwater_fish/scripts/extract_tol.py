"""Extract freshwater fish images from TreeOfLife-200M into ImageFolder format.

Usage:
    uv run python contrib/freshwater_fish/scripts/extract_tol.py \
        --taxa-file path/to/freshwater_taxa.csv \
        --output-dpath data/freshwater-fish

The taxa file should be a CSV with columns like 'family', 'genus', 'species'.
Any subset of these columns can be used for filtering.
"""

import concurrent.futures
import dataclasses
import io
import logging
import pathlib
import typing as tp

import beartype
import h5py
import polars as pl
import tyro
from PIL import Image

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("extract_tol")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    taxa_file: pathlib.Path | None = None
    """CSV/parquet file with taxa to filter. Should have columns like 'order', 'family', 'genus', 'species' (any subset). If provided, overrides class_filter and order_filter."""
    class_filter: str = ""
    """Taxonomic class to filter by (e.g., 'Actinopterygii'). Note: GBIF has null class for fish - use order_filter instead."""
    order_filter: tuple[str, ...] = ()
    """Taxonomic orders to filter by (e.g., 'Cypriniformes', 'Perciformes'). Works well for GBIF fish data."""
    resolved_taxa_dpath: pathlib.Path = pathlib.Path(
        "/fs/ess/PAS2136/TreeOfLife/annotations_staging/2025-12-20/resolved_taxa"
    )
    """Directory containing resolved_taxa parquet files."""
    lookup_tables_dpath: pathlib.Path = pathlib.Path(
        "/fs/ess/PAS2136/TreeOfLife/lookup_tables_staging/2025-12-19"
    )
    """Directory containing lookup_tables parquet files (uuid -> h5_file)."""
    output_dpath: pathlib.Path = pathlib.Path("data/freshwater-fish")
    """Where to write the ImageFolder output."""
    label_column: tp.Literal["order", "family", "genus", "species"] = "species"
    """Which taxonomic rank to use as the label (folder name)."""
    n_workers: int = 16
    """Number of parallel workers for extraction."""
    sources: tuple[str, ...] = ("gbif", "eol", "fathomnet", "bioscan")
    """Which TreeOfLife sources to include."""
    # Slurm options
    slurm_acct: str = ""
    """Slurm account (e.g., 'PAS2136'). If empty, run locally."""
    slurm_partition: str = "nextgen"
    """Slurm partition."""
    n_hours: float = 4.0
    """Time limit in hours for Slurm job."""
    log_to: pathlib.Path = pathlib.Path("logs/extract_tol")
    """Directory for Slurm logs."""


@beartype.beartype
def load_taxa(fpath: pathlib.Path) -> pl.DataFrame:
    """Load taxa file (CSV or parquet)."""
    if fpath.suffix == ".parquet":
        return pl.read_parquet(fpath)
    else:
        return pl.read_csv(fpath, infer_schema_length=None)


TAXA_COLS = ("uuid", "class", "order", "family", "genus", "species")


@beartype.beartype
def load_resolved_taxa(dpath: pathlib.Path, sources: tuple[str, ...]) -> pl.DataFrame:
    """Load all resolved_taxa parquet files for specified sources."""
    frames = []
    for source in sources:
        source_dpath = dpath / f"source={source}"
        if not source_dpath.exists():
            logger.warning(f"Source directory not found: {source_dpath}")
            continue
        parquet_files = list(source_dpath.glob("*.parquet"))
        logger.info(f"Loading {len(parquet_files)} parquet files from {source}")
        for fpath in parquet_files:
            df = pl.read_parquet(fpath, columns=list(TAXA_COLS))
            frames.append(df)
    if not frames:
        msg = f"No parquet files found in {dpath}"
        raise FileNotFoundError(msg)
    return pl.concat(frames)


@beartype.beartype
def load_lookup_tables(dpath: pathlib.Path, uuids: set[str]) -> pl.DataFrame:
    """Load lookup_tables parquet files, filtering for specific UUIDs."""
    parquet_files = list(dpath.glob("*.parquet"))
    logger.info(
        f"Scanning {len(parquet_files)} lookup table files for {len(uuids)} UUIDs"
    )
    frames = []
    for fpath in parquet_files:
        df = pl.scan_parquet(fpath).filter(pl.col("uuid").is_in(uuids)).collect()
        if len(df) > 0:
            frames.append(df)
            logger.info(f"Found {len(df)} matches in {fpath.name}")
    if not frames:
        return pl.DataFrame({"uuid": [], "h5_file": []})
    return pl.concat(frames)


@beartype.beartype
def filter_by_taxa(resolved: pl.DataFrame, taxa: pl.DataFrame) -> pl.DataFrame:
    """Filter resolved_taxa by matching taxa file columns."""
    # Find which columns are in both dataframes
    taxa_cols = set(taxa.columns)
    match_cols = [c for c in ["family", "genus", "species"] if c in taxa_cols]
    if not match_cols:
        msg = f"Taxa file must have at least one of: family, genus, species. Found: {taxa.columns}"
        raise ValueError(msg)
    logger.info(f"Filtering by columns: {match_cols}")

    # Build filter expression by joining on available columns
    # Get unique taxa values
    taxa_unique = taxa.select(match_cols).unique()
    return resolved.join(taxa_unique, on=match_cols, how="inner")


@beartype.beartype
def extract_h5_file(
    h5_fpath: pathlib.Path,
    tasks: list[tuple[str, pathlib.Path]],
) -> int:
    """Extract all images from a single h5 file. Returns count of successful extractions."""
    n_success = 0
    try:
        with h5py.File(h5_fpath, "r") as f:
            images = f["images"]
            for uuid, output_fpath in tasks:
                try:
                    if uuid not in images:
                        continue
                    img_bytes = images[uuid][:]
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    output_fpath.parent.mkdir(parents=True, exist_ok=True)
                    img.save(output_fpath, "JPEG", quality=95)
                    n_success += 1
                except Exception as e:
                    logger.warning(f"Failed to extract {uuid}: {e}")
    except Exception as e:
        logger.warning(f"Failed to open {h5_fpath}: {e}")
    return n_success


@beartype.beartype
def main(cfg: Config) -> None:
    logger.info(f"Loading resolved taxa from: {cfg.resolved_taxa_dpath}")
    resolved = load_resolved_taxa(cfg.resolved_taxa_dpath, cfg.sources)
    logger.info(f"Loaded {len(resolved)} total resolved taxa entries")

    # Filter by taxa file, class, or order
    if cfg.taxa_file is not None:
        logger.info(f"Loading taxa file: {cfg.taxa_file}")
        taxa = load_taxa(cfg.taxa_file)
        logger.info(f"Loaded {len(taxa)} taxa entries")
        logger.info("Filtering by taxa file...")
        filtered = filter_by_taxa(resolved, taxa)
    elif cfg.class_filter:
        logger.info(f"Filtering by class: {cfg.class_filter}")
        filtered = resolved.filter(pl.col("class") == cfg.class_filter)
    elif cfg.order_filter:
        logger.info(f"Filtering by orders: {cfg.order_filter}")
        filtered = resolved.filter(pl.col("order").is_in(cfg.order_filter))
    else:
        logger.info("No filtering - extracting all images")
        filtered = resolved
    logger.info(f"Found {len(filtered)} matching images")

    if len(filtered) == 0:
        logger.warning("No matching images found. Check your filter settings.")
        return

    uuids = set(filtered["uuid"].to_list())
    logger.info(f"Loading lookup tables from: {cfg.lookup_tables_dpath}")
    lookup = load_lookup_tables(cfg.lookup_tables_dpath, uuids)

    logger.info("Joining with lookup tables...")
    joined = filtered.join(lookup, on="uuid", how="inner")
    logger.info(f"Found {len(joined)} images with h5 paths")

    # Filter out rows with null labels
    joined = joined.filter(pl.col(cfg.label_column).is_not_null())
    logger.info(f"After removing null labels: {len(joined)} images")

    # Group tasks by h5 file for efficient I/O
    h5_tasks: dict[pathlib.Path, list[tuple[str, pathlib.Path]]] = {}
    n_skipped = 0
    for row in joined.select("uuid", cfg.label_column, "h5_file").iter_rows():
        uuid, label, h5_file = row
        label_safe = str(label).replace("/", "_").replace(" ", "_")
        output_fpath = cfg.output_dpath / label_safe / f"{uuid}.jpg"
        if output_fpath.exists():
            n_skipped += 1
            continue
        h5_path = pathlib.Path(h5_file)
        if h5_path not in h5_tasks:
            h5_tasks[h5_path] = []
        h5_tasks[h5_path].append((uuid, output_fpath))

    n_tasks = sum(len(tasks) for tasks in h5_tasks.values())
    logger.info(
        f"Prepared {n_tasks} tasks across {len(h5_tasks)} h5 files (skipped {n_skipped} existing)"
    )

    if not h5_tasks:
        logger.info("No new images to extract.")
        return

    # Process h5 files in parallel
    h5_items = list(h5_tasks.items())
    logger.info(f"Extracting with {cfg.n_workers} workers...")
    n_total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.n_workers) as executor:
        futures = {
            executor.submit(extract_h5_file, h5_path, tasks): h5_path
            for h5_path, tasks in h5_items
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            n_success = future.result()
            n_total += n_success
            if (i + 1) % 50 == 0 or (i + 1) == len(futures):
                logger.info(
                    f"Progress: {i + 1}/{len(futures)} h5 files, {n_total} images"
                )

    logger.info(f"Extraction complete. {n_total} images saved to {cfg.output_dpath}")


@beartype.beartype
def cli() -> int:
    """CLI entrypoint with optional Slurm submission."""
    cfg = tyro.cli(Config)

    if not cfg.slurm_acct:
        logger.info("Running locally (no --slurm-acct provided).")
        main(cfg)
        return 0

    import submitit

    cfg.log_to.mkdir(parents=True, exist_ok=True)
    executor = submitit.SlurmExecutor(folder=cfg.log_to)
    executor.update_parameters(
        time=int(cfg.n_hours * 60),
        partition=cfg.slurm_partition,
        gpus_per_node=0,
        ntasks_per_node=1,
        cpus_per_task=cfg.n_workers,
        stderr_to_stdout=True,
        account=cfg.slurm_acct,
    )
    job = executor.submit(main, cfg)
    logger.info(f"Submitted job {job.job_id} to Slurm. Logs: {cfg.log_to}")
    return 0


if __name__ == "__main__":
    cli()
