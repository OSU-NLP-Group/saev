"""Extract freshwater fish images from TreeOfLife-200M into ImageFolder format.

Usage:
    uv run python contrib/freshwater_fish/scripts/extract_tol.py \
        --taxa-file path/to/freshwater_taxa.csv \
        --output-dpath data/freshwater-fish

The taxa file should be a CSV with columns like 'family', 'genus', 'species'.
Any subset of these columns can be used for filtering.

Uses pyarrow instead of polars to avoid segfaults on large GBIF files.
"""

import concurrent.futures
import dataclasses
import io
import logging
import pathlib
import typing as tp
from typing import NamedTuple

import beartype
import h5py
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tyro
from PIL import Image

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("root")


class ImagePair(NamedTuple):
    """A (uuid, label) pair for an image to extract."""

    uuid: str
    label: str


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
    mem_gb: int = 64
    """Memory in GB for Slurm job."""


TAXA_COLS = ("uuid", "class", "order", "family", "genus", "species")


@dataclasses.dataclass(frozen=True, slots=True)
class TaxaFilter:
    """Filter for matching taxa in parquet files."""

    match_cols: list[str]
    """Column names to filter by, e.g. ['family']."""
    taxa_unique: pl.DataFrame
    """DataFrame with unique values to match."""

    @classmethod
    def load(cls, fpath: pathlib.Path) -> tp.Self:
        """Load taxa filter from CSV or parquet file."""
        if fpath.suffix == ".parquet":
            df = pl.read_parquet(fpath)
        else:
            df = pl.read_csv(
                fpath, infer_schema_length=None, truncate_ragged_lines=True
            )
        # Normalize column names to lowercase
        df = df.rename({col: col.lower() for col in df.columns})
        logger.info(f"Loaded {len(df)} taxa entries from {fpath}")

        taxa_cols = set(df.columns)
        match_cols = [c for c in ["family", "genus", "species"] if c in taxa_cols]
        if not match_cols:
            msg = f"Taxa file must have at least one of: family, genus, species. Found: {df.columns}"
            raise ValueError(msg)
        logger.info(f"Will filter by columns: {match_cols}")
        taxa_unique = df.select(match_cols).unique()
        logger.info(f"Found {len(taxa_unique)} unique taxa combinations")
        return cls(match_cols=match_cols, taxa_unique=taxa_unique)


@beartype.beartype
def load_lookup_tables_pyarrow(dpath: pathlib.Path, uuids: set[str]) -> dict[str, str]:
    """Load lookup_tables parquet files, filtering for specific UUIDs. Returns uuid->h5_file dict."""
    parquet_files = sorted(dpath.glob("*.parquet"))
    logger.info(
        f"Loading lookup tables for {len(uuids)} UUIDs from {len(parquet_files)} files"
    )
    uuid_array = pa.array(list(uuids))
    uuid_to_h5: dict[str, str] = {}
    for fpath in parquet_files:
        table = pq.read_table(fpath, columns=["uuid", "h5_file"])
        mask = pc.is_in(table["uuid"], value_set=uuid_array)
        filtered = table.filter(mask)
        n_matches = filtered.num_rows
        if n_matches > 0:
            for uuid, h5_file in zip(
                filtered["uuid"].to_pylist(), filtered["h5_file"].to_pylist()
            ):
                uuid_to_h5[uuid] = h5_file
            logger.info(f"  {fpath.name}: {n_matches} matches")
    logger.info(f"Found {len(uuid_to_h5)} UUIDs in lookup tables")
    return uuid_to_h5


@beartype.beartype
def extract_h5_file(
    h5_fpath: pathlib.Path, tasks: list[tuple[str, pathlib.Path]]
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
def load_and_filter_source_pyarrow(
    dpath: pathlib.Path,
    source: str,
    taxa_filter: TaxaFilter | None,
    class_filter: str,
    order_filter: tuple[str, ...],
    label_column: str,
) -> list[ImagePair]:
    """Load and filter source using pyarrow."""
    source_dpath = dpath / f"source={source}"
    if not source_dpath.exists():
        logger.warning(f"Source directory not found: {source_dpath}")
        return []

    parquet_files = sorted(source_dpath.glob("*.parquet"))
    logger.info(f"Processing {len(parquet_files)} parquet files from {source}")

    # Build filter value sets for pyarrow
    filter_col: str | None = None
    filter_values: pa.Array | None = None
    if taxa_filter is not None:
        # Use first match column (usually 'family')
        filter_col = taxa_filter.match_cols[0]
        filter_values = pa.array(taxa_filter.taxa_unique[filter_col].unique().to_list())
    elif class_filter:
        filter_col = "class"
        filter_values = pa.array([class_filter])
    elif order_filter:
        filter_col = "order"
        filter_values = pa.array(list(order_filter))

    results: list[ImagePair] = []
    for fpath in parquet_files:
        # Read only needed columns
        cols = ["uuid", label_column]
        if filter_col and filter_col not in cols:
            cols.append(filter_col)
        table = pq.read_table(fpath, columns=cols)

        # Apply filter
        if filter_col and filter_values is not None:
            mask = pc.is_in(table[filter_col], value_set=filter_values)
            table = table.filter(mask)

        # Filter nulls in label column
        mask = pc.is_valid(table[label_column])
        table = table.filter(mask)

        n_matches = table.num_rows
        if n_matches > 0:
            uuids = table["uuid"].to_pylist()
            labels = table[label_column].to_pylist()
            results.extend(ImagePair(u, lbl) for u, lbl in zip(uuids, labels))
            logger.info(f"  {fpath.name}: {n_matches} matches")

    logger.info(f"  {source} total: {len(results)} matching images")
    return results


@beartype.beartype
def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)
    logger.info(f"Starting extraction: sources={cfg.sources}")

    # Build filter once
    taxa_filter = None
    if cfg.taxa_file is not None:
        taxa_filter = TaxaFilter.load(cfg.taxa_file)

    # Step 1: Collect all (uuid, label) pairs from all sources
    logger.info("Step 1: Collecting image metadata from sources...")
    all_pairs: list[ImagePair] = []
    for source in cfg.sources:
        pairs = load_and_filter_source_pyarrow(
            cfg.resolved_taxa_dpath,
            source,
            taxa_filter,
            cfg.class_filter,
            cfg.order_filter,
            cfg.label_column,
        )
        all_pairs.extend(pairs)
        logger.info(f"After {source}: {len(all_pairs)} total pairs")

    if not all_pairs:
        logger.warning("No matching images found. Check your filter settings.")
        return

    logger.info(f"Total: {len(all_pairs)} matching images across all sources")

    # Step 2: Load lookup tables ONCE with all UUIDs
    logger.info("Step 2: Loading lookup tables...")
    uuids = {pair.uuid for pair in all_pairs}
    uuid_to_h5 = load_lookup_tables_pyarrow(cfg.lookup_tables_dpath, uuids)
    uuid_to_label = {pair.uuid: pair.label for pair in all_pairs}

    # Step 3: Build h5_tasks grouped by h5 file
    logger.info("Step 3: Building extraction tasks...")
    h5_tasks: dict[pathlib.Path, list[tuple[str, pathlib.Path]]] = {}
    n_skipped = 0
    for uuid, h5_file in uuid_to_h5.items():
        label = uuid_to_label.get(uuid)
        if label is None:
            continue
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

    # Step 4: Extract images in parallel
    logger.info(f"Step 4: Extracting with {cfg.n_workers} workers...")
    h5_items = list(h5_tasks.items())
    n_total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.n_workers) as executor:
        futures = {
            executor.submit(extract_h5_file, h5_path, tasks): h5_path
            for h5_path, tasks in h5_items
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            n_success = future.result()
            n_total += n_success
            if (i + 1) % 100 == 0 or (i + 1) == len(futures):
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

    # Request enough CPUs to get desired memory (~10GB per CPU on nextgen)
    n_cpus = max(cfg.n_workers, cfg.mem_gb // 10)

    params = dict(
        job_name="extract-tol",
        time=int(cfg.n_hours * 60),
        partition=cfg.slurm_partition,
        gpus_per_node=0,
        ntasks_per_node=1,
        cpus_per_task=n_cpus,
        stderr_to_stdout=True,
        account=cfg.slurm_acct,
    )
    executor.update_parameters(**params)
    job = executor.submit(main, cfg)
    logger.info(f"Submitted job {job.job_id} to Slurm. Logs: {cfg.log_to}")
    try:
        job.result()
        logger.info(f"Job {job.job_id} completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Job {job.job_id} failed: {e}")
        return 1


if __name__ == "__main__":
    cli()
