# /// script
# requires-python = ">=3.12"
# dependencies = ["beartype", "tyro", "requests", "beautifulsoup4", "polars", "tqdm"]
# ///
"""
Scrape FishBase for species traits from FishVista dataset.

Usage:
    uv run contrib/trait_discovery/scripts/scrape_fishbase.py --help
    uv run contrib/trait_discovery/scripts/scrape_fishbase.py
"""

import dataclasses
import logging
import pathlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import beartype
import polars as pl
import requests
import tqdm
import tyro
from bs4 import BeautifulSoup

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("scrape_fishbase")

MIRRORS = ["org", "se", "de", "net.br", "org.au", "us", "ca"]

BINARY_TRAITS = [
    # Habitat/position
    "demersal",
    "benthopelagic",
    "bathydemersal",
    "pelagic",
    "pelagic-neritic",
    "pelagic-oceanic",
    "reef-associated",
    # Depth zones
    "epipelagic",
    "mesopelagic",
    "bathypelagic",
    "abyssopelagic",
    # Water type
    "marine",
    "freshwater",
    "brackish",
    # Migration
    "anadromous",
    "catadromous",
    "amphidromous",
    "potamodromous",
    "limnodromous",
    "oceanodromous",
    "non-migratory",
]

NUMERIC_TRAITS = [
    "min_depth_m",
    "max_depth_m",
    "usual_min_depth_m",
    "usual_max_depth_m",
    "min_ph",
    "max_ph",
    "min_dh",
    "max_dh",
]

ALL_TRAITS = BINARY_TRAITS + NUMERIC_TRAITS


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    fishvista_dpath: pathlib.Path = pathlib.Path(
        "/fs/ess/PAS2136/samuelstevens/datasets/fish-vista"
    )
    """Path to FishVista dataset with CSV files."""
    output_fpath: pathlib.Path = pathlib.Path("data/fishvista_fishbase.csv")
    """Output CSV path."""
    err_fpath: pathlib.Path = pathlib.Path("data/fishvista_fishbase_errors.csv")
    """Error log CSV (genus, species, reason)."""
    crawl_delay: int = 10
    """Seconds between requests per mirror (from robots.txt)."""
    timeout: int = 30
    """HTTP request timeout."""
    max_retries: int = 3
    """Maximum number of retries per request."""


@beartype.beartype
def load_species(fishvista_dpath: pathlib.Path) -> list[tuple[str, str, str]]:
    """Load unique (family, genus, epithet) from all FishVista CSVs."""
    all_dfs = []
    for csv_fpath in fishvista_dpath.glob("*.csv"):
        # Skip ND_Processing_Files subdirectory
        if "ND_Processing_Files" in str(csv_fpath):
            continue
        try:
            df = pl.read_csv(csv_fpath, infer_schema_length=None)
            if "family" in df.columns and "standardized_species" in df.columns:
                all_dfs.append(df.select(["family", "standardized_species"]))
        except Exception as err:
            logger.warning("Failed to read %s: %s", csv_fpath, err)

    if not all_dfs:
        return []

    combined = pl.concat(all_dfs).unique()

    # Deduplicate on (genus, epithet), keeping first family seen
    seen: set[tuple[str, str]] = set()
    species_list = []
    for row in combined.iter_rows(named=True):
        family = row["family"]
        species_str = row["standardized_species"]
        if not species_str or not isinstance(species_str, str):
            continue
        parts = species_str.strip().split()
        if len(parts) < 2:
            logger.warning("Invalid species format: %s", species_str)
            continue
        genus, epithet = parts[0], parts[1]
        if (genus, epithet) in seen:
            continue
        seen.add((genus, epithet))
        species_list.append((family, genus, epithet))

    return species_list


@beartype.beartype
def load_existing(output_fpath: pathlib.Path) -> set[tuple[str, str]]:
    """Load already-scraped (genus, epithet) pairs from output CSV."""
    if not output_fpath.exists():
        return set()
    try:
        df = pl.read_csv(output_fpath, null_values=["?"])
        return {(row["genus"], row["species"]) for row in df.iter_rows(named=True)}
    except Exception:
        return set()


@beartype.beartype
def parse_environment(html: str) -> dict[str, str | float | None] | None:
    """Parse FishBase HTML for Environment section traits. Returns None if page is invalid."""
    soup = BeautifulSoup(html, "html.parser")

    # Find the Environment/Biology section - look for text patterns
    text = soup.get_text(separator=" ", strip=True)

    # Check for invalid pages
    if "not in the public version of FishBase" in text:
        return None

    result: dict[str, str | float | None] = {trait: "" for trait in ALL_TRAITS}

    # Binary traits: check if keywords appear in environment description
    text_lower = text.lower()
    for trait in BINARY_TRAITS:
        # Handle hyphenated traits
        trait_pattern = trait.replace("-", r"[\s-]")
        if re.search(rf"\b{trait_pattern}\b", text_lower):
            result[trait] = 1.0
        else:
            result[trait] = ""

    # Parse depth range: "depth range X - Y m"
    depth_match = re.search(
        r"depth range\s*[:\s]*(\?|\d+)\s*-\s*(\?|\d+)\s*m", text_lower
    )
    if depth_match:
        min_d, max_d = depth_match.groups()
        result["min_depth_m"] = float(min_d) if min_d != "?" else "?"
        result["max_depth_m"] = float(max_d) if max_d != "?" else "?"

    # Parse usual depth: "usually X - Y m" or "usually ? - Y m"
    usual_match = re.search(r"usually\s*(\?|\d+)\s*-\s*(\?|\d+)\s*m", text_lower)
    if usual_match:
        min_u, max_u = usual_match.groups()
        result["usual_min_depth_m"] = float(min_u) if min_u != "?" else "?"
        result["usual_max_depth_m"] = float(max_u) if max_u != "?" else "?"

    # Parse pH range: "pH range: X - Y" or "pH X - Y"
    ph_match = re.search(
        r"ph\s*(?:range)?[:\s]*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", text_lower
    )
    if ph_match:
        result["min_ph"] = float(ph_match.group(1))
        result["max_ph"] = float(ph_match.group(2))

    # Parse dH range: "dH range: X - Y" or "dH X - Y"
    dh_match = re.search(
        r"dh\s*(?:range)?[:\s]*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", text_lower
    )
    if dh_match:
        result["min_dh"] = float(dh_match.group(1))
        result["max_dh"] = float(dh_match.group(2))

    return result


class MirrorWorker:
    """Worker that fetches from a single FishBase mirror with rate limiting."""

    def __init__(self, tld: str, crawl_delay: int, timeout: int, max_retries: int):
        self.tld = tld
        self.crawl_delay = crawl_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_request = 0.0
        self.lock = threading.Lock()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _wait_for_rate_limit(self):
        """Wait to respect crawl-delay."""
        with self.lock:
            elapsed = time.time() - self.last_request
            if elapsed < self.crawl_delay:
                time.sleep(self.crawl_delay - elapsed)
            self.last_request = time.time()

    def fetch(self, genus: str, epithet: str) -> tuple[str | None, str | None, str]:
        """Fetch HTML for species with retries. Returns (html, error_reason, url)."""
        genus_cap = genus.capitalize()
        epithet_low = epithet.lower()
        url = f"https://www.fishbase.{self.tld}/summary/{genus_cap}_{epithet_low}.html"

        last_error = None
        for attempt in range(self.max_retries):
            self._wait_for_rate_limit()

            try:
                resp = self.session.get(url, timeout=self.timeout)
                status = resp.status_code

                # Success
                if 200 <= status < 300:
                    return resp.text, None, url

                # 404 - species not found (don't retry)
                if status == 404:
                    return None, f"http_{status}", url

                # Client errors (4xx except 429) - don't retry
                if 400 <= status < 500 and status != 429:
                    return None, f"http_{status}", url

                # Server errors (5xx) or 429 - retry
                last_error = f"http_{status}"

            except requests.Timeout:
                last_error = "timeout"
            except requests.RequestException as err:
                last_error = f"request_error:{type(err).__name__}"

            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                backoff = (2**attempt) * self.crawl_delay
                time.sleep(backoff)

        return None, last_error, url


def log_error(
    err_fpath: pathlib.Path, genus: str, epithet: str, reason: str, lock: threading.Lock
):
    """Append failed species to error log."""
    with lock:
        with open(err_fpath, "a") as fd:
            fd.write(f"{genus},{epithet},{reason}\n")


def write_result(
    output_fpath: pathlib.Path,
    family: str,
    genus: str,
    epithet: str,
    traits: dict,
    url: str,
    lock: threading.Lock,
):
    """Append result to output CSV."""
    with lock:
        with open(output_fpath, "a") as fd:
            row = (
                [genus, epithet, family]
                + [str(traits.get(t, "")) for t in ALL_TRAITS]
                + [url]
            )
            fd.write(",".join(row) + "\n")


@beartype.beartype
def main(cfg: Config):
    logger.info("Loading species from FishVista CSVs...")
    all_species = load_species(cfg.fishvista_dpath)
    logger.info("Found %d unique species", len(all_species))

    # Load already-scraped species for resumability
    done = load_existing(cfg.output_fpath)
    todo = [(f, g, e) for f, g, e in all_species if (g, e) not in done]
    logger.info("Already scraped: %d, remaining: %d", len(done), len(todo))

    if not todo:
        logger.info("Nothing to do!")
        return

    # Create output directory
    cfg.output_fpath.parent.mkdir(parents=True, exist_ok=True)
    cfg.err_fpath.parent.mkdir(parents=True, exist_ok=True)

    # Initialize error log with header if doesn't exist
    if not cfg.err_fpath.exists():
        with open(cfg.err_fpath, "w") as fd:
            fd.write("genus,species,reason\n")

    # Create workers for each mirror
    workers = [
        MirrorWorker(tld, cfg.crawl_delay, cfg.timeout, cfg.max_retries)
        for tld in MIRRORS
    ]

    # Locks for thread-safe file writes
    output_lock = threading.Lock()
    err_lock = threading.Lock()

    # Write header if starting fresh
    if not cfg.output_fpath.exists() or len(done) == 0:
        with open(cfg.output_fpath, "w") as fd:
            header = ["genus", "species", "family"] + ALL_TRAITS + ["url"]
            fd.write(",".join(header) + "\n")

    def process_species(args: tuple[int, tuple[str, str, str]]) -> bool:
        idx, (family, genus, epithet) = args
        # Try all mirrors, starting from idx to distribute load
        errors = []
        for i in range(len(workers)):
            worker = workers[(idx + i) % len(workers)]
            html, error, url = worker.fetch(genus, epithet)

            if html is None:
                # Fetch failed - try next mirror
                errors.append(f"{worker.tld}:{error}")
                continue

            # Got HTML - try to parse it
            try:
                traits = parse_environment(html)
            except Exception as err:
                errors.append(f"{worker.tld}:parse_failed:{type(err).__name__}")
                continue

            if traits is None:
                # Page exists but has no data (e.g., "not in public version")
                errors.append(f"{worker.tld}:not_public")
                continue

            # Success - write result
            write_result(
                cfg.output_fpath,
                family,
                genus,
                epithet,
                traits,
                url,
                output_lock,
            )
            return True

        # All mirrors failed
        log_error(cfg.err_fpath, genus, epithet, ";".join(errors), err_lock)
        return False

    # Process with thread pool
    n_success = 0
    n_failed = 0

    with ThreadPoolExecutor(max_workers=len(MIRRORS)) as pool:
        futures = [
            pool.submit(process_species, (i, spec)) for i, spec in enumerate(todo)
        ]

        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="Scraping"
        ):
            try:
                if future.result():
                    n_success += 1
                else:
                    n_failed += 1
            except Exception as err:
                logger.error("Unexpected error: %s", err)
                n_failed += 1

    logger.info("Done! Success: %d, Failed: %d", n_success, n_failed)
    logger.info("Output: %s", cfg.output_fpath)
    logger.info("Errors: %s", cfg.err_fpath)


if __name__ == "__main__":
    main(tyro.cli(Config))
