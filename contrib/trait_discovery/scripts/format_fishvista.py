# contrib/trait_discovery/scripts/format_fishvista.py

import concurrent.futures
import dataclasses
import logging
import pathlib
import shutil

import beartype
import polars as pl
import tyro

from saev import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("format")

split_mapping = {"train": "training", "val": "validation", "test": "test"}


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    fv_root: pathlib.Path = pathlib.Path("./data/fish-vista")
    """Where you downloaded FishVista from HuggingFace."""
    dump_to: pathlib.Path = pathlib.Path("./data/segfolder")
    """Where to write the new directories."""
    cls_label_fname: str = "image_labels.txt"
    """Filename for the image labels file."""
    n_threads: int = 16
    """Number of concurrent threads for copying files on disk."""
    job_size: int = 256
    """Number of images to copy per job."""


@beartype.beartype
def _cp(cfg: Config, fv_split: str, tgt_split: str, start: int, end: int):
    seg_df = pl.read_csv(cfg.fv_root / f"segmentation_{fv_split}.csv")

    for (fname,) in seg_df.select("filename").slice(start, end - start).iter_rows():
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
        fname = f"{pathlib.Path(fname).stem}.png"
        src_fpath = cfg.fv_root / "segmentation_masks" / "images" / fname
        if not src_fpath.exists():
            continue

        dst_fpath = cfg.dump_to / "annotations" / tgt_split / fname
        if dst_fpath.exists():
            continue

        shutil.copy2(src_fpath, dst_fpath)


@beartype.beartype
def _write_labels(cfg: Config):
    with open(cfg.dump_to / cfg.cls_label_fname, "w") as fd:
        for fv_split in split_mapping:
            seg_df = pl.read_csv(cfg.fv_root / f"segmentation_{fv_split}.csv")
            for fname, family, sci_name in seg_df.select(
                "filename", "family", "standardized_species"
            ).iter_rows():
                fname = pathlib.Path(fname)
                cls_label = f"{family}_{sci_name.replace(' ', '_')}"
                fd.write(f"{fname.stem} {cls_label}\n")


@beartype.beartype
def main(cfg: Config):
    """
    Convert FishVista to a format useable with SegFolderDataset in src/saev/data/images.py.
    """

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.n_threads) as pool:
        futs = [pool.submit(_write_labels, cfg)]

        for fv_split, tgt_split in split_mapping.items():
            (cfg.dump_to / "images" / tgt_split).mkdir(exist_ok=True, parents=True)
            (cfg.dump_to / "annotations" / tgt_split).mkdir(exist_ok=True, parents=True)
            seg_df = pl.read_csv(cfg.fv_root / f"segmentation_{fv_split}.csv")

            futs.extend([
                pool.submit(_cp, cfg, fv_split, tgt_split, start, end)
                for start, end in helpers.batched_idx(seg_df.height, cfg.job_size)
            ])

        for fut in helpers.progress(
            concurrent.futures.as_completed(futs), total=len(futs), desc="copying"
        ):
            if err := fut.exception():
                logger.warning("Exception: %s", err)


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Config)))
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
