import concurrent.futures
import csv
import dataclasses
import logging
import os
import pathlib
import shutil
import typing as tp

import beartype
import tyro

from saev import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("format_ade20k")

SPLITS = ("training", "validation")
SUBDIRS = ("images", "annotations")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    src_root_dpath: pathlib.Path = pathlib.Path(
        "/fs/ess/PAS2136/samuelstevens/datasets/ADEChallengeData2016"
    )
    """Raw ADE20K root with images/, annotations/, and sceneCategories.txt."""
    dump_to_dpath: pathlib.Path | None = None
    """Destination SegFolder root. Defaults to src_root_dpath when unset."""
    scene_categories_fname: str = "sceneCategories.txt"
    """Scene categories filename (stem label)."""
    labels_csv_fname: str = "labels.csv"
    """Optional labels.csv filename (stem,label_col)."""
    label_col: str = "scene"
    """Label column name when labels_csv_fname is present."""
    image_labels_fname: str = "image_labels.txt"
    """Output image label filename (stem label)."""
    link_mode: tp.Literal["symlink", "hardlink", "copy"] = "symlink"
    """How to materialize files when dump_to_dpath differs from src_root_dpath."""
    n_threads: int = 16
    """Number of concurrent threads for linking or copying."""
    job_size: int = 1024
    """Number of files to process per job."""


@beartype.beartype
def _read_labels(cfg: Config, root_dpath: pathlib.Path) -> dict[str, str]:
    labels: dict[str, str] = {}

    labels_csv_fpath = root_dpath / cfg.labels_csv_fname
    if labels_csv_fpath.is_file():
        with open(labels_csv_fpath, newline="") as fd:
            reader = csv.DictReader(fd)
            assert reader.fieldnames is not None, f"{labels_csv_fpath} has no header"
            assert reader.fieldnames[0] == "stem", (
                f"First column must be 'stem', got '{reader.fieldnames[0]}'"
            )
            assert cfg.label_col in reader.fieldnames, (
                f"Missing label column '{cfg.label_col}' in {labels_csv_fpath}"
            )
            for row in reader:
                stem = row["stem"]
                label = row[cfg.label_col]
                assert stem, f"Empty stem in {labels_csv_fpath}"
                assert label, f"Empty label for stem '{stem}' in {labels_csv_fpath}"
                assert stem not in labels, (
                    f"Duplicate stem '{stem}' in {labels_csv_fpath}"
                )
                labels[stem] = label
        logger.info("Loaded %d labels from %s", len(labels), labels_csv_fpath)
        return labels

    scene_fpath = root_dpath / cfg.scene_categories_fname
    assert scene_fpath.is_file(), f"Missing scene categories file: {scene_fpath}"
    with open(scene_fpath) as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            stem, _, label = line.rpartition(" ")
            assert stem, f"Malformed line in {scene_fpath}: '{line}'"
            assert label, f"Malformed line in {scene_fpath}: '{line}'"
            assert stem not in labels, f"Duplicate stem '{stem}' in {scene_fpath}"
            labels[stem] = label
    logger.info("Loaded %d labels from %s", len(labels), scene_fpath)
    return labels


@beartype.beartype
def _collect_stems(root_dpath: pathlib.Path) -> set[str]:
    stems: set[str] = set()
    if not root_dpath.is_dir():
        return stems

    for split in SPLITS:
        split_dpath = root_dpath / split
        if not split_dpath.is_dir():
            continue
        for fpath in split_dpath.rglob("*"):
            if not fpath.is_file():
                continue
            stems.add(fpath.stem)
    return stems


@beartype.beartype
def _write_image_labels(labels: dict[str, str], *, output_fpath: pathlib.Path) -> None:
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fpath, "w") as fd:
        for stem, label in sorted(labels.items()):
            fd.write(f"{stem} {label}\n")
    logger.info("Wrote %d labels to %s", len(labels), output_fpath)


@beartype.beartype
def _collect_pairs(
    src_root_dpath: pathlib.Path,
    dst_root_dpath: pathlib.Path,
    *,
    subdir: str,
    split: str,
) -> list[tuple[pathlib.Path, pathlib.Path]]:
    src_split_dpath = src_root_dpath / subdir / split
    if not src_split_dpath.is_dir():
        return []

    dst_split_dpath = dst_root_dpath / subdir / split
    dst_split_dpath.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for src_fpath in src_split_dpath.rglob("*"):
        if not src_fpath.is_file():
            continue
        rel = src_fpath.relative_to(src_split_dpath)
        dst_fpath = dst_split_dpath / rel
        pairs.append((src_fpath, dst_fpath))
    return pairs


@beartype.beartype
def _link_one(cfg: Config, src_fpath: pathlib.Path, dst_fpath: pathlib.Path) -> None:
    if dst_fpath.exists():
        return

    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    if cfg.link_mode == "copy":
        shutil.copy2(src_fpath, dst_fpath)
        return

    if cfg.link_mode == "hardlink":
        os.link(src_fpath, dst_fpath)
        return

    if cfg.link_mode == "symlink":
        os.symlink(src_fpath, dst_fpath)
        return

    assert False, f"Unknown link_mode: {cfg.link_mode}"


@beartype.beartype
def _link_batch(
    cfg: Config, pairs: list[tuple[pathlib.Path, pathlib.Path]], start: int, end: int
) -> None:
    for src_fpath, dst_fpath in pairs[start:end]:
        _link_one(cfg, src_fpath, dst_fpath)


@beartype.beartype
def _materialize_pairs(cfg: Config, pairs: list[tuple[pathlib.Path, pathlib.Path]]):
    if not pairs:
        return

    n_pairs = len(pairs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.n_threads) as pool:
        futs = [
            pool.submit(_link_batch, cfg, pairs, start, end)
            for start, end in helpers.batched_idx(n_pairs, cfg.job_size)
        ]
        for fut in helpers.progress(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc="linking",
        ):
            if err := fut.exception():
                logger.warning("Exception: %s", err)


@beartype.beartype
def main(cfg: tp.Annotated[Config, tyro.conf.arg(name="")]) -> int:
    src_root_dpath = cfg.src_root_dpath
    assert src_root_dpath.is_dir(), f"Missing source root: {src_root_dpath}"

    dump_to_dpath = cfg.dump_to_dpath or src_root_dpath
    labels = _read_labels(cfg, src_root_dpath)
    assert labels, "No labels found for ADE20K"

    img_stems = _collect_stems(src_root_dpath / "images")
    ann_stems = _collect_stems(src_root_dpath / "annotations")
    label_stems = set(labels)

    assert img_stems, "No images found under images/"
    assert ann_stems, "No annotations found under annotations/"
    assert img_stems == label_stems, (
        f"Image stems ({len(img_stems)}) != label stems ({len(label_stems)})"
    )
    assert ann_stems == img_stems, (
        f"Annotation stems ({len(ann_stems)}) != image stems ({len(img_stems)})"
    )

    _write_image_labels(labels, output_fpath=dump_to_dpath / cfg.image_labels_fname)

    if dump_to_dpath == src_root_dpath:
        logger.info("Using in-place formatting at %s", dump_to_dpath)
        return 0

    for subdir in SUBDIRS:
        for split in SPLITS:
            dst_split_dpath = dump_to_dpath / subdir / split
            dst_split_dpath.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for subdir in SUBDIRS:
        for split in SPLITS:
            pairs.extend(
                _collect_pairs(
                    src_root_dpath,
                    dump_to_dpath,
                    subdir=subdir,
                    split=split,
                )
            )

    logger.info(
        "Materializing %d files via %s into %s",
        len(pairs),
        cfg.link_mode,
        dump_to_dpath,
    )
    _materialize_pairs(cfg, pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Config)))
