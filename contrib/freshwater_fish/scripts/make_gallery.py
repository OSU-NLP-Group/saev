"""Build a self-contained HTML gallery of SAE feature visualizations.

Given a run directory and shard ID with pre-rendered images, produces a single
HTML file with all SAE highlight images base64-encoded inline. JavaScript
pagination keeps the DOM light (20 features per page) even when the file
contains thousands of images.

Usage:
    uv run python contrib/freshwater_fish/scripts/make_gallery.py \
        --run /fs/ess/PAS2136/samuelstevens/saev/runs/um6hbn05 \
        --shard 8692dfa9 \
        --out gallery.html
"""

import argparse
import base64
import io
import json
import logging
import pathlib

import polars as pl
from PIL import Image

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("make_gallery")


def png_to_jpeg_b64(fpath: pathlib.Path, quality: int = 80) -> str:
    """Read a PNG, convert to JPEG, return base64-encoded data URI."""
    img = Image.open(fpath).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def load_species_labels_from_hf(hf_config: str, hf_split: str) -> dict[str, str]:
    """Download FishVista metadata from HuggingFace and build stem -> label map.

    Label format: (Family) Genus species
    """
    from datasets import load_dataset

    ds = load_dataset(
        "imageomics/fish-vista", hf_config, split=hf_split, streaming=True
    )
    stem_to_label: dict[str, str] = {}
    for row in ds:
        stem = pathlib.Path(row["filename"]).stem
        parts = row["standardized_species"].split()
        genus = parts[0].capitalize() if parts else "?"
        species = parts[1] if len(parts) > 1 else "sp."
        family = row["family"].capitalize()
        stem_to_label[stem] = f"({family}) {genus} {species}"
    return stem_to_label


def load_species_labels(
    dataset_dpath: pathlib.Path, split: str, stem_to_label: dict[str, str]
) -> list[str]:
    """Build an index-aligned list of species labels from sorted image filenames."""
    img_dir = dataset_dpath / "images" / split
    assert img_dir.is_dir(), f"No images directory at '{img_dir}'"
    fnames = sorted(f.name for f in img_dir.iterdir() if f.is_file())
    return [stem_to_label.get(pathlib.Path(f).stem, "?") for f in fnames]


def build_features(
    images_dpath: pathlib.Path,
    var_df: pl.DataFrame,
    species: list[str] | None,
) -> list[dict]:
    """Collect features that have pre-rendered images on disk."""
    features = []
    available = {
        int(d.name)
        for d in images_dpath.iterdir()
        if d.is_dir() and d.name.isdigit()
    }

    for row in var_df.iter_rows(named=True):
        fid = row["feature"]
        if fid not in available:
            continue

        feature_dpath = images_dpath / str(fid)
        topk_idx = row["topk_example_idx"]

        # Deduplicate example indices (matching visuals.py's seen_example_idx).
        deduped_idx: list[int] = []
        seen_idx: set[int] = set()
        for ex_idx in topk_idx:
            if ex_idx not in seen_idx:
                seen_idx.add(ex_idx)
                deduped_idx.append(ex_idx)

        imgs = []
        for j in range(100):
            sae_fpath = feature_dpath / f"{j}_sae_img.png"
            if not sae_fpath.exists():
                break

            label = "?"
            if species is not None and j < len(deduped_idx):
                ex_idx = deduped_idx[j]
                if 0 <= ex_idx < len(species):
                    label = species[ex_idx]

            imgs.append({"src": png_to_jpeg_b64(sae_fpath), "label": label})

        if not imgs:
            continue

        freq_pct = 10 ** row["log10_freq"] * 100
        mean_val = 10 ** row["log10_value"]
        features.append({
            "id": fid,
            "log10_freq": round(row["log10_freq"], 3),
            "log10_value": round(row["log10_value"], 3),
            "freq_pct": f"{freq_pct:.4f}",
            "mean_val": f"{mean_val:.2f}",
            "images": imgs,
        })

    features.sort(key=lambda f: f["log10_freq"])
    return features


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAE Feature Gallery</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 20px; }
h1 { margin-bottom: 8px; }
.subtitle { color: #666; margin-bottom: 16px; font-size: 14px; }
.controls { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
.controls button { padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; font-size: 13px; }
.controls button:hover { background: #e8e8e8; }
.controls button.active { background: #333; color: #fff; border-color: #333; }
.controls select { padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }
.page-info { font-size: 13px; color: #666; }
.feature { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.feature-header { font-size: 15px; font-weight: 600; margin-bottom: 4px; }
.feature-meta { font-size: 12px; color: #888; margin-bottom: 10px; }
.feature-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; }
.feature-grid figure { margin: 0; text-align: center; }
.feature-grid img { width: 100%; border-radius: 4px; display: block; }
.feature-grid figcaption { font-size: 11px; color: #666; margin-top: 3px; font-style: italic; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.nav { display: flex; gap: 8px; justify-content: center; margin: 12px 0; }
.nav button { padding: 8px 20px; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; font-size: 14px; }
.nav button:hover { background: #e8e8e8; }
.note-box { background: #e8f4fd; border: 1px solid #b8daef; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; line-height: 1.5; }
.dir-toggle { background: none; border: 1px solid #aaa; border-radius: 3px; padding: 2px 6px; cursor: pointer; font-size: 12px; margin-left: 2px; }
.dir-toggle:hover { background: #e8e8e8; }
</style>
</head>
<body>
<h1>SAE Feature Gallery</h1>
<p class="subtitle">TITLE_PLACEHOLDER</p>

<div class="note-box">
<strong>How to read this:</strong> Each card is one SAE "feature" (a pattern the model learned). The highlighted images show which parts of the fish activate this feature (brighter = stronger activation). Species labels are shown below each image in the format (Family) Genus species. Use the controls to sort and browse.<br><br>
<strong>SAE run:</strong> <code>RUN_ID_PLACEHOLDER</code>
</div>

<div class="controls">
  <span>Sort:</span>
  <button id="sort-freq" class="active" onclick="sortBy('freq')">Frequency <span class="dir-toggle" id="dir-freq" onclick="event.stopPropagation(); flipDir('freq')">&#9650;</span></button>
  <button id="sort-value" onclick="sortBy('value')">Mean value <span class="dir-toggle" id="dir-value" onclick="event.stopPropagation(); flipDir('value')">&#9660;</span></button>
  <button id="sort-id" onclick="sortBy('id')">Feature ID <span class="dir-toggle" id="dir-id" onclick="event.stopPropagation(); flipDir('id')">&#9650;</span></button>
  <span style="margin-left:12px">Per page:</span>
  <select id="per-page" onchange="setPerPage(this.value)">
    <option value="10">10</option>
    <option value="20" selected>20</option>
    <option value="50">50</option>
  </select>
  <span class="page-info" id="page-info-top"></span>
</div>

<div class="nav" id="nav-top">
  <button onclick="prevPage()">&#8592; Previous</button>
  <button onclick="nextPage()">Next &#8594;</button>
</div>

<div id="gallery"></div>

<div class="nav" id="nav-bottom">
  <button onclick="prevPage()">&#8592; Previous</button>
  <button onclick="nextPage()">Next &#8594;</button>
</div>

<div style="text-align:center; margin-top:4px;">
  <span class="page-info" id="page-info-bottom"></span>
</div>

<script>
const FEATURES = FEATURES_JSON;

let sortKey = "freq";
let page = 0;
let perPage = 20;
let sorted = [...FEATURES];
// asc=true means ascending (smallest first). Default directions per sort key.
let dirs = { freq: true, value: false, id: true };

function cmp(key, asc) {
  const field = key === "freq" ? "log10_freq" : key === "value" ? "log10_value" : "id";
  return (a, b) => asc ? a[field] - b[field] : b[field] - a[field];
}

function updateDirArrows() {
  for (const key of ["freq", "value", "id"]) {
    document.getElementById("dir-" + key).innerHTML = dirs[key] ? "&#9650;" : "&#9660;";
  }
}

function sortBy(key) {
  sortKey = key;
  page = 0;
  sorted.sort(cmp(key, dirs[key]));
  document.querySelectorAll(".controls > button").forEach(b => b.classList.remove("active"));
  document.getElementById("sort-" + key).classList.add("active");
  render();
}

function flipDir(key) {
  dirs[key] = !dirs[key];
  updateDirArrows();
  if (sortKey === key) sortBy(key);
}

function setPerPage(n) { perPage = parseInt(n); page = 0; render(); }
function prevPage() { if (page > 0) { page--; render(); window.scrollTo(0, 0); } }
function nextPage() { if ((page + 1) * perPage < sorted.length) { page++; render(); window.scrollTo(0, 0); } }

function render() {
  const start = page * perPage;
  const end = Math.min(start + perPage, sorted.length);
  const totalPages = Math.ceil(sorted.length / perPage);
  const info = "Showing " + (start + 1) + "\\u2013" + end + " of " + sorted.length + " features (page " + (page + 1) + "/" + totalPages + ")";
  document.getElementById("page-info-top").textContent = info;
  document.getElementById("page-info-bottom").textContent = info;

  const el = document.getElementById("gallery");
  el.innerHTML = "";
  for (let i = start; i < end; i++) {
    const f = sorted[i];
    const div = document.createElement("div");
    div.className = "feature";

    let grid = "";
    for (const img of f.images) {
      grid += '<figure><img src="' + img.src + '" loading="lazy"><figcaption>' + img.label + '</figcaption></figure>';
    }

    div.innerHTML =
      '<div class="feature-header">Feature ' + f.id + '</div>' +
      '<div class="feature-meta">Fires on ' + f.freq_pct + '% of patches | Mean activation: ' + f.mean_val + '</div>' +
      '<div class="feature-grid">' + grid + '</div>';
    el.appendChild(div);
  }
}

render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, required=True, help="Run directory (e.g. /fs/ess/.../runs/um6hbn05)")
    parser.add_argument("--shard", type=str, required=True, help="Shard ID (e.g. 8692dfa9)")
    parser.add_argument("--dataset", type=pathlib.Path, default=None, help="Dataset root (segfolder with images/ dir, for label ordering)")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split name (default: validation)")
    parser.add_argument("--hf-config", type=str, default="trait_segmentation", help="HuggingFace FishVista config name")
    parser.add_argument("--hf-split", type=str, default="val", help="HuggingFace split name (default: val)")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("gallery.html"), help="Output HTML path")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality (0-100)")
    parser.add_argument("--title", type=str, default="", help="Gallery subtitle/description")
    args = parser.parse_args()

    inference_dpath = args.run / "inference" / args.shard
    images_dpath = inference_dpath / "images"
    var_fpath = inference_dpath / "var.parquet"

    msg = f"No images directory at '{images_dpath}'"
    assert images_dpath.is_dir(), msg
    msg = f"No var.parquet at '{var_fpath}'"
    assert var_fpath.exists(), msg

    var_df = pl.read_parquet(var_fpath)
    logger.info("Loaded var.parquet with %d features.", var_df.height)

    species = None
    if args.dataset is not None:
        logger.info("Downloading species labels from HuggingFace...")
        stem_to_label = load_species_labels_from_hf(args.hf_config, args.hf_split)
        logger.info("Got %d labels from HF.", len(stem_to_label))
        species = load_species_labels(args.dataset, args.split, stem_to_label)
        logger.info("Mapped %d image indices to species labels.", len(species))

    logger.info("Collecting and converting images...")
    features = build_features(images_dpath, var_df, species)
    logger.info("Packaged %d features.", len(features))

    n_imgs = sum(len(f["images"]) for f in features)
    logger.info("Total images: %d", n_imgs)

    title = args.title or f"SAE run {args.run.name}, shard {args.shard} | {len(features)} features, {n_imgs} images"
    html = HTML_TEMPLATE.replace("FEATURES_JSON", json.dumps(features))
    html = html.replace("TITLE_PLACEHOLDER", title)
    html = html.replace("RUN_ID_PLACEHOLDER", args.run.name)

    args.out.write_text(html)
    size_mb = args.out.stat().st_size / 1024 / 1024
    logger.info("Wrote %s (%.1f MB)", args.out, size_mb)


if __name__ == "__main__":
    main()
