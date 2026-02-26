"""Upload Pareto-optimal DINOv3 SAE checkpoints to Hugging Face.

Usage:
    uv run contrib/trait_discovery/scripts/push_dinov3.py [--no-dry-run]

After uploading, manually create/update HF collections:
- New collection: "Towards Open-Ended Visual Scientific Discovery with Sparse Autoencoders"
- Existing collection: osunlp/sae-v
"""

import dataclasses
import hashlib
import json
import logging
import math
import pathlib
import shutil
import tempfile
import warnings

warnings.filterwarnings("ignore", module="pydantic")

import beartype
import wandb

import saev.nn

logger = logging.getLogger(__name__)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    dry_run: bool = True
    """Show what would be uploaded without actually uploading."""
    runs_root: pathlib.Path = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
    """Root directory containing run checkpoints."""
    wandb_entity: str = "samuelstevens"
    """W&B entity (username or team)."""
    wandb_project: str = "saev"
    """W&B project name."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Repo:
    repo_id: str
    run_ids: dict[int, list[str]]
    title: str


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RunMetrics:
    run_id: str
    layer: int
    l0: float
    mse: float


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class StagedRun:
    run_id: str
    layer: int
    l0: float
    mse: float
    path: str
    sha256: str


# All run IDs from sweep files (including previously commented-out ones).
# These are fixed historical IDs and will not change.

# fmt: off
vits_run_ids: dict[int, list[str]] = {
    6: ["3zih0tpa", "ct5w12zx", "f5qog0he", "l6da5024", "xc0h7cq7", "u5aqv7t7", "usiwpodi", "gcpbnr9n", "nbqbvh45", "r2sf1w19", "sfpco1tn"],
    7: ["l9stkmwt", "tbfdr3cc", "c7w1t9jc", "xvxn1ed1", "zaxl9nqu", "r2g7cj5v", "u6r4jsdm", "d5ej9yuh", "pvtt26ky", "hu77o1op", "36538viq"],
    8: ["p0z7t1ci", "125xh1t9", "cx2g9omb", "q2i5lq7h", "qwf07reo", "tzwgex0i", "y563o43r", "5gjy7lwi", "33oh6osq", "bvsb2257"],
    9: ["qt0fmmxm", "fj8b9r5o", "1o4uc5bf", "z1qvy51u", "1ihxsv0i", "euoj6wv0", "flfplqsa", "5o0mby2h", "ickedctl"],
    10: ["chn5wi3x", "219r3phu", "knglrhzb", "21d1kgyk", "jt45lucm", "6hrok1al", "qrjtyj70", "3j06kxdt", "g4dexqq1"],
    11: ["jgu19fzx", "8yd05vxi", "utmjp20e", "gc6iqrf2", "5ewxrjg4", "x7e75z6t", "hyda2tk7", "36ztscy4"],
}

vitb_run_ids: dict[int, list[str]] = {
    6: ["y7uk853s", "odd8ogb4", "db4qjvkf", "e4i667sz", "extgn8yv", "ku0rwwex", "8hyzbyht", "vsvkrqfg", "t2xcozei", "gyyfc054", "lppv40ws"],
    7: ["iyb7ec1w", "rxjh04w7", "688bm8ht", "f6i2ow8w", "uiqc8e2f", "40j71aj2", "xucm378k", "21trz0ik", "knz7yndg", "1hcm0oqu", "6yhupj05"],
    8: ["wgh9hgih", "poe1kh3i", "ttghd72n", "sabum27l", "q6pg7hl9", "r3opp7dy", "2obnw9ky", "bk7iwhfu", "o4cheohl", "dk6k8hc0", "rc82kpln"],
    9: ["cozptrw2", "cynce806", "1aod3v62", "5g0ez3ix", "zvx4qkov", "na2k2dyp", "6h9n14t3", "g6ga929x", "tqk8igwb", "hs18j6i2", "t0qdoi9u", "893a4vol"],
    10: ["eqht2edc", "bzmeiyat", "1hjlnu1s", "ssoshhfv", "oc5jcdu8", "jpnwfh3w", "2we45xxf", "bv1h09se", "0akkhcjf", "yb185c6g", "jjewtqwp"],
    11: ["n1xwev0z", "ef657fwa", "qoc1660r", "6crsj9gj", "d4v8aruu", "7mpdhd0n", "abhe5g2j", "22p3bnt8"],
}

vitl_relu_run_ids: dict[int, list[str]] = {
    13: ["jsqj2arm", "3hr3d3w0", "lq18pdy9", "kk60aru4", "fh0jta0t", "y8q60ohz", "u7tz0xii", "ag8agm56", "ja4zp5kn", "220r8j1q", "fjravp6a", "yhu9d2z9", "fkl5sxba"],
    15: ["fi7qafny", "txmrh5nd", "qtvsac3e", "aq8vvjub", "xfgouwrz", "rsmrpkly", "e9oeml82"],
    17: ["di427rrs", "edx9q34f", "pn1f9cge", "n7pv6rkj", "4rhpmk3f", "syuerpif", "egid27oa", "jqx6qdxv", "vrepu5ey", "av2qk4oj", "vkdu21ck"],
    19: ["y6osup5x", "yi5zik0k", "aa30r3nm", "sq1ccr13", "0tj48gqd", "7dr58kwn", "2uqtzyv6", "s96104bm"],
    21: ["qcyausyf", "i6pxw0q9", "zyj9edre", "x7py290w", "v4pyroov", "71u6kzuq", "t1ip1brk", "pz4up9fd", "36al8yw7", "y8vhxwya"],
    23: ["lnleoyf6", "ibt2fgta", "6l12fjm9", "rfic94if", "t1vh0qy1", "mccrm7u8", "t88ez13w", "eosnewqp", "fxcpfysr", "kd2pd8rs", "9drbwvhg", "1qynjykb", "0pz90ly4", "ybm0jqi4", "2pdk23cz", "9fn4l6rf"],
}

vitl_topk_run_ids: dict[int, list[str]] = {
    13: ["3ld8ilmo", "l03epvhu", "co7dpa0w", "kpadjov4", "2edpn91i", "1up044nl"],
    15: ["6r92o6t6", "e4w7u0np", "jsr327fs", "emz255bp", "ffqb9b3n", "3hzenf5e"],
    17: ["tkdd41tq", "4g4lbmgs", "h8nfg6ci", "2hsh4w50", "jjz6a7ja", "huzxe3hu"],
    19: ["0c4mlnn7", "6x4t5t76", "xk0a9w3g", "cdu13t6j", "hh7d7yop", "32zm1zcd"],
    21: ["rez38zbu", "jxxje744", "2k6kq9f2", "jttb6ijl", "s5srn2q7", "qurkdz1r"],
    23: ["a95jzikd", "elwq2g19", "ztnu4ml1", "flqkcam7", "s3pqewz1", "l8hooa3r"],
}
# fmt: on

repos = [
    Repo("osunlp/SAE_DINOv3_ViT-S-16_IN1K", vits_run_ids, "DINOv3 ViT-S/16"),
    Repo("osunlp/SAE_DINOv3_ViT-B-16_IN1K", vitb_run_ids, "DINOv3 ViT-B/16"),
    Repo("osunlp/SAE_DINOv3_ViT-L-16_IN1K", vitl_relu_run_ids, "DINOv3 ViT-L/16"),
    Repo(
        "osunlp/SAE_DINOv3_TopK_ViT-L-16_IN1K",
        vitl_topk_run_ids,
        "DINOv3 TopK ViT-L/16",
    ),
]


def ckpt_fpath(runs_root: pathlib.Path, run_id: str) -> pathlib.Path:
    return runs_root / run_id / "checkpoint" / "sae.pt"


@beartype.beartype
def fetch_metrics(run_ids: dict[int, list[str]], cfg: Config) -> list[RunMetrics]:
    """Query W&B for L0 and MSE for each run."""
    api = wandb.Api()
    metrics = []
    for layer, ids in sorted(run_ids.items()):
        for run_id in ids:
            run = api.run(f"{cfg.wandb_entity}/{cfg.wandb_project}/{run_id}")
            l0 = run.summary.get("eval/l0")
            mse = run.summary.get("eval/mse")
            if l0 is None or mse is None:
                logger.warning(
                    "Run %s missing metrics (l0=%s, mse=%s), skipping.", run_id, l0, mse
                )
                continue
            metrics.append(
                RunMetrics(run_id=run_id, layer=layer, l0=float(l0), mse=float(mse))
            )
    return metrics


@beartype.beartype
def select_pareto(metrics: list[RunMetrics], *, max_n: int = 6) -> list[RunMetrics]:
    """Select up to max_n Pareto-optimal runs per layer, log-spaced by L0."""
    by_layer: dict[int, list[RunMetrics]] = {}
    for m in metrics:
        by_layer.setdefault(m.layer, []).append(m)

    selected: list[RunMetrics] = []
    for layer in sorted(by_layer):
        runs = sorted(by_layer[layer], key=lambda r: (r.l0, r.mse))

        # Pareto frontier: keep only runs where MSE strictly improves.
        frontier: list[RunMetrics] = []
        best_mse = float("inf")
        for run in runs:
            if run.mse < best_mse:
                best_mse = run.mse
                frontier.append(run)

        if not frontier:
            continue

        if len(frontier) <= max_n:
            selected.extend(frontier)
            continue

        # Endpoints + interior via log-L0 quantiles.
        picked_i: set[int] = {0, len(frontier) - 1}
        n_interior = max_n - 2
        log_lo = math.log1p(frontier[0].l0)
        log_hi = math.log1p(frontier[-1].l0)
        for i in range(1, n_interior + 1):
            target = log_lo + (log_hi - log_lo) * i / (n_interior + 1)
            best_j = min(
                (j for j in range(len(frontier)) if j not in picked_i),
                key=lambda j: abs(math.log1p(frontier[j].l0) - target),
            )
            picked_i.add(best_j)

        selected.extend(frontier[j] for j in sorted(picked_i))

    return selected


@beartype.beartype
def preflight(selected: list[RunMetrics], runs_root: pathlib.Path) -> None:
    """Load each checkpoint to verify it exists and loads correctly."""
    for run in selected:
        fpath = ckpt_fpath(runs_root, run.run_id)
        assert fpath.exists(), f"Checkpoint missing: {fpath}"
        saev.nn.load(fpath)
        logger.info("OK %s (layer %d)", run.run_id, run.layer)


@beartype.beartype
def sha256_file(fpath: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(fpath, "rb") as fd:
        for chunk in iter(lambda: fd.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@beartype.beartype
def stage(
    selected: list[RunMetrics], staging_dpath: pathlib.Path, runs_root: pathlib.Path
) -> list[StagedRun]:
    """Copy sae.pt files into staging directory and compute sha256s."""
    staged = []
    for run in selected:
        src = ckpt_fpath(runs_root, run.run_id)
        rel = f"layer_{run.layer}/{run.run_id}/sae.pt"
        dst = staging_dpath / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        staged.append(
            StagedRun(
                run_id=run.run_id,
                layer=run.layer,
                l0=run.l0,
                mse=run.mse,
                path=rel,
                sha256=sha256_file(dst),
            )
        )
    return staged


@beartype.beartype
def make_readme(repo: Repo, staged: list[StagedRun]) -> str:
    """Generate model card README.md."""
    ordered = sorted(staged, key=lambda s: (s.layer, s.l0))
    rows = "\n".join(
        f"| {s.run_id} | {s.layer} | {s.l0:.1f} | {s.mse:.4f} | `{s.path}` |"
        for s in ordered
    )
    example = ordered[-1]

    return f"""---
license: mit
---

# SAE for Meta's {repo.title} trained on ImageNet-1K Activations

* **Homepage:** https://osu-nlp-group.github.io/saev
* **Code:** https://github.com/OSU-NLP-Group/saev
* **Preprint:** https://arxiv.org/abs/2511.17735
* **Demos:** https://osu-nlp-group.github.io/saev#demos
* **Point of Contact:** [Sam Stevens](mailto:stevens.994@buckeyemail.osu.edu)

## Checkpoints

Each checkpoint is a sparse autoencoder (SAE) trained on a different layer with a different sparsity level. Pick the checkpoint that matches your target layer and desired sparsity (L0).

| Run ID | Layer | L0 | MSE | Path |
|--------|-------|----|-----|------|
{rows}

This metadata is also available in `manifest.jsonl` at the repo root for programmatic access.

## Usage

```python
from huggingface_hub import hf_hub_download

import saev.nn

path = hf_hub_download("{repo.repo_id}", "{example.path}")
sae = saev.nn.load(path)
```

## Inference Instructions

Follow the instructions [here](https://osu-nlp-group.github.io/saev/api/saev/#inference-instructions).
"""


@beartype.beartype
def make_manifest(staged: list[StagedRun]) -> str:
    """Generate manifest.jsonl with machine-readable metadata per checkpoint."""
    ordered = sorted(staged, key=lambda s: (s.layer, s.l0))
    lines = [
        json.dumps({
            "run_id": s.run_id,
            "layer": s.layer,
            "l0": round(s.l0, 2),
            "mse": round(s.mse, 6),
            "sha256": s.sha256,
            "path": s.path,
        })
        for s in ordered
    ]
    return "\n".join(lines) + "\n"


@beartype.beartype
def upload(staging_dpath: pathlib.Path, repo_id: str) -> None:
    """Create repo (if needed) and upload staging directory."""
    import huggingface_hub as hfhub

    hfapi = hfhub.HfApi()
    hfapi.create_repo(repo_id, repo_type="model", exist_ok=True)
    hfapi.upload_folder(
        folder_path=str(staging_dpath), repo_id=repo_id, repo_type="model"
    )
    logger.info("Uploaded %s", repo_id)


@beartype.beartype
def smoke_test(repo_id: str, staged: list[StagedRun]) -> None:
    """Download and load one checkpoint from HF to verify the upload."""
    import huggingface_hub as hfhub

    test = staged[0]
    fpath = hfhub.hf_hub_download(repo_id, test.path)
    saev.nn.load(fpath)
    logger.info("Smoke test passed for %s: loaded %s", repo_id, test.run_id)


@beartype.beartype
def print_selection(repo: Repo, selected: list[RunMetrics]) -> None:
    """Print selection table for dry-run output."""
    print(f"\n{'=' * 60}")
    print(f"  {repo.repo_id}")
    print(f"  {len(selected)} checkpoints selected")
    print(f"{'=' * 60}")
    print(f"  {'Run ID':<12} {'Layer':>5} {'L0':>8} {'MSE':>10}")
    print(f"  {'-' * 12} {'-' * 5} {'-' * 8} {'-' * 10}")
    for run in sorted(selected, key=lambda r: (r.layer, r.l0)):
        print(f"  {run.run_id:<12} {run.layer:>5} {run.l0:>8.1f} {run.mse:>10.4f}")


@beartype.beartype
def main(cfg: Config):
    """Upload Pareto-optimal DINOv3 SAE checkpoints to HF."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    for repo in repos:
        logger.info("Fetching metrics for %s...", repo.repo_id)
        metrics = fetch_metrics(repo.run_ids, cfg)
        selected = select_pareto(metrics)
        print_selection(repo, selected)

        if cfg.dry_run:
            continue

        logger.info("Preflight: loading %d checkpoints...", len(selected))
        preflight(selected, cfg.runs_root)

        with tempfile.TemporaryDirectory() as tmpdir:
            staging_dpath = pathlib.Path(tmpdir)
            staged = stage(selected, staging_dpath, cfg.runs_root)
            (staging_dpath / "README.md").write_text(make_readme(repo, staged))
            (staging_dpath / "manifest.jsonl").write_text(make_manifest(staged))

            logger.info("Uploading %s (%d checkpoints)...", repo.repo_id, len(staged))
            upload(staging_dpath, repo.repo_id)

        # staged metadata persists after tmpdir cleanup; smoke_test downloads from HF.
        smoke_test(repo.repo_id, staged)

    if not cfg.dry_run:
        logger.info("Done! Verify repos at https://huggingface.co/osunlp")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))
