"""
Checks which checkpoints have the best validation loss, mean IoU, class-specific IoU, validation accuracy, and qualitative results.

Writes results to CSV files and hparam graphs (in-progress).
"""

import csv
import json
import logging
import os

import altair as alt
import beartype
import einops
import polars as pl
import torch

import saev.helpers

from . import config, training

logger = logging.getLogger(__name__)


@beartype.beartype
def main(cfg: config.Validation):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than float16 and almost as accurate as float32. This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    dataset = training.Dataset(cfg.imgs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=False,
        persistent_workers=(cfg.n_workers > 0),
    )

    ckpts = load_ckpts(cfg.ckpt_root, device=cfg.device)

    vit = training.DinoV2()
    vit = vit.to(cfg.device)
    pred_label_list, true_label_list = [], []
    for batch in saev.helpers.progress(dataloader, every=1):
        imgs_BWHC = batch["image"].to(cfg.device)
        with torch.inference_mode():
            vit_acts = vit(imgs_BWHC)
            acts_BWHD = einops.rearrange(
                vit_acts,
                "batch (width height) dim -> batch width height dim",
                width=16,
                height=16,
            )
        pixel_labels_BWH = batch["pixel_labels"]
        true_label_list.append(pixel_labels_BWH)

        logits_MBWHC = torch.stack([model(acts_BWHD) for _, model in ckpts])
        logits_MB_CWH = einops.rearrange(
            logits_MBWHC,
            "models batch width height classes -> (models batch) classes width height",
        )

        pred_MB_WH = training.batched_upsample_and_pred(
            logits_MB_CWH, size=(224, 224), mode="bilinear"
        )
        del logits_MB_CWH

        pred_MBWH = einops.rearrange(
            pred_MB_WH,
            "(models batch) width height -> models batch width height",
            models=len(ckpts),
        )
        pred_label_list.append(pred_MBWH)

    pred_labels_MNWH = torch.cat(pred_label_list, dim=1).int()
    true_labels_MNWH = torch.cat(true_label_list).int().expand(len(ckpts), -1, -1, -1)

    logger.info("Evaluated all validation batchs.")
    ious_MC = training.get_class_ious(
        pred_labels_MNWH,
        true_labels_MNWH.expand(len(ckpts), -1, -1, -1),
        training.n_classes,
    )
    acc_M = (pred_labels_MNWH == true_labels_MNWH).float().mean(dim=(1, 2, 3)) * 100

    lookup = {}
    with open(os.path.join(cfg.imgs.root, "objectInfo150.txt")) as fd:
        for row in csv.DictReader(fd, delimiter="\t"):
            lookup[int(row["Idx"])] = row["Name"]

    class_iou_headers = [name for _, name in sorted(lookup.items())]

    os.makedirs(cfg.dump_to, exist_ok=True)

    # Save CSV file
    csv_fpath = os.path.join(cfg.dump_to, "results.csv")
    with open(csv_fpath, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(
            ["learning_rate", "weight_decay", "acc", "mean_iou"] + class_iou_headers
        )
        for (c, _), acc, ious_C in zip(ckpts, acc_M, ious_MC):
            writer.writerow(
                [c.learning_rate, c.weight_decay, acc.item(), ious_C.mean().item()]
                + ious_C.tolist()
            )

    # Save hyperparameter sweep charts
    df = pl.read_csv(csv_fpath).with_columns(
        pl.col("weight_decay").add(1e-9).alias("weight_decay")
    )
    alt.Chart(df).mark_point().encode(
        alt.X(alt.repeat("column"), type="quantitative").scale(type="log"),
        alt.Y(alt.repeat("row"), type="quantitative").scale(zero=False),
    ).repeat(row=["acc", "mean_iou"], column=["learning_rate", "weight_decay"]).save(
        os.path.join(cfg.dump_to, "hparam-sweeps.png")
    )


@beartype.beartype
def load_ckpts(
    root: str, *, device: str = "cpu"
) -> list[tuple[config.Train, torch.nn.Module]]:
    """
    Loads the latest checkpoints for each directory within root.

    Arguments:
        root: directory containing other directories with cfg.json and model_step{step}.pt files.
        device: where to load models.

    Returns:
        List of cfg, model pairs.
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"Checkpoint root not found: {root}")

    results = []

    # Find all subdirectories that contain cfg.json
    for dname in os.listdir(root):
        dpath = os.path.join(root, dname)
        if not os.path.isdir(dpath):
            continue

        cfg_path = os.path.join(dpath, "cfg.json")
        if not os.path.exists(cfg_path):
            continue

        # Load config
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        # Handle the nested dataclasses.
        cfg_dict["imgs"] = saev.config.Ade20kDataset(**cfg_dict["imgs"])
        cfg_dict.pop("patch_size_px")
        cfg = config.Train(**cfg_dict)

        # Load latest model checkpoint
        model = training.load_latest(dpath, device=device)

        results.append((cfg, model))

    if not results:
        raise FileNotFoundError(f"No valid checkpoint directories found in: {root}")

    return results
