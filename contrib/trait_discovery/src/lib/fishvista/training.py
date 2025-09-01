"""
Trains multiple linear probes in parallel on DINOv3's FishVista activations.

Size key:
* B: Batch size
* D: ViT activation dimension (typically 768 or 1024)
* N: Number of images
"""

import dataclasses
import itertools
import logging
import typing as tp

import beartype
import einops
import numpy as np
import sklearn.metrics
import torch.utils.data
from jaxtyping import Shaped, jaxtyped
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

import saev.data.images
import saev.data.models
import saev.data.transforms
import saev.helpers

n_classes = 10

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("fishvista")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    vit_ckpt: str = "checkpoints/dinov3_vitl16.pth"
    """Specific DINOv3 variant."""
    d_vit: int = 1024
    """Residual ViT dimension."""
    imgs: saev.data.images.SegFolder = dataclasses.field(
        default_factory=saev.data.images.SegFolder
    )
    """Which dataset to use."""
    vit_batch_size: int = 256
    """Batch size for ViT inference."""
    n_workers: int = 8
    """Number of dataloader workers."""
    patch_labeling: tp.Literal["mode", "no-bg"] = "no-bg"

    # Linear Probe
    lrs: list[float] = dataclasses.field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    wds: list[float] = dataclasses.field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    n_epochs: int = 200
    """Number of linear probing epochs."""
    eval_every: int = 20

    # Hardware
    device: str = "cuda"
    """Which device to use."""
    n_hours: float = 24.0
    """Slurm job length."""
    slurm_acct: str = ""
    """Slurm account string."""
    slurm_partition: str = ""
    """Slurm partition."""
    log_to: str = "./logs"
    """Where to log Slurm job stdout/stderr."""


@jaxtyped(typechecker=beartype.beartype)
def patch_label_no_bg(
    pixel_labels_nd: Shaped[Tensor, "n k"], *, n_classes: int, bg: int = 0
) -> Shaped[Tensor, " n"]:
    x = pixel_labels_nd.to(torch.long)
    N, _ = x.shape

    # counts[i, c] = number of times class c appears in patch i
    offsets = torch.arange(N, device=x.device).unsqueeze(1) * n_classes
    flat = (x + offsets).reshape(-1)
    counts = torch.bincount(flat, minlength=N * n_classes).reshape(N, n_classes)

    nonbg = counts.clone()
    nonbg[:, bg] = 0
    has_nonbg = nonbg.sum(dim=1) > 0
    nonbg_arg = nonbg.argmax(dim=1)
    bg = torch.full_like(nonbg_arg, bg)
    return torch.where(has_nonbg, nonbg_arg, bg)


@beartype.beartype
class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Config):
        self.cfg = cfg

        img_transform = v2.Compose([
            saev.data.transforms.FlexResize(patch_size=16, n_patches=640),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])

        seg_transform = v2.Compose([
            saev.data.transforms.FlexResize(
                patch_size=16, n_patches=640, resample=Image.NEAREST
            ),
            v2.ToImage(),
        ])
        sample_transform = v2.Compose([
            saev.data.transforms.Patchify(patch_size=16, n_patches=640),
            saev.data.transforms.Patchify(
                patch_size=16, n_patches=640, key="segmentation"
            ),
        ])

        self.samples = saev.data.images.SegFolderDataset(
            cfg.imgs,
            img_transform=img_transform,
            seg_transform=seg_transform,
            sample_transform=sample_transform,
        )

        self.patch_size_px = (16, 16)

    def __getitem__(self, i: int) -> dict[str, object]:
        # Get patch and pixel level semantic labels.
        sample = self.samples[i]
        pixel_labels = sample["segmentation"].squeeze()
        if self.cfg.patch_labeling == "mode":
            patch_labels = pixel_labels.mode(axis=1).values
        elif self.cfg.patch_labeling == "no-bg":
            patch_labels = patch_label_no_bg(pixel_labels, bg=0, n_classes=10)
        else:
            tp.assert_never(self.cfg.patch_labeling)

        return {
            "index": i,
            "image": sample["image"],
            "pixel_labels": pixel_labels,
            "patch_labels": patch_labels,
            "grid": sample["grid"],
        }

    def __len__(self) -> int:
        return len(self.samples)


@beartype.beartype
def get_dataloader(cfg: Config, *, is_train: bool):
    if is_train:
        shuffle = True
        img_cfg = dataclasses.replace(cfg.imgs, split="training")
    else:
        shuffle = False
        img_cfg = dataclasses.replace(cfg.imgs, split="validation")

    cfg = dataclasses.replace(cfg, imgs=img_cfg)
    dataset = Dataset(cfg)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.vit_batch_size,
        num_workers=cfg.n_workers,
        shuffle=shuffle,
        persistent_workers=(cfg.n_workers > 0),
    )


@beartype.beartype
def make_models(cfg: Config) -> tuple[torch.nn.ModuleList, list[dict[str, object]]]:
    param_groups = []
    models = []
    for lr, wd in itertools.product(cfg.lrs, cfg.wds):
        model = torch.nn.Linear(cfg.d_vit, n_classes)
        models.append(model)
        param_groups.append({
            "params": model.parameters(),
            "lr": lr,
            "weight_decay": wd,
        })

    return torch.nn.ModuleList(models), param_groups


@beartype.beartype
def train(cfg: Config):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than float16 and almost as accurate as float32. This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    train_dataloader = get_dataloader(cfg, is_train=True)
    val_dataloader = get_dataloader(cfg, is_train=False)

    vit = saev.data.models.make_vit("dinov3", cfg.vit_ckpt).to(cfg.device)
    models, params = make_models(cfg)
    models = models.to(cfg.device)

    optim = torch.optim.AdamW(params, lr=0, weight_decay=0)

    global_step = 0

    for epoch in range(cfg.n_epochs):
        models.train()
        for batch in train_dataloader:
            imgs_bnd = batch["image"].to(cfg.device)
            grid = batch["grid"].to(cfg.device)
            with torch.inference_mode():
                vit_acts_bnd = vit(imgs_bnd, grid=grid)
                # Remove CLS
                vit_acts_bnd = vit_acts_bnd[:, 1:, :]

            patch_labels_bn = batch["patch_labels"].to(cfg.device)
            patch_labels_mbn = patch_labels_bn.expand(len(models), -1, -1)
            logits_mbnc = torch.stack([model(vit_acts_bnd) for model in models])
            loss = torch.nn.functional.cross_entropy(
                logits_mbnc.view(-1, n_classes), patch_labels_mbn.reshape(-1)
            )
            loss.backward()
            optim.step()
            optim.zero_grad()

            global_step += 1

        # Show last batch's loss and acc.
        acc_M = einops.reduce(
            (logits_mbnc.argmax(axis=-1) == patch_labels_mbn).float(),
            "models batch n -> models",
            "mean",
        )
        logger.info(
            "epoch: %d, step: %d, mean train loss: %.5f, max train acc: %.3f",
            epoch,
            global_step,
            loss.item(),
            acc_M.max().item() * 100,
        )

        if epoch % cfg.eval_every == 0 or epoch + 1 == cfg.n_epochs:
            with torch.inference_mode():
                pred_label_list, true_label_list = [], []
                logits_list = []
                for batch in val_dataloader:
                    imgs_bnd = batch["image"].to(cfg.device)
                    grid = batch["grid"].to(cfg.device)
                    vit_acts_bnd = vit(imgs_bnd, grid=grid)
                    # Remove CLS
                    vit_acts_bnd = vit_acts_bnd[:, 1:, :]

                    logits_mbnc = torch.stack([model(vit_acts_bnd) for model in models])
                    preds_mbn = logits_mbnc.argmax(axis=-1)
                    pred_label_list.append(preds_mbn)
                    logits_list.append(logits_mbnc)

                    labels_bn = batch["patch_labels"].to(cfg.device)
                    true_label_list.append(labels_bn.expand(len(models), -1, -1))

                pred_labels_mn = einops.rearrange(
                    torch.cat(pred_label_list, dim=1).int(), "m b n -> m (b n)"
                )
                true_labels_mn = einops.rearrange(
                    torch.cat(true_label_list, dim=1).int(), "m b n -> m (b n)"
                )
                logits_mnc = einops.rearrange(
                    torch.cat(logits_list, dim=1), "m b n c -> m (b n) c"
                )

                logger.info("Evaluated all validation batchs.")
                acc_m = einops.reduce(
                    (pred_labels_mn == true_labels_mn).float(),
                    "models n -> models",
                    "mean",
                )

            # Compute per-class AP and mAP (one-vs-rest) for each model
            with torch.inference_mode():
                y_mn = true_labels_mn.cpu().numpy()
                s_mnc = logits_mnc.cpu().numpy()

                ap_mc = np.zeros((len(models), n_classes), dtype=np.float64)
                for m in range(len(models)):
                    for c in range(n_classes):
                        y_true = (y_mn[m] == c).astype(np.int32)
                        scores = s_mnc[m, :, c]
                        # Guard against degenerate cases with no positives or no negatives
                        if y_true.max() == 0 or y_true.min() == 1:
                            ap = np.nan
                        else:
                            ap = float(
                                sklearn.metrics.average_precision_score(y_true, scores)
                            )
                        ap_mc[m, c] = ap

                mAP_m = np.nanmean(ap_mc, axis=1)
                best_i = int(np.nanargmin(-mAP_m))  # index of max mAP
                best_mAP = float(mAP_m[best_i])

            logger.info(
                "epoch: %d, step: %d, best val acc: %.3f, best val mAP: %.4f",
                epoch,
                global_step,
                acc_m.max().item() * 100,
                best_mAP,
            )

            # for cfg, model in zip(cfgs, models):
            #     dump(cfg, model, step=global_step)
