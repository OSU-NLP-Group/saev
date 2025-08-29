# Trains multiple linear probes in parallel on DINOv3's FishVista activations.
import dataclasses
import itertools
import logging
import typing as tp

import beartype
import einops
import torch.utils.data
from PIL import Image
from torchvision.transforms import v2

import saev.data.images
import saev.data.models
import saev.data.transforms

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
    patch_label_mode: tp.Literal["mode", "no-bg"] = "no-bg"

    # Linear Probe
    lrs: list[float] = dataclasses.field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    wds: list[float] = dataclasses.field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    n_epochs: int = 400
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


@beartype.beartype
class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg: saev.data.images.SegFolder):
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
            cfg,
            img_transform=img_transform,
            seg_transform=seg_transform,
            sample_transform=sample_transform,
        )

        self.patch_size_px = (16, 16)

    def __getitem__(self, i: int) -> dict[str, object]:
        # Get patch and pixel level semantic labels.
        sample = self.samples[i]
        pixel_labels = sample["segmentation"].squeeze()
        patch_labels = pixel_labels.mode(axis=1).values

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
        dataset = Dataset(dataclasses.replace(cfg.imgs, split="training"))
    else:
        shuffle = False
        dataset = Dataset(dataclasses.replace(cfg.imgs, split="validation"))

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
            break

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

        # if epoch % cfg.eval_every == 0 or epoch + 1 == cfg.n_epochs:
        #     with torch.inference_mode():
        #         pred_label_list, true_label_list = [], []
        #         for batch in val_dataloader:
        #             imgs_bnd = batch["image"].to(cfg.device)
        #             grid = batch["grid"].to(cfg.device)
        #             with torch.inference_mode():
        #                 vit_acts_bnd = vit(imgs_bnd, grid=grid)
        #                 # Remove CLS
        #                 vit_acts_bnd = vit_acts_bnd[:, 1:, :]

        #             pixel_labels_bnk = batch["pixel_labels"]
        #             true_label_list.append(pixel_labels_bnk)

        #             logits_mbnc = torch.stack([model(vit_acts_bnd) for model in models])
        #             logits_mb_cn = einops.rearrange(
        #                 logits_mbnc,
        #                 "models batch n classes -> (models batch) classes n",
        #             )

        #             pred_mb_n = lib.fishvista.utils.batched_upsample_and_pred(
        #                 logits_mb_cn, size=(224, 224), mode="bilinear"
        #             )
        #             del logits_mb_cn

        #             pred_MBWH = einops.rearrange(
        #                 pred_MB_WH,
        #                 "(models batch) width height -> models batch width height",
        #                 models=len(models),
        #             )
        #             pred_label_list.append(pred_MBWH)

        #         pred_labels_MNWH = torch.cat(pred_label_list, dim=1).int()
        #         true_labels_MNWH = (
        #             torch.cat(true_label_list).int().expand(len(models), -1, -1, -1)
        #         )

        #         logger.info("Evaluated all validation batchs.")
        #         class_ious_MC = get_class_ious(
        #             pred_labels_MNWH,
        #             true_labels_MNWH.expand(len(models), -1, -1, -1),
        #             n_classes,
        #         )
        #         mean_ious_M = einops.reduce(
        #             class_ious_MC, "models classes -> models", "mean"
        #         )
        #         acc_M = einops.reduce(
        #             (pred_labels_MNWH == true_labels_MNWH).float(),
        #             "models n width height -> models",
        #             "mean",
        #         )

        #     logger.info(
        #         "epoch: %d, step: %d, max val miou: %.5f, max val acc: %.3f",
        #         epoch,
        #         global_step,
        #         mean_ious_M.max().item(),
        #         acc_M.max().item() * 100,
        #     )

        #     for cfg, model in zip(cfgs, models):
        #         dump(cfg, model, step=global_step)
