import dataclasses
import typing as tp

import beartype
import torch
from jaxtyping import Shaped, jaxtyped
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

import saev.data
import saev.helpers

n_classes = 10


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Result:
    method: tp.Literal["linear-clf", "random", "k-means", "sae", "pca"]
    n_prototypes: int
    n_train: int
    seed: int
    class_idx: int
    average_precision: float
    best_prototype_idx: int
    vit_family: str
    vit_ckpt: str
    layer: int
    d_vit: int
    extra: dict[str, object] = dataclasses.field(default_factory=dict)


@beartype.beartype
class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: saev.data.datasets.SegFolder,
        patch_labeling: tp.Literal["mode", "no-bg"],
    ):
        self.cfg = cfg
        self.patch_labeling = patch_labeling

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

        self.samples = saev.data.datasets.SegFolderDataset(
            self.cfg,
            img_transform=img_transform,
            seg_transform=seg_transform,
            sample_transform=sample_transform,
        )

        self.patch_size_px = (16, 16)

    def __getitem__(self, i: int) -> dict[str, object]:
        # Get patch and pixel level semantic labels.
        sample = self.samples[i]
        pixel_labels = sample["segmentation"].squeeze()
        if self.patch_labeling == "mode":
            patch_labels = pixel_labels.mode(axis=1).values
        elif self.patch_labeling == "no-bg":
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
