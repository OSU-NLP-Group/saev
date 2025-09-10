import dataclasses
import typing as tp

import beartype
import torch.utils.data


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Butterflies:
    pass


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class FishVista:
    pass


Config = Butterflies | FishVista


@beartype.beartype
def get_dataset(
    cfg: Config, *, img_transform, sample_transform=None
) -> torch.utils.data.Dataset:
    """
    Gets the dataset for the current experiment; delegates construction to dataset-specific functions.

    Args:
        cfg: Experiment config.
        img_transform: Image transform to be applied to each image.
        sample_transform: Transform to be applied to each sample dict.
    Returns:
        A dataset that has dictionaries with 'image', 'index', 'target' and 'label' keys (and maybe some extras), containing examples.
    """
    if isinstance(cfg, Butterflies):
        return ButterfliesDataset(
            cfg, img_transform=img_transform, sample_transform=sample_transform
        )
    elif isinstance(cfg, FishVista):
        return FishVistaDataset(
            cfg, img_transform=img_transform, sample_transform=sample_transform
        )
    else:
        tp.assert_never(cfg)


class ButterfliesDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Butterflies, *, img_transform=None, sample_transform=None):
        pass


class FishVistaDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Butterflies, *, img_transform=None, sample_transform=None):
        pass
