import random

import pytest
import torch

from saev.data.iterable import Config as IterableConfig
from saev.data.iterable import DataLoader
from saev.data.torch import Config as TorchConfig
from saev.data.torch import Dataset

N_SAMPLES = 25_000  # quick but representative
BATCH_SIZE = 4_096
PATCHES_PER_IMG = {256, 196}  # acceptable values; read from metadata


@pytest.fixture(scope="module")
def loaders():
    # minimal cfg pointing to a few shards
    ref_ds = Dataset(
        TorchConfig(
            shard_root="/fs/scratch/PAS2136/samuelstevens/cache/saev/713c9e11788e5a4258d4201704cf083047b3fccacc2a25224cdd326520cd0d18",
            patches="patches",
            layer=23,
            scale_mean=False,
            scale_norm=False,
        )
    )
    fast_dl = DataLoader(IterableConfig())
    return ref_ds, fast_dl


def _global_index(img_i: int, patch_i: int, n_patches: int) -> int:
    """
    Map (image_i, patch_i) to linear index used by Dataset when
    cfg.patches == "patches" and cfg.layer is fixed.
    """
    return img_i * n_patches + patch_i


def test_loader_matches_reference(loaders):
    ds, dl = loaders

    # pick n_patches_per_img directly from metadata to avoid hard-coding
    n_patches = ds.metadata.n_patches_per_img
    assert n_patches in PATCHES_PER_IMG

    random.seed(0)
    n_checked = 0

    for batch in dl:
        for i in range(batch["act"].shape[0]):
            img_i = int(batch["image_i"][i])
            patch_i = int(batch["patch_i"][i])
            if patch_i < 0:  # CLS / mean-pool examples; skip here
                continue

            idx = _global_index(img_i, patch_i, n_patches)
            ref = ds[idx]

            err_msg = f"Mismatch at global idx {idx}"
            assert torch.allclose(batch["act"][i], ref["act"], atol=0, rtol=0), err_msg
            n_checked += 1
            if n_checked >= N_SAMPLES:
                return


# def test_same_seed_same_stream():
#     stream1 = list(stream_batches(n=10_000, seed=42, workers=4))
#     stream2 = list(stream_batches(n=10_000, seed=42, workers=8))
#     assert stream1 == stream2  # tuples compare element-wise


# def test_diff_seed_diff_stream():
#     a = list(stream_batches(n=1000, seed=1))
#     b = list(stream_batches(n=1000, seed=2))
#     overlap = sum(x == y for x, y in zip(a, b))
#     assert overlap < 0.1 * len(a)  # fewer than 10 % identical
