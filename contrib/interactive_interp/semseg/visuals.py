"""
Propose features for manual verification.
"""

import beartype
import einops
import numpy as np
import torch
from jaxtyping import Int, Shaped, jaxtyped

import saev.helpers
import saev.nn

from . import config, training


@beartype.beartype
@torch.no_grad
def main(cfg: config.Visuals):
    sae = saev.nn.load(cfg.sae_ckpt)
    sae = sae.to(cfg.device)

    dataset = training.Dataset(cfg.acts, cfg.imgs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=False,
        persistent_workers=(cfg.n_workers > 0),
    )

    tp = torch.zeros((sae.cfg.d_sae,), dtype=int, device=cfg.device)
    fp = torch.zeros((sae.cfg.d_sae,), dtype=int, device=cfg.device)
    fn = torch.zeros((sae.cfg.d_sae,), dtype=int, device=cfg.device)

    for batch in saev.helpers.progress(dataloader):
        pixel_labels = einops.rearrange(
            batch["pixel_labels"],
            "batch (w pw) (h ph) -> batch w h (pw ph)",
            # TODO: change from hard-coded values
            pw=16,
            ph=16,
        )
        unique, counts = axis_unique(pixel_labels.numpy(), null_value=0)

        # TODO: change from hard-coded values
        # 256 is 16x16
        idx = counts[:, :, :, 0] > (256 * cfg.label_threshold)
        acts = batch["acts"][idx].to(cfg.device)
        labels = unique[idx][:, 0]

        _, f_x, _ = sae(acts)

        pred = f_x > 0
        true = torch.from_numpy(labels == cfg.ade20k_cls).view(-1, 1).to(cfg.device)

        tp += (pred & true).sum(axis=0)
        fp += (pred & ~true).sum(axis=0)
        fn += (~pred & true).sum(axis=0)

    f1 = (2 * tp) / (2 * tp + fp + fn)
    latents = " ".join(str(i) for i in f1.topk(cfg.k).indices.tolist())

    scale_mean_flag = (
        "--data.scale-mean" if cfg.acts.scale_mean else "--data.no-scale-mean"
    )
    scale_norm_flag = (
        "--data.scale-norm" if cfg.acts.scale_norm else "--data.no-scale-norm"
    )

    print("Run this command to save best images:")
    print()
    print(
        f"  uv run python -m saev visuals --ckpt {cfg.sae_ckpt} --include-latents {latents} --data.shard-root {cfg.acts.shard_root} {scale_mean_flag} {scale_norm_flag} images:ade20k-dataset --images.root {cfg.imgs.root} --images.split {cfg.imgs.split}"
    )
    print()
    print("Be sure to add --dump-to to this command.")


@jaxtyped(typechecker=beartype.beartype)
def axis_unique(
    a: Shaped[np.ndarray, "*axes"],
    axis: int = -1,
    return_counts: bool = True,
    *,
    null_value: int = -1,
) -> (
    Shaped[np.ndarray, "*axes"]
    | tuple[Shaped[np.ndarray, "*axes"], Int[np.ndarray, "*axes"]]
):
    """
    Calculate unique values and their counts along any axis of a matrix.

    Arguments:
        a: Input array
        axis: The axis along which to find unique values.
        return_counts: If true, also return the count of each unique value

    Returns:
        unique: Array of unique values, with zeros replacing duplicates
        counts: (optional) Count of each unique value (only if return_counts=True)
    """
    assert isinstance(axis, int)

    # Move the target axis to the end for consistent processing
    a_transformed = np.moveaxis(a, axis, -1)

    # Sort along the last axis
    sorted_a = np.sort(a_transformed, axis=-1)

    # Find duplicates
    duplicates = sorted_a[..., 1:] == sorted_a[..., :-1]

    # Create output array
    unique = sorted_a.copy()
    unique[..., 1:][duplicates] = null_value

    if not return_counts:
        # Move axis back to original position
        return np.moveaxis(unique, -1, axis)

    # Calculate counts
    shape = list(a_transformed.shape)
    count_matrix = np.zeros(shape, dtype=int)

    # Process each slice along other dimensions
    for idx in np.ndindex(*shape[:-1]):
        slice_unique = unique[idx]
        idxs = np.flatnonzero(slice_unique)
        if len(idxs) > 0:
            # Calculate counts using diff for intermediate positions
            counts = np.diff(idxs)
            count_matrix[idx][idxs[:-1]] = counts
            # Handle the last unique value
            count_matrix[idx][idxs[-1]] = shape[-1] - idxs[-1]

    # Move axes back to original positions
    unique = np.moveaxis(unique, -1, axis)
    count_matrix = np.moveaxis(count_matrix, -1, axis)

    return unique, count_matrix
