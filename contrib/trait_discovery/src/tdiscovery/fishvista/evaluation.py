import dataclasses
import json
import logging
import os
import os.path
import time
import typing as tp

import beartype
import einops
import numpy as np
import torch
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

import saev.data
import saev.data.datasets
import saev.data.transforms
import saev.helpers

from .. import baselines, metrics, saes
from . import utils

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for unified trait discovery evaluation pipeline on FishVista.
    """

    # Method configuration
    method: tp.Literal["random", "pca", "kmeans", "sae"] = "random"
    """Which method we are evaluating."""
    n_prototypes: int = 1024 * 32
    """Number of prototypes/components."""

    sae_ckpt: str = ""
    """Path to pre-trained SAE checkpoint (only for method='sae')."""

    # Data configuration
    train_acts: saev.data.ShuffledConfig = saev.data.ShuffledConfig()
    """Train activations shard configuration."""
    test_acts: saev.data.OrderedConfig = saev.data.OrderedConfig()
    """Test activations shard configuration."""
    layer: int = 23
    """ViT layer."""
    imgs: saev.data.datasets.SegFolder = saev.data.datasets.SegFolder()

    # Evaluation configuration
    n_train: int = -1
    """Number of images to use to pick best prototypes. Less than 0 indicates all images."""

    # Output configuration
    dump_to: str = os.path.join(".", "results")
    """Where to save evaluation results."""
    output_format: tp.Literal["json", "csv", "both"] = "json"
    """Output format for results."""

    seed: int = 42
    """Random seed."""
    device: str = "cuda"
    """Device to use for computation."""

    # Slurm configuration
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 6.0
    """Slurm job length in hours."""
    log_to: str = os.path.join(".", "logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
def get_scorer(cfg: Config) -> baselines.Scorer:
    """Initialize the scorer based on the method."""
    d_vit = saev.data.Metadata.load(cfg.train_acts.shard_root).d_vit

    if cfg.method == "random":
        return baselines.RandomVectors(
            n_prototypes=cfg.n_prototypes, d=d_vit, seed=cfg.seed
        )
    elif cfg.method == "kmeans":
        return baselines.KMeans(
            n_means=cfg.n_prototypes, d=d_vit, seed=cfg.seed, device=cfg.device
        )
    elif cfg.method == "pca":
        return baselines.PCA(
            n_components=cfg.n_prototypes, d=d_vit, seed=cfg.seed, device=cfg.device
        )
    elif cfg.method == "sae":
        if not cfg.sae_ckpt:
            raise ValueError("sae_ckpt must be provided for method='sae'")
        return saes.SparseAutoencoderScorer(cfg.sae_ckpt)
    else:
        raise RuntimeError(f"Method '{cfg.method}' not implemented.")


@jaxtyped(typechecker=beartype.beartype)
def make_keep_mask(n_total: int, n_keep: int, *, seed: int) -> Bool[Tensor, " n_total"]:
    if n_keep < 0:
        return torch.ones(n_total, dtype=torch.bool)

    if n_keep >= n_total:
        return torch.ones(n_total, dtype=torch.bool)

    g = torch.Generator(device="cpu").manual_seed(seed)
    keep_idx_k = torch.randperm(n_total, generator=g)[:n_keep]
    keep_mask_n = torch.zeros(n_total, dtype=torch.bool)
    keep_mask_n[keep_idx_k] = True
    return keep_mask_n


@torch.no_grad()
@jaxtyped(typechecker=beartype.beartype)
def compute_patch_scores(
    cfg: Config,
    scorer: baselines.Scorer,
    acts_dl: saev.data.OrderedDataLoader,
    imgs_dl: torch.utils.data.DataLoader,
    *,
    n: int = -1,
) -> tuple[Float[Tensor, "n k"], Int[Tensor, " n"]]:
    """
    Compute prototype scores for all patches in the dataset.

    Returns:
        scores: Prototype scores for each patch
        labels: Ground truth segmentation labels for each patch
    """
    n_patches = acts_dl.metadata.n_imgs * acts_dl.metadata.n_patches_per_img
    keep_mask_n = make_keep_mask(n_patches, n, seed=cfg.seed)
    if n < 0:
        scores_nk = torch.full((n_patches, scorer.n_prototypes), -torch.inf)
        labels_n = torch.full((n_patches,), -1, dtype=int)
    else:
        scores_nk = torch.full((n, scorer.n_prototypes), -torch.inf)
        labels_n = torch.full((n,), -1, dtype=int)

    assert len(labels_n) == keep_mask_n.int().sum().item()

    scorer = scorer.to(cfg.device)
    scorer.eval()

    n_kept_total = 0
    for acts, imgs in zip(saev.helpers.progress(acts_dl, desc="scoring"), imgs_dl):
        # Check that our two dataloaders are synced up.
        image_i_b = imgs["index"].repeat_interleave(acts_dl.metadata.n_patches_per_img)
        assert (image_i_b == acts["image_i"]).all()

        patch_i_b = torch.arange(acts_dl.metadata.n_patches_per_img).repeat(
            len(imgs["index"])
        )
        assert (patch_i_b == acts["patch_i"]).all()

        # Check if any of these patches are kept.
        i_b = image_i_b * acts_dl.metadata.n_patches_per_img + patch_i_b
        keep_b = keep_mask_n[i_b]
        if not keep_b.any():
            continue

        # Do inference.
        acts_bd = acts["act"][keep_b].to(cfg.device)
        # Open a fresh jaxtyping dynamic shape context so B rebinds per batch; without this, the outer @jaxtyped pins B from earlier iterations (e.g., 16000) and a smaller final batch (e.g., 10240) triggers a shape violation.
        with jaxtyped("context"):
            scores_bk = scorer(acts_bd)

        # Update scores_nk and labels_n in the correct indices.
        n_kept_batch = keep_b.sum().item()
        scores_nk[n_kept_total : n_kept_total + n_kept_batch] = scores_bk
        labels_n[n_kept_total : n_kept_total + n_kept_batch] = einops.rearrange(
            imgs["patch_labels"], "imgs patches -> (imgs patches)"
        )[keep_b]
        n_kept_total += n_kept_batch

    return scores_nk, labels_n


@jaxtyped(typechecker=beartype.beartype)
def get_best_aps(
    train_scores_nk: Float[Tensor, "n_train k"],
    train_labels_n: Int[Tensor, " n_train"],
    test_scores_nk: Float[Tensor, "n_test k"],
    test_labels_n: Int[Tensor, " n_test"],
    cfg: Config,
    *,
    bsz: int = 512,
) -> list[tuple[int, float]]:
    # Pick best prototypes
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    n, k = train_scores_nk.shape
    best_ap_c = torch.full((utils.n_classes,), 0, dtype=torch.float32)
    best_idx_c = torch.randint(0, k, (utils.n_classes,), generator=g, dtype=torch.int64)

    # I want the best prototype (measured via AP score) for each of the 10 fishvista classes.
    train_labels_nc = torch.nn.functional.one_hot(
        train_labels_n, num_classes=utils.n_classes
    ).bool()
    for start, end in saev.helpers.batched_idx(k, bsz):
        ap_bc = metrics.calc_avg_prec(train_scores_nk[:, start:end], train_labels_nc)
        max_in_batch, row_idx = ap_bc.max(dim=0)
        update_mask = max_in_batch > best_ap_c
        best_ap_c[update_mask] = max_in_batch[update_mask]
        best_idx_c[update_mask] = start + row_idx[update_mask]

    # Now that I have the best protoypes for each class, as measured on the training set, we need to get AP for each class, using the test set.
    test_labels_nc = torch.nn.functional.one_hot(
        test_labels_n, num_classes=utils.n_classes
    ).bool()
    val_ap_c = torch.full((utils.n_classes,), 0.0, dtype=torch.float32)
    for start, end in saev.helpers.batched_idx(k, bsz):
        ap_bc = metrics.calc_avg_prec(test_scores_nk[:, start:end], test_labels_nc)
        max_in_batch, row_idx = ap_bc.max(dim=0)
        update_mask = max_in_batch > val_ap_c
        val_ap_c[update_mask] = max_in_batch[update_mask]

    return list(zip(best_idx_c.tolist(), val_ap_c.tolist()))


@beartype.beartype
def evaluate(cfg: Config):
    """Main worker function for training and evaluation."""
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("worker")

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load scorer
    scorer = get_scorer(cfg)

    # Update layer field with the given layer.
    # TODO: Why do we need a cfg.layer? To prevent the grid() from making a job where the train layer != test layer.
    cfg = dataclasses.replace(
        cfg,
        train_acts=dataclasses.replace(cfg.train_acts, layer=cfg.layer),
        test_acts=dataclasses.replace(cfg.test_acts, layer=cfg.layer),
    )
    train_imgs_cfg = dataclasses.replace(cfg.imgs, split="training")
    test_imgs_cfg = dataclasses.replace(cfg.imgs, split="validation")

    logger.info("Training '%s' with %d prototypes", cfg.method, cfg.n_prototypes)
    train_acts_md = saev.data.Metadata.load(cfg.train_acts.shard_root)
    acts_batch_size = (
        cfg.train_acts.batch_size // train_acts_md.n_patches_per_img
    ) * train_acts_md.n_patches_per_img
    imgs_batch_size = cfg.train_acts.batch_size // train_acts_md.n_patches_per_img
    train_acts_sh_dl = saev.data.ShuffledDataLoader(
        dataclasses.replace(cfg.train_acts, batch_size=acts_batch_size)
    )
    train_acts_dl = saev.data.OrderedDataLoader(
        saev.data.make_ordered_config(cfg.train_acts, batch_size=acts_batch_size)
    )
    train_imgs_dl = torch.utils.data.DataLoader(
        utils.ImageDataset(train_imgs_cfg, "no-bg"),
        batch_size=imgs_batch_size,
        num_workers=4,
        shuffle=False,
    )
    scorer.fit(train_acts_sh_dl)

    # We need the activations and the images to be be coordinated. This requires a bit of fiddling with respect to batch sizes.
    test_acts_md = saev.data.Metadata.load(cfg.test_acts.shard_root)
    acts_batch_size = (
        cfg.test_acts.batch_size // test_acts_md.n_patches_per_img
    ) * test_acts_md.n_patches_per_img
    imgs_batch_size = cfg.test_acts.batch_size // test_acts_md.n_patches_per_img
    test_acts_dl = saev.data.OrderedDataLoader(
        dataclasses.replace(cfg.test_acts, batch_size=acts_batch_size)
    )
    test_imgs_dl = torch.utils.data.DataLoader(
        utils.ImageDataset(test_imgs_cfg, "no-bg"),
        batch_size=imgs_batch_size,
        num_workers=4,
        shuffle=False,
    )

    # Compute scores for all test patches
    logger.info("Scoring test split.")
    test_scores, test_labels = compute_patch_scores(
        cfg, scorer, test_acts_dl, test_imgs_dl
    )
    # Also compute scores for a subset of the train set to determine prototype assignments
    logger.info("Scoring train split.")
    train_scores, train_labels = compute_patch_scores(
        cfg, scorer, train_acts_dl, train_imgs_dl, n=cfg.n_train
    )

    aps = get_best_aps(train_scores, train_labels, test_scores, test_labels, cfg)

    results = []
    for i, (prototype_i, ap) in enumerate(aps):
        result = utils.Result(
            method=cfg.method,
            n_prototypes=cfg.n_prototypes,
            n_train=cfg.n_train,
            seed=cfg.seed,
            class_idx=i,
            average_precision=ap,
            best_prototype_idx=prototype_i,
            vit_family=test_acts_md.vit_family,
            vit_ckpt=test_acts_md.vit_ckpt,
            layer=cfg.test_acts.layer,
            d_vit=test_acts_md.d_vit,
            extra={"sae_ckpt": cfg.sae_ckpt},
        )
        results.append(result)

    # Save results
    os.makedirs(cfg.dump_to, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"fishvista__{cfg.method}__n-prototypes_{cfg.n_prototypes}__n-train_{cfg.n_train}__seed_{cfg.seed}__vit-ckpt_{saev.helpers.fssafe(test_acts_md.vit_ckpt)}__layer_{cfg.test_acts.layer}_{timestamp}.json"

    json_path = os.path.join(cfg.dump_to, fname)
    with open(json_path, "w") as fd:
        json.dump([dataclasses.asdict(r) for r in results], fd)
    logger.info("Saved JSON results to %s", json_path)
