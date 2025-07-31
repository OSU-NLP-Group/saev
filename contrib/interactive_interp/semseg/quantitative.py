import csv
import dataclasses
import logging
import math
import os
import random
from collections.abc import Callable

import beartype
import einops
import numpy as np
import torch
from jaxtyping import Bool, Float, Int, UInt8, jaxtyped
from torch import Tensor

import saev.helpers
import saev.nn

from . import config, training

logger = logging.getLogger("contrib.semseg.quantitative")


@beartype.beartype
@torch.inference_mode
def main(cfg: config.Quantitative):
    """Main entry point for quantitative evaluation."""
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than float16 and almost as accurate as float32. This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Load models
    sae = saev.nn.load(cfg.sae_ckpt).to(cfg.device)
    clf = training.load_latest(cfg.seg_ckpt, device=cfg.device)

    # Get validation data
    dataset = training.Dataset(cfg.imgs)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=False
    )

    # For each method (random vector, random feature, etc)
    reports = []
    for fn in (eval_auto_feat,):  # (eval_rand_vec, eval_rand_feat, eval_auto_feat):
        report = fn(cfg, sae, clf, dataloader)
        reports.append(report)

    # Save results
    save(reports, os.path.join(cfg.dump_to, "results.csv"))


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ClassResults:
    """Results for a single class."""

    class_id: int
    """Numeric identifier for the class."""

    class_name: str
    """Human-readable name of the class."""

    n_orig_patches: int
    """Original patches that were this class."""

    n_changed_patches: int
    """After intervention, how many patches changed."""

    n_other_patches: int
    """Total patches that weren't this class."""

    n_other_changed: int
    """After intervention, how many of the other patches changed."""

    change_distribution: dict[int, int]
    """What classes did patches change to? Tracks how many times <value> a patch changed from self.class_id to <key>."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Report:
    """Complete results from an intervention experiment."""

    method: str
    """Which intervention method was used."""

    class_results: list[ClassResults]
    """Per-class detailed results."""

    intervention_scale: float
    """Magnitude of intervention."""

    @property
    def mean_target_change(self) -> float:
        """Percentage of target patches that changed class."""
        total_target = sum(r.n_orig_patches for r in self.class_results)
        total_changed = sum(r.n_changed_patches for r in self.class_results)
        return total_changed / total_target if total_target > 0 else 0.0

    @property
    def mean_other_change(self) -> float:
        """Percentage of non-target patches that changed class."""
        total_other = sum(r.n_other_patches for r in self.class_results)
        total_changed = sum(r.n_other_changed for r in self.class_results)
        return total_changed / total_other if total_other > 0 else 0.0

    @property
    def target_change_std(self) -> float:
        """Standard deviation of change percentage across classes."""
        per_class_target_changes = np.array([
            r.n_changed_patches / r.n_orig_patches if r.n_orig_patches > 0 else 0.0
            for r in self.class_results
        ])
        return float(np.std(per_class_target_changes))

    @property
    def other_change_std(self) -> float:
        """Standard deviation of non-target patch changes across classes."""
        per_class_other_changes = np.array([
            r.n_other_changed / r.n_other_patches if r.n_other_patches > 0 else 0.0
            for r in self.class_results
        ])
        return float(np.std(per_class_other_changes))

    def to_csv_row(self) -> dict[str, float]:
        """Convert to a row for the summary CSV."""
        return {
            "method": self.method,
            "target_change": self.mean_target_change,
            "other_change": self.mean_other_change,
            "target_std": self.target_change_std,
            "other_std": self.other_change_std,
        }


@beartype.beartype
def save(results: list[Report], fpath: str) -> None:
    """
    Save evaluation results to a CSV file.

    Args:
        results: List of Report objects containing evaluation results
        dpath: Path to save the CSV file
    """

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    columns = ["method", "target_change", "target_std", "other_change", "other_std"]

    with open(fpath, "w") as fd:
        writer = csv.DictWriter(fd, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())


@beartype.beartype
def eval_rand_vec(
    cfg: config.Quantitative,
    sae: saev.nn.SparseAutoencoder,
    clf: torch.nn.Module,
    dataloader,
) -> Report:
    """
    Evaluates the effects of adding a random unit vector to the patches.

    Args:
        cfg: Configuration for quantitative evaluation
        sae: Trained sparse autoencoder model
        clf: Trained classifier model
        dataloader: DataLoader providing batches of images

    Returns:
        Report containing intervention results, including per-class changes
    """
    torch.manual_seed(cfg.seed)

    @jaxtyped(typechecker=beartype.beartype)
    def hook(
        x_BPD: Float[Tensor, "batch patches dim"],
    ) -> Float[Tensor, "batch patches dim"]:
        """
        Adds random unit vectors to patch activations.

        Args:
            x_BPD: Activation tensor with shape (batch_size, n_patches, d_vit) where d_vit is the ViT feature dimension

        Returns:
            Modified activation tensor with random unit vectors added
        """
        batch_size, n_patches, dim = x_BPD.shape
        rand_vecs = torch.randn((batch_size, n_patches, dim), device=cfg.device)
        # Normalize each vector to unit norm along the last dimension
        rand_vecs = rand_vecs / torch.norm(rand_vecs, dim=-1, keepdim=True)

        rand_vecs = rand_vecs * cfg.scale

        x_BPD += rand_vecs
        return x_BPD

    vit = saev.activations.make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)

    hooked_vit = saev.activations.make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)
    register_hook(hooked_vit, hook, cfg.vit_layer, cfg.n_patches_per_img)

    orig_preds, mod_preds = [], []
    for batch in dataloader:
        x_BCWH = batch["image"].to(cfg.device)

        orig_acts = vit(x_BCWH)
        orig_logits = clf(orig_acts[:, 1:, :])
        orig_preds.append(argmax_logits(orig_logits).cpu())

        # Create a custom hook for this batch
        def current_hook(module, inputs, outputs):
            x = outputs[:, patches, :]
            x = hook(x, current_batch=batch)
            outputs[:, patches, :] = x
            return outputs

        # Register the hook for this batch
        patches = hooked_vit.get_patches(cfg.n_patches_per_img)
        handle = hooked_vit.get_residuals()[cfg.vit_layer - 1].register_forward_hook(
            current_hook
        )

        mod_acts = hooked_vit(x_BCWH)
        mod_logits = clf(mod_acts[:, 1:, :])
        mod_preds.append(argmax_logits(mod_logits).cpu())

        # Remove the hook after use
        handle.remove()

    # Concatenate all predictions
    orig_preds = torch.cat(orig_preds, dim=0)
    mod_preds = torch.cat(mod_preds, dim=0)

    class_results = compute_class_results(orig_preds, mod_preds)

    return Report(
        method="random-vector",
        class_results=class_results,
        intervention_scale=cfg.scale,
    )


@beartype.beartype
def eval_rand_feat(
    cfg: config.Quantitative,
    sae: saev.nn.SparseAutoencoder,
    clf: torch.nn.Module,
    dataloader,
) -> Report:
    """
    Evaluates the effects of suppressing a random SAE feature.

    Args:
        cfg: Configuration for quantitative evaluation
        sae: Trained sparse autoencoder model
        clf: Trained classifier model
        dataloader: DataLoader providing batches of images

    Returns:
        Report containing intervention results, including per-class changes
    """
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    top_values = torch.load(cfg.top_values, map_location="cpu", weights_only=True)

    @jaxtyped(typechecker=beartype.beartype)
    def hook(
        x_BPD: Float[Tensor, "batch patches dim"],
    ) -> Float[Tensor, "batch patches dim"]:
        latent = random.randrange(0, sae.cfg.d_sae)

        x_hat_BPD, f_x_BPS, _ = sae(x_BPD)

        err_BPD = x_BPD - x_hat_BPD

        value = unscaled(cfg.scale, top_values[latent].max().item())
        f_x_BPS[..., latent] = value

        # Reproduce the SAE forward pass after f_x
        mod_x_hat_BPD = sae.decode(f_x_BPS)
        mod_BPD = err_BPD + mod_x_hat_BPD
        return mod_BPD

    vit = saev.activations.make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)

    hooked_vit = saev.activations.make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)
    register_hook(hooked_vit, hook, cfg.vit_layer, cfg.n_patches_per_img)

    orig_preds, mod_preds = [], []
    for batch in dataloader:
        x_BCWH = batch["image"].to(cfg.device)

        orig_acts = vit(x_BCWH)
        orig_logits = clf(orig_acts[:, 1:, :])
        orig_preds.append(argmax_logits(orig_logits).cpu())

        mod_acts = hooked_vit(x_BCWH)
        mod_logits = clf(mod_acts[:, 1:, :])
        mod_preds.append(argmax_logits(mod_logits).cpu())

    # Concatenate all predictions
    orig_preds = torch.cat(orig_preds, dim=0)
    mod_preds = torch.cat(mod_preds, dim=0)

    class_results = compute_class_results(orig_preds, mod_preds)

    return Report(
        method="random-feature",
        class_results=class_results,
        intervention_scale=cfg.scale,
    )


@beartype.beartype
def eval_auto_feat(
    cfg: config.Quantitative,
    sae: saev.nn.SparseAutoencoder,
    clf: torch.nn.Module,
    dataloader,
) -> Report:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    top_values = torch.load(cfg.top_values, map_location="cpu", weights_only=True)

    # First, for each class, we need to pick an appropriate latent.
    # Then we can use that to lookup which latent to suppress for each patch.
    latent_lookup = get_latent_lookup(cfg, sae, dataloader)

    @jaxtyped(typechecker=beartype.beartype)
    def hook(
        x_BPD: Float[Tensor, "batch patches dim"],
        current_batch=None,
    ) -> Float[Tensor, "batch patches dim"]:
        batch_size, n_patches, _ = x_BPD.shape
        x_hat_BPD, f_x_BPS, _ = sae(x_BPD)

        err_BPD = x_BPD - x_hat_BPD

        # Get patch labels and reshape to match batch_size x patches
        patch_labels = (
            current_batch["patch_labels"]
            .view(batch_size, n_patches)
            .int()
            .to(cfg.device)
        )

        # For each patch, look up the latent to modify based on its class
        for b in range(batch_size):
            for p in range(n_patches):
                class_id = patch_labels[b, p].item()
                if class_id > 0 and class_id < 151:  # Valid class ID
                    latent = latent_lookup[class_id]
                    value = unscaled(cfg.scale, top_values[latent].max().item())
                    f_x_BPS[b, p, latent] = value

        # Reproduce the SAE forward pass after f_x
        mod_x_hat_BPD = sae.decode(f_x_BPS)
        mod_BPD = err_BPD + mod_x_hat_BPD
        return mod_BPD

    vit = saev.activations.make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)

    hooked_vit = saev.activations.make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)
    register_hook(hooked_vit, hook, cfg.vit_layer, cfg.n_patches_per_img)

    orig_preds, mod_preds = [], []
    for batch in dataloader:
        x_BCWH = batch["image"].to(cfg.device)

        orig_acts = vit(x_BCWH)
        orig_logits = clf(orig_acts[:, 1:, :])
        orig_preds.append(argmax_logits(orig_logits).cpu())

        mod_acts = hooked_vit(x_BCWH)
        mod_logits = clf(mod_acts[:, 1:, :])
        mod_preds.append(argmax_logits(mod_logits).cpu())

    # Concatenate all predictions
    orig_preds = torch.cat(orig_preds, dim=0)
    mod_preds = torch.cat(mod_preds, dim=0)

    class_results = compute_class_results(orig_preds, mod_preds)

    return Report(
        method="automatic-feature",
        class_results=class_results,
        intervention_scale=cfg.scale,
    )


@jaxtyped(typechecker=beartype.beartype)
def get_latent_lookup(
    cfg: config.Quantitative, sae: saev.nn.SparseAutoencoder, dataloader
) -> Int[Tensor, "151"]:
    """
    Dimension key:

    * B: batch dimension
    * P: patches per image
    * D: ViT hidden dimension
    * S: SAE feature dimension
    * T: threshold dimension
    * C: class dimension
    * L: layer dimension
    """
    act_mean = torch.load(cfg.act_mean, weights_only=True, map_location=cfg.device)

    latent_lookup = torch.zeros((151,), dtype=int)
    thresholds_T = torch.tensor([0, 0.1, 0.3, 1.0], device=cfg.device)

    vit = saev.activations.make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)
    recorded_vit = saev.activations.RecordedVisionTransformer(
        vit, cfg.n_patches_per_img, cfg.cls_token, [cfg.vit_layer - 1]
    )

    tp_counts_CTS = torch.zeros(
        (151, len(thresholds_T), sae.cfg.d_sae), dtype=torch.int32, device=cfg.device
    )
    fp_counts_CTS = torch.zeros(
        (151, len(thresholds_T), sae.cfg.d_sae), dtype=torch.int32, device=cfg.device
    )
    fn_counts_CTS = torch.zeros(
        (151, len(thresholds_T), sae.cfg.d_sae), dtype=torch.int32, device=cfg.device
    )

    # Load sparsity and set up frequency mask.
    sparsity_S = torch.load(cfg.sparsity, weights_only=True, map_location="cpu")
    mask_S = (sparsity_S < cfg.max_freq).to(cfg.device)

    for batch in saev.helpers.progress(dataloader, every=1):
        x_BCWH = batch["image"].to(cfg.device)
        _, vit_acts_BLPD = recorded_vit(x_BCWH)

        # Normalize activations
        vit_acts_BPD = (
            vit_acts_BLPD[:, 0, 1:, :].to(cfg.device).clamp(-1e-5, 1e5) - act_mean
        ) / cfg.act_norm
        _, sae_acts_BPS, _ = sae(vit_acts_BPD)

        # Merge batch and patches dimensions for easier processing
        sae_acts_BS = einops.rearrange(
            sae_acts_BPS, "batch patches d_sae -> (batch patches) d_sae"
        )

        pw, ph = cfg.patch_size_px
        patch_labels_B = batch["patch_labels"].to(cfg.device).reshape(-1)
        pixel_labels_BP = einops.rearrange(
            batch["pixel_labels"].to(cfg.device),
            "batch (w pw) (h ph) -> (batch w h) (pw ph)",
            pw=pw,
            ph=ph,
        )

        # Create mask for patches that meet the threshold
        valid_mask = get_patch_mask(pixel_labels_BP, cfg.label_threshold)

        # Filter patch labels to only include those meeting the threshold
        patch_labels_B = patch_labels_B[valid_mask]
        sae_acts_BS = sae_acts_BS[valid_mask]

        unique_classes = torch.unique(patch_labels_B)

        for class_id in unique_classes:
            if class_id == 0:  # Skip background/null class if needed
                continue

            class_mask_B = patch_labels_B == class_id

            # Skip if no patches of this class
            if not torch.any(class_mask_B):
                continue

            # Process all thresholds at once
            # Create binary activation masks for all thresholds
            binary_activations_TBS = (
                sae_acts_BS[None, :, :] > thresholds_T[:, None, None]
            )

            # Compute TP, FP, FN for all thresholds and features at once
            # Each has shape: [thresholds, features] = [T, S]

            # True Positives: activation is 1 AND patch is this class
            tp_TS = torch.sum(
                binary_activations_TBS & class_mask_B[None, :, None], dim=1
            )
            tp_counts_CTS[class_id] += tp_TS

            # False Positives: activation is 1 BUT patch is NOT this class
            fp_TS = torch.sum(
                binary_activations_TBS & ~class_mask_B[None, :, None], dim=1
            )
            fp_counts_CTS[class_id] += fp_TS

            # False Negatives: activation is 0 BUT patch IS this class
            fn_TS = torch.sum(
                (~binary_activations_TBS) & class_mask_B[None, :, None], dim=1
            )
            fn_counts_CTS[class_id] += fn_TS

    class_lookup = {}
    with open(os.path.join(cfg.imgs.root, "objectInfo150.txt")) as fd:
        for row in csv.DictReader(fd, delimiter="\t"):
            class_lookup[int(row["Idx"])] = row["Name"]

    for class_id, class_name in saev.helpers.progress(class_lookup.items()):
        # Compute F1 scores: 2*TP / (2*TP + FP + FN)
        tp_TS = tp_counts_CTS[class_id]
        fp_TS = fp_counts_CTS[class_id]
        fn_TS = fn_counts_CTS[class_id]

        # Add small epsilon to avoid division by zero
        f1_TS = (2 * tp_TS) / (2 * tp_TS + fp_TS + fn_TS + 1e-10)

        # Calculate precision and recall for each threshold
        # precision_TS = tp_TS / (tp_TS + fp_TS + 1e-10)  # TP / (TP + FP)
        # recall_TS = tp_TS / (tp_TS + fn_TS + 1e-10)  # TP / (TP + FN)

        f1_S, best_thresh_i_S = f1_TS.max(dim=0)
        f1_S = torch.where(mask_S, f1_S, torch.tensor(-1.0, device=f1_S.device))
        best_thresholds_S = thresholds_T[best_thresh_i_S]

        # Get top performing features
        topk_scores, topk_latents = torch.topk(f1_S, k=cfg.top_k)

        print(f"Top {cfg.top_k} features for {class_name}:")
        for score, latent in zip(topk_scores, topk_latents):
            print(f"{latent:>6} >{best_thresholds_S[latent]}: {score:.3f}")

        latent_lookup[class_id] = topk_latents[0].item()

    return latent_lookup


@jaxtyped(typechecker=beartype.beartype)
def get_patch_mask(
    pixel_labels_NP: UInt8[Tensor, "n patch_px"], threshold: float
) -> Bool[Tensor, " n"]:
    """
    Create a mask for patches where at least threshold proportion of pixels have the same label.

    Args:
        pixel_labels_NP: Tensor of shape [n, patch_pixels] with pixel labels
        threshold: Minimum proportion of pixels with same label

    Returns:
        Tensor of shape [n] with True for patches that pass the threshold
    """
    # For each patch, count occurrences of each unique label
    _, patch_pixels = pixel_labels_NP.shape

    mode_N = pixel_labels_NP.mode(axis=-1).values

    # Count occurrences of the mode value in each patch using vectorized operations
    counts_N = (pixel_labels_NP == mode_N[:, None]).sum(dim=1)

    # Calculate proportion and create mask
    mask_N = (counts_N / patch_pixels) >= threshold
    return mask_N


@jaxtyped(typechecker=beartype.beartype)
def register_hook(
    vit: torch.nn.Module,
    hook: Callable[[Float[Tensor, "..."]], Float[Tensor, "..."]],
    layer: int,
    n_patches_per_img: int,
):
    patches = vit.get_patches(n_patches_per_img)

    @jaxtyped(typechecker=beartype.beartype)
    def _hook(
        block: torch.nn.Module,
        inputs: tuple,
        outputs: Float[Tensor, "batch patches dim"],
    ) -> Float[Tensor, "batch patches dim"]:
        x = outputs[:, patches, :]
        x = hook(x)
        outputs[:, patches, :] = x
        return outputs

    return vit.get_residuals()[layer].register_forward_hook(_hook)


@jaxtyped(typechecker=beartype.beartype)
def compute_class_results(
    orig_preds: Int[Tensor, "n_imgs patches"], mod_preds: Int[Tensor, "n_imgs patches"]
) -> list[ClassResults]:
    class_results = []
    for class_id in range(1, 151):
        # Count original patches of this class
        orig_mask = orig_preds == class_id
        n_orig = orig_mask.sum().item()

        # Count how many changed
        changed_mask = (mod_preds != orig_preds) & orig_mask
        n_changed = changed_mask.sum().item()

        # Count changes in other patches
        other_mask = ~orig_mask
        n_other = other_mask.sum().item()
        n_other_changed = ((mod_preds != orig_preds) & other_mask).sum().item()

        # Track what classes patches changed to
        changes = {}
        for new_class in range(1, 151):
            if new_class != class_id:
                count = ((mod_preds == new_class) & changed_mask).sum().item()
                if count > 0:
                    changes[new_class] = count

        class_results.append(
            ClassResults(
                class_id=class_id,
                class_name="TODO",  # class_names[class_id],
                n_orig_patches=n_orig,
                n_changed_patches=n_changed,
                n_other_patches=n_other,
                n_other_changed=n_other_changed,
                change_distribution=changes,
            )
        )

    return class_results


@jaxtyped(typechecker=beartype.beartype)
def argmax_logits(
    logits_BPC: Float[Tensor, "batch patches channels_with_null"],
) -> Int[Tensor, "batch patches"]:
    return logits_BPC[:, :, 1:].argmax(axis=-1) + 1


@jaxtyped(typechecker=beartype.beartype)
def unscaled(
    x: Float[Tensor, "*batch"], max_obs: float | int
) -> Float[Tensor, "*batch"]:
    """Scale from [-10, 10] to [10 * -max_obs, 10 * max_obs]."""
    return map_range(x, (-10.0, 10.0), (-10.0 * max_obs, 10.0 * max_obs))


@jaxtyped(typechecker=beartype.beartype)
def map_range(
    x: Float[Tensor, "*batch"],
    domain: tuple[float | int, float | int],
    range: tuple[float | int, float | int],
) -> Float[Tensor, "*batch"]:
    a, b = domain
    c, d = range
    if not (a <= x <= b):
        raise ValueError(f"x={x:.3f} must be in {[a, b]}.")
    return c + (x - a) * (d - c) / (b - a)


@jaxtyped(typechecker=beartype.beartype)
def get_patch_i(
    i: Int[Tensor, "batch width height"], n_patches_per_img: int
) -> Int[Tensor, "batch width height"]:
    w = h = int(math.sqrt(n_patches_per_img))

    i = i * n_patches_per_img
    i = i + torch.arange(n_patches_per_img).view(w, h)
    return i
