import logging
import os.path
import random
import typing

import beartype
import einops
import saev.activations
import torch
import tyro

import saev.nn

from . import config

logger = logging.getLogger("contrib.semprobe")


@beartype.beartype
@torch.inference_mode
def score(cfg: typing.Annotated[config.Score, tyro.conf.arg(name="")]):
    sae = saev.nn.load(cfg.sae_ckpt)
    sae = sae.to(cfg.device)
    logger.info("Loaded SAE.")

    acts_dataset = saev.activations.Dataset(cfg.acts)
    imgs_dataset = saev.activations.ImageFolder(cfg.imgs.root)

    assert len(acts_dataset) // acts_dataset.metadata.n_patches_per_img == len(
        imgs_dataset
    )

    batch_size = (
        cfg.batch_size
        // acts_dataset.metadata.n_patches_per_img
        * acts_dataset.metadata.n_patches_per_img
    )
    n_imgs_per_batch = batch_size // acts_dataset.metadata.n_patches_per_img

    dataloader = torch.utils.data.DataLoader(
        acts_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
        drop_last=False,
    )
    logger.info("Loaded data.")

    pred_labels_SN = torch.zeros((sae.cfg.d_sae, len(imgs_dataset)))
    true_labels_N = torch.zeros((len(imgs_dataset)))
    task_N = torch.zeros((len(imgs_dataset)))
    task_lookup = {}

    all_latents = []

    for batch in dataloader:
        vit_acts_BD = batch["act"].to(cfg.device)
        i_im = torch.sort(torch.unique(batch["image_i"])).values
        assert len(i_im) <= n_imgs_per_batch
        _, sae_acts_BS, _ = sae(vit_acts_BD)
        sae_acts_SB = einops.rearrange(sae_acts_BS, "batch d_sae -> d_sae batch")
        sae_acts_SIP = sae_acts_SB.view(
            sae.cfg.d_sae, len(i_im), acts_dataset.metadata.n_patches_per_img
        )
        sae_acts_SI = einops.reduce(
            sae_acts_SIP, "d_sae n_img n_patch -> d_sae n_img", "sum"
        )

        # Predictions for each latent is 1 for sae_acts_SI[latent] > threshold, 0 otherwise.
        for j, i in enumerate(i_im.tolist()):
            task, label = imgs_dataset[i]["label"].split("-")
            if task not in task_lookup:
                task_lookup[task] = len(task_lookup)

            task_N[i] = task_lookup[task]
            true_labels_N[i] = 1.0 if label == "positive" else 0.0
            pred_labels_SN[:, i] = (sae_acts_SI[:, j] > cfg.threshold).cpu().float()

    logger.info("Made %d predictions.", len(imgs_dataset))
    for task_name, task_value in task_lookup.items():
        true_pos_S = einops.reduce(
            (pred_labels_SN == true_labels_N)
            & (true_labels_N == 1)
            & (task_N == task_value),
            "d_sae n_image -> d_sae",
            "sum",
        )
        false_pos_S = einops.reduce(
            (pred_labels_SN != true_labels_N)
            & (true_labels_N == 0)
            & (task_N == task_value),
            "d_sae n_image -> d_sae",
            "sum",
        )
        false_neg_S = einops.reduce(
            (pred_labels_SN != true_labels_N)
            & (true_labels_N == 1)
            & (task_N == task_value),
            "d_sae n_image -> d_sae",
            "sum",
        )
        f1_S = (2 * true_pos_S) / (2 * true_pos_S + false_pos_S + false_neg_S)
        # TODO
        # 2. Save visuals for these example images.

        # Get top performing features
        topk_scores, topk_indices = torch.topk(f1_S, k=cfg.top_k)

        print(f"Top {cfg.top_k} features for {task_name}:")
        for score, i in zip(topk_scores, topk_indices):
            print(f"{i:>6}: {score:.3f}")

        print(f"Manually included features for {task_name}:")
        for i in cfg.include_latents:
            print(f"{i:>6}: {f1_S[i]:.3f}")

        all_latents.extend(topk_indices.tolist())
        all_latents.extend(cfg.include_latents)

    # Construct command to visualize top features
    latents_str = " ".join(str(i) for i in all_latents)
    cmd = f"""
uv run python -m saev visuals \\
    --ckpt {cfg.sae_ckpt} \\
    --include-latents {latents_str} \\
    --log-freq-range -4.0 -1.0 \\
    --log-value-range -1.0 1.0 \\
    --n-latents {len(topk_indices) + 10} \\
    --dump-to $DUMP_TO
""".strip()
    print(f"\nTo visualize, run:\n\n{cmd}")
    print(
        "\nNote that you need to update/add:\n* $DUMP_TO\n* --data.shard-root\n* images:*\n*--images."
    )


@beartype.beartype
def negatives(cfg: typing.Annotated[config.Negatives, tyro.conf.arg(name="")]):
    """
    Sample negative images for each class.
    """
    imgs = saev.activations.get_dataset(cfg.imgs, img_transform=None)

    indices = list(range(len(imgs)))
    random.seed(cfg.seed)

    for cls in cfg.classes:
        random.shuffle(indices)
        dpath = os.path.join(cfg.dump_to, f"{cls}-negative")
        os.makedirs(dpath, exist_ok=True)
        n_saved = 0
        for i in indices:
            if i in cfg.skip:
                continue

            sample = imgs[i]
            fpath = os.path.join(dpath, f"example_{cls}_{i}.png")
            sample["image"].save(fpath)
            n_saved += 1

            if n_saved >= cfg.n_imgs:
                break


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "score": score,
        "negatives": negatives,
    })
