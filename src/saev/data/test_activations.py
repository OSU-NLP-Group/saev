"""
Test that the cached activations are actually correct.
These tests are quite slow
"""

import tempfile

import pytest
import torch

from . import activations, config


@pytest.mark.slow
def test_dataloader_batches():
    cfg = config.Activations(
        vit_ckpt="ViT-B-32/openai",
        d_vit=768,
        vit_layers=[-2, -1],
        n_patches_per_img=49,
        vit_batch_size=8,
    )
    dataloader = activations.get_dataloader(
        cfg, img_transform=activations.make_img_transform(cfg.vit_family, cfg.vit_ckpt)
    )
    batch = next(iter(dataloader))

    assert isinstance(batch, dict)
    assert "image" in batch
    assert "index" in batch

    torch.testing.assert_close(batch["index"], torch.arange(8))
    assert batch["image"].shape == (8, 3, 224, 224)


@pytest.mark.slow
def test_shard_writer_and_dataset_e2e():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = config.Activations(
            vit_family="dinov2",
            vit_ckpt="dinov2_vits14_reg",
            d_vit=384,
            n_patches_per_img=256,
            vit_layers=[-2, -1],
            vit_batch_size=8,
            n_workers=8,
            dump_to=tmpdir,
        )
        vit = activations.make_vit(cfg.vit_family, cfg.vit_ckpt)
        vit = activations.RecordedVisionTransformer(
            vit, cfg.n_patches_per_img, cfg.cls_token, cfg.vit_layers
        )
        dataloader = activations.get_dataloader(
            cfg,
            img_transform=activations.make_img_transform(cfg.vit_family, cfg.vit_ckpt),
        )
        writer = activations.ShardWriter(cfg)
        dataset = activations.Dataset(
            config.DataLoad(
                shard_root=activations.get_acts_dir(cfg),
                patches="cls",
                layer=-1,
                scale_mean=False,
                scale_norm=False,
            )
        )

        i = 0
        for b, batch in zip(range(10), dataloader):
            # Don't care about the forward pass.
            out, cache = vit(batch["image"])
            del out

            writer[i : i + len(cache)] = cache
            i += len(cache)
            assert cache.shape == (cfg.vit_batch_size, len(cfg.vit_layers), 257, 384)

            acts = [dataset[i.item()]["act"] for i in batch["index"]]
            from_dataset = torch.stack(acts)
            torch.testing.assert_close(cache[:, -1, 0], from_dataset)
            print(f"Batch {b} matched.")
