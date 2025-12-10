import dataclasses
import pathlib

import pytest
import torch

import saev.data
import saev.data.datasets
import saev.framework.train
import saev.nn
from saev.nn.modeling import BatchTopK, Relu, TopK


def test_split_cfgs_separates_different_shards():
    """Configs with different train_data.shards must be in separate groups."""
    base = saev.framework.train.Config(
        train_data=saev.data.ShuffledConfig(shards=pathlib.Path("/path/to/shards_a")),
        val_data=saev.data.ShuffledConfig(shards=pathlib.Path("/path/to/shards_a")),
    )
    cfg_a1 = dataclasses.replace(base, lr=0.001)
    cfg_a2 = dataclasses.replace(base, lr=0.002)
    cfg_b = dataclasses.replace(
        base,
        train_data=saev.data.ShuffledConfig(shards=pathlib.Path("/path/to/shards_b")),
        val_data=saev.data.ShuffledConfig(shards=pathlib.Path("/path/to/shards_b")),
    )

    groups = saev.framework.train.split_cfgs([cfg_a1, cfg_a2, cfg_b])

    # Should have 2 groups: one for shards_a, one for shards_b
    assert len(groups) == 2

    # Find which group has which configs
    group_shards = []
    for group in groups:
        shards = {cfg.train_data.shards for cfg in group}
        assert len(shards) == 1, "All configs in a group must have same shards"
        group_shards.append(shards.pop())

    assert set(group_shards) == {
        pathlib.Path("/path/to/shards_a"),
        pathlib.Path("/path/to/shards_b"),
    }


def test_split_cfgs_groups_same_shards():
    """Configs with same train_data.shards but different hyperparams are grouped."""
    shards_path = pathlib.Path("/path/to/shards")
    base = saev.framework.train.Config(
        train_data=saev.data.ShuffledConfig(shards=shards_path),
        val_data=saev.data.ShuffledConfig(shards=shards_path),
    )
    cfg1 = dataclasses.replace(base, lr=0.001)
    cfg2 = dataclasses.replace(base, lr=0.002)
    cfg3 = dataclasses.replace(base, lr=0.003)

    groups = saev.framework.train.split_cfgs([cfg1, cfg2, cfg3])

    # All should be in one group since they share shards
    assert len(groups) == 1
    assert len(groups[0]) == 3


# Configs with placeholder paths - patched at runtime via dataclasses.replace().
# Add new configs here to test different combinations during development.
TRAIN_CONFIGS = {
    "topk-adam": saev.framework.train.Config(
        n_train=40,
        n_val=40,
        device="cpu",
        sae=saev.nn.SparseAutoencoderConfig(activation=TopK(top_k=8), reinit_blend=0.0),
    ),
    "batchtopk-adam": saev.framework.train.Config(
        n_train=40,
        n_val=40,
        device="cpu",
        sae=saev.nn.SparseAutoencoderConfig(
            activation=BatchTopK(top_k=8), reinit_blend=0.0
        ),
    ),
    "relu-adam": saev.framework.train.Config(
        n_train=40,
        n_val=40,
        device="cpu",
        sae=saev.nn.SparseAutoencoderConfig(activation=Relu(), reinit_blend=0.0),
    ),
    "topk-muon": saev.framework.train.Config(
        n_train=40,
        n_val=40,
        device="cpu",
        optim="muon",
        sae=saev.nn.SparseAutoencoderConfig(activation=TopK(top_k=8), reinit_blend=0.0),
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("cfg_id", TRAIN_CONFIGS.keys())
def test_train_and_eval(cfg_id: str):
    """Integration test: train and eval for a few batches.

    Add configs to TRAIN_CONFIGS to test different combinations.
    """
    base_cfg = TRAIN_CONFIGS[cfg_id]
    if base_cfg.device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    with (
        pytest.helpers.tmp_shards_root() as shards_root,
        pytest.helpers.tmp_runs_root() as runs_root,
    ):
        shards_dir = pytest.helpers.write_shards(
            shards_root, data=saev.data.datasets.FakeImg(n_examples=64)
        )
        md = saev.data.Metadata.load(shards_dir)

        # Patch runtime values into config
        cfg = dataclasses.replace(
            base_cfg,
            train_data=saev.data.ShuffledConfig(
                shards=shards_dir, layer=0, batch_size=4
            ),
            val_data=saev.data.ShuffledConfig(shards=shards_dir, layer=0, batch_size=4),
            runs_root=runs_root,
            track=False,
            sae=dataclasses.replace(
                base_cfg.sae, d_model=md.d_model, d_sae=md.d_model * 4
            ),
        )
        run_ids = saev.framework.train.worker_fn([cfg])

        assert len(run_ids) == 1
        run = saev.disk.Run(runs_root / run_ids[0])
        assert run.ckpt.exists()
        assert run.ckpt.stat().st_size > 0

        sae = saev.nn.load(run.ckpt)
        assert sae.cfg.d_model == md.d_model
