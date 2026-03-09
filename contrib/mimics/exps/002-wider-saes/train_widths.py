"""Width sweep: 16K-32K latent SAEs on Cambridge butterflies (384p, v1.6).

Same config as the original 32K sweep but varying d_sae to isolate the width effect.
"""


def make_cfgs() -> list[dict]:
    batch_size: int = 1024 * 16
    n_train = 100_000_000

    # Cambridge butterfly v1.6 shards, DINOv3 ViT-L/16, 384 patches, prefer-fg
    shards_384 = "/fs/scratch/PAS2136/samuelstevens/saev/shards/a6be28a1"

    cfgs = []
    for layer in [21, 23]:
        for d_sae in [1024 * w for w in [16, 20, 24, 28, 32]]:
            for k in [16, 32, 64, 128]:
                for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
                    cfgs.append({
                        "tags": ["mimics-widths-384p-v1.6"],
                        "slurm_acct": "PAS2136",
                        "slurm_partition": "nextgen",
                        "n_hours": 8.0,
                        "lr": lr,
                        "n_lr_warmup": 500,
                        "n_sparsity_warmup": n_train // batch_size,
                        "runs_root": "/fs/ess/PAS2136/samuelstevens/saev/runs",
                        "n_train": n_train,
                        "sae": {
                            "d_model": 1024,
                            "d_sae": d_sae,
                            "normalize_w_dec": True,
                            "remove_parallel_grads": True,
                            "activation": {"top_k": k},
                            "reinit_blend": 0.8,
                        },
                        "train_data": {
                            "layer": layer,
                            "shards": shards_384,
                            "min_buffer_fill": 0.2,
                            "ignore_labels": [0],
                            "use_tmpdir": True,
                        },
                        "val_data": {
                            "layer": layer,
                            "shards": shards_384,
                            "ignore_labels": [0],
                            "use_tmpdir": True,
                        },
                    })
    return cfgs
