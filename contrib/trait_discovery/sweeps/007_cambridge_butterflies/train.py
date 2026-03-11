def make_cfgs() -> list[dict]:
    batch_size: int = 1024 * 16
    n_train = 100_000_000

    # Cambridge butterfly shards with different patch counts (DINOv3 ViT-L-16, layers 21 and 23, prefer-fg)
    shards_v10 = {
        256: "/fs/scratch/PAS2136/samuelstevens/saev/shards/592615d0",
        384: "/fs/scratch/PAS2136/samuelstevens/saev/shards/bf629dcb",
        512: "/fs/scratch/PAS2136/samuelstevens/saev/shards/efbdb4d2",
        640: "/fs/scratch/PAS2136/samuelstevens/saev/shards/69e6d6fd",
    }
    # v1.2 shards
    shards_v12 = {
        384: "/fs/scratch/PAS2136/samuelstevens/saev/shards/7c2ba646",
        640: "/fs/scratch/PAS2136/samuelstevens/saev/shards/71ba8292",
    }
    # v1.6 shards (cambridge-segfolder-v1.6)
    shards_v16 = {
        384: "/fs/scratch/PAS2136/samuelstevens/saev/shards/a6be28a1",
        640: "/fs/scratch/PAS2136/samuelstevens/saev/shards/79239bdd",
    }

    shards_by_version = {"v1.0": shards_v10, "v1.2": shards_v12, "v1.6": shards_v16}

    version = "v1.6"

    cfgs = []
    for n_patches, shards_dpath in shards_by_version[version].items():
        for layer in [21, 23]:
            for k in [16, 32, 64, 128, 256]:
                for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
                    cfgs.append({
                        "tags": [f"cambridge-{n_patches}p-{version}"],
                        "slurm_acct": "PAS2136",
                        "slurm_partition": "nextgen",
                        "n_hours": 12.0,
                        "lr": lr,
                        "n_lr_warmup": 500,
                        "n_sparsity_warmup": n_train // batch_size,
                        "runs_root": "/fs/ess/PAS2136/samuelstevens/saev/runs",
                        "n_train": n_train,
                        "sae": {
                            "d_model": 1024,
                            "d_sae": 1024 * 16,
                            "normalize_w_dec": True,
                            "remove_parallel_grads": True,
                            "activation": {"top_k": k},
                            "reinit_blend": 0.8,
                        },
                        "train_data": {
                            "layer": layer,
                            "shards": shards_dpath,
                            "min_buffer_fill": 0.2,
                            "ignore_labels": [0],
                            "use_tmpdir": True,
                        },
                        "val_data": {
                            "layer": layer,
                            "shards": shards_dpath,
                            "ignore_labels": [0],
                            "use_tmpdir": True,
                        },
                    })
    return cfgs
