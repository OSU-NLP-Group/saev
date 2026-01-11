def make_cfgs() -> list[dict]:
    batch_size: int = 1024 * 16
    n_train = 100_000_000

    # Cambridge butterfly shards with different patch counts (DINOv3 ViT-L-16, layers 21 and 23, prefer-fg)
    shards = {
        256: "/fs/scratch/PAS2136/samuelstevens/saev/shards/592615d0",
        384: "/fs/scratch/PAS2136/samuelstevens/saev/shards/bf629dcb",
        512: "/fs/scratch/PAS2136/samuelstevens/saev/shards/efbdb4d2",
        640: "/fs/scratch/PAS2136/samuelstevens/saev/shards/69e6d6fd",
    }

    cfgs = []
    for n_patches, shards_dpath in shards.items():
        for layer in [21, 23]:
            for k in [16, 32, 64, 128, 256]:
                for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
                    cfgs.append({
                        "tag": f"cambridge-butterflies-{n_patches}p",
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
                        },
                        "val_data": {
                            "layer": layer,
                            "shards": shards_dpath,
                            "ignore_labels": [0],
                        },
                    })
    return cfgs
