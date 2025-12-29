def make_cfgs() -> list[dict]:
    batch_size: int = 1024 * 16
    n_train = 100_000_000

    dinov3_vitl_butterflies_imgfolder = (
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/8bb48fc6"
    )

    cfgs = []
    # for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
    for lr in [3e-3, 1e-2, 3e-2]:
        for layer in [13, 15, 17, 19, 21, 23]:
            for k in [16, 32, 64, 128, 256]:
                cfgs.append({
                    "tag": "butterflies-v0.1",
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
                        "shards": dinov3_vitl_butterflies_imgfolder,
                        "min_buffer_fill": 0.2,
                    },
                    "val_data": {
                        "layer": layer,
                        "shards": dinov3_vitl_butterflies_imgfolder,
                    },
                })
    return cfgs
