def make_cfgs() -> list[dict]:
    batch_size: int = 1024 * 16
    n_train = 100_000_000

    dinov3_vitl_fishvista_imgfolder = (
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/2e339319"
    )
    dinov3_vitl_fishvista_segfolder_train = (
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/5dcb2f75"
    )
    dinov3_vitl_in1k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/51567c6c"
    dinov3_vitl_in1k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3e27794f"

    shard_pairs = [
        (dinov3_vitl_fishvista_imgfolder, dinov3_vitl_fishvista_segfolder_train),
        (dinov3_vitl_in1k_train, dinov3_vitl_in1k_val),
    ]

    cfgs = []
    for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        for layer in [13, 15, 17, 19, 21, 23]:
            for k in [16, 64, 256]:
                for train, val in shard_pairs:
                    cfgs.append({
                        "tag": "auxk-comparison-v0.3",
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
                            "shards": train,
                            "min_buffer_fill": 0.2,
                        },
                        "val_data": {"layer": layer, "shards": val},
                    })
    return cfgs
