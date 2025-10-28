def make_cfgs() -> list[dict]:
    batch_size: int = 1024 * 16
    n_train = 100_000_000
    cfgs = []
    for lr in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        for sparsity_coeff in [3e-4, 1e-3, 3e-3]:
            for layer in [6, 7, 8, 9, 10, 11]:
                cfgs.append({
                    "lr": lr,
                    "n_lr_warmup": 500,
                    "n_sparsity_warmup": n_train // batch_size,
                    "runs_root": "/fs/ess/PAS2136/samuelstevens/saev/runs",
                    "n_train": n_train,
                    "sae": {
                        "normalize_w_dec": True,
                        "remove_parallel_grads": True,
                        "exp_factor": 16,
                    },
                    "objective": {"sparsity_coeff": sparsity_coeff},
                    "train_data": {"layer": layer},
                    "val_data": {"layer": layer},
                })
    return cfgs
