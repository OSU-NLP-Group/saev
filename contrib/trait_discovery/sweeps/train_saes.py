def make_cfgs() -> list[dict]:
    cfgs = []
    for lr in [3e-4, 1e-3, 3e-3, 1e-2]:
        for sparsity_coeff in [3e-4, 1e-3, 3e-3]:
            for layer in [13, 15, 17, 19, 21, 23]:
                cfgs.append({
                    "lr": lr,
                    "n_lr_warmup": 500,
                    "n_sparsity_warmup": 6000,
                    "runs_root": "/fs/ess/PAS2136/samuelstevens/saev/runs",
                    "n_train": 100_000_000,
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
