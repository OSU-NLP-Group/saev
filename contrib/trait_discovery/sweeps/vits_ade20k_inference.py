def make_cfgs():
    import os.path

    cfgs = []

    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    dinov3_vits_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/781f8739"
    dinov3_vits_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/5e195bbf"

    # Pareto run IDs for ViT-S/16 trained on IN1K with 256 patches.
    run_ids: dict[int, list[str]] = {
        6: ["xc0h7cq7", "u5aqv7t7", "usiwpodi", "gcpbnr9n"],
        7: ["zaxl9nqu", "r2g7cj5v", "u6r4jsdm", "d5ej9yuh"],
        8: ["cx2g9omb", "q2i5lq7h", "qwf07reo", "tzwgex0i", "y563o43r", "5gjy7lwi"],
    }

    for layer, ids in run_ids.items():
        for id in ids:
            for shards in [dinov3_vits_ade20k_train, dinov3_vits_ade20k_val]:
                cfgs.append({
                    "run": os.path.join(run_root, id),
                    "data": {"shards": shards, "layer": layer},
                })

    return cfgs
