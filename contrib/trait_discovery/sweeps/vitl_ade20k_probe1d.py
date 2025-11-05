def make_cfgs():
    import os.path

    cfgs = []

    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    dinov3_vitl_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0"
    dinov3_vitl_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3802cb66"

    # Pareto run IDs for ViT-S/16 trained on IN1K with 256 patches.
    run_ids: dict[int, list[str]] = {
        21: [
            # "qcyausyf",
            # "i6pxw0q9",
            # "zyj9edre",
            # "x7py290w",
            # "v4pyroov",
            # "71u6kzuq",
            # "t1ip1brk",
            "pz4up9fd",
            "36al8yw7",
            "y8vhxwya",
        ],
        23: [
            # "5mv59srt",
            # "rfic94if",
            # "q91eu62e",
            # "t88ez13w",
            # "9drbwvhg",
            "ybm0jqi4",
        ],
    }

    for _, ids in run_ids.items():
        for id in ids:
            cfgs.append({
                "run": os.path.join(run_root, id),
                "train_shards": dinov3_vitl_ade20k_train,
                "test_shards": dinov3_vitl_ade20k_val,
                "debug": True,
            })

    return cfgs
