def make_cfgs() -> list[dict]:
    import os.path

    cfgs: list[dict] = []

    runs_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    dinov3_vitb_in1k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/8762551e"
    dinov3_vits_in1k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/52ec5790"

    # Source: contrib/trait_discovery/scripts/push_dinov3.py
    # vitb_run_ids[11] and vits_run_ids[11], final layer for ViT-B/16 and ViT-S/16.
    run_ids_by_shard = {
        dinov3_vitb_in1k_val: [
            "n1xwev0z",
            "ef657fwa",
            "qoc1660r",
            "6crsj9gj",
            "d4v8aruu",
            "7mpdhd0n",
            "abhe5g2j",
            "22p3bnt8",
        ],
        dinov3_vits_in1k_val: [
            "jgu19fzx",
            "8yd05vxi",
            "utmjp20e",
            "gc6iqrf2",
            "5ewxrjg4",
            "x7e75z6t",
            "hyda2tk7",
            "36ztscy4",
        ],
    }

    for shards, run_ids in run_ids_by_shard.items():
        for run_id in run_ids:
            cfgs.append({
                "run": os.path.join(runs_root, run_id),
                "save": False,
                "data": {"shards": shards, "layer": 11},
            })

    return cfgs
