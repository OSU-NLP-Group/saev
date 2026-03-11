def make_cfgs() -> list[dict]:
    import os.path

    cfgs: list[dict] = []

    runs_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    dinov3_vitl_in1k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3e27794f"

    in1k_matryoshka_topk_run_ids = [
        "flqkcam7",
        "s3pqewz1",
        "l8hooa3r",
    ]

    for run_id in in1k_matryoshka_topk_run_ids:
        cfgs.append({
            "run": os.path.join(runs_root, run_id),
            "save": False,
            "data": {"shards": dinov3_vitl_in1k_val, "layer": 23},
        })

    return cfgs
