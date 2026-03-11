def make_cfgs() -> list[dict]:
    import os.path

    cfgs: list[dict] = []

    runs_root = "/fs/ess/PAS2136/samuelstevens/tdiscovery/saev/runs"
    dinov3_vitl_in1k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3e27794f"

    baseline_run_ids = [
        "myy5btgw",  # kmeans, k=1
        "qmbo5jxw",  # pca, k=1
        "kwh4twl0",  # pca, k=4
        "za1xuhhn",  # pca, k=16
        "a1x1laxm",  # pca, k=64
        "unu6dbfb",  # pca, k=256
        "dzv7ha4u",  # pca, k=1024
        "lm51bf37",  # semi-nmf, k=1
        "em7hzdw0",  # semi-nmf, k=4
        "cmf1j0gd",  # semi-nmf, k=16
        "q6qtn8f6",  # semi-nmf, k=64
        "rv1wfbws",  # semi-nmf, k=256
        "k9sot7dd",  # semi-nmf, k=1024
    ]

    for run_id in baseline_run_ids:
        cfgs.append({
            "run": os.path.join(runs_root, run_id),
            "save": False,
            "data": {"shards": dinov3_vitl_in1k_val, "layer": 23},
        })

    return cfgs
