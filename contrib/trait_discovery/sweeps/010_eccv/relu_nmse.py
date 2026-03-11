def make_cfgs() -> list[dict]:
    import os.path

    cfgs: list[dict] = []

    runs_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    dinov3_vitl_in1k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3e27794f"

    # W&B filter:
    # project=samuelstevens/saev
    # tags contains in1k-v0.4.1
    # config.train_data.layer == 23
    # config.train_data.shards == /fs/scratch/PAS2136/samuelstevens/saev/shards/51567c6c
    # config.val_data.shards == /fs/scratch/PAS2136/samuelstevens/saev/shards/3e27794f
    in1k_relu_run_ids = [
        "o1p9wl76",
        "4mlqkei5",
        "afyydka9",
        "wrnz7h7h",
        "xayzq0hf",
        "0pdum8cq",
        "rggc5m8n",
        "kqlfguzz",
        "uojvw5o4",
        "t1na9yxo",
        "extl56w1",
        "6iom0amk",
        "i1ujcsi6",
        "ho86h0gp",
        "as770651",
        "yt2roil5",
        "dt1y8m94",
        "xu8n5209",
        "p2ycew4h",
        "e2jsvsbx",
        "cjtedfwa",
        "iy4mtm9y",
        "l2yvdllc",
        "99l40o12",
    ]

    for run_id in in1k_relu_run_ids:
        cfgs.append({
            "run": os.path.join(runs_root, run_id),
            "save": False,
            "data": {"shards": dinov3_vitl_in1k_val, "layer": 23},
        })

    return cfgs
