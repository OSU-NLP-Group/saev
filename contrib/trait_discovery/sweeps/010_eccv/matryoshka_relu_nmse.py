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
    in1k_matryoshka_relu_run_ids = [
        "lnleoyf6",
        "ibt2fgta",
        "6l12fjm9",
        "5mv59srt",
        "rfic94if",
        "t1vh0qy1",
        "u3gj24az",
        "mccrm7u8",
        "q91eu62e",
        "t88ez13w",
        "eosnewqp",
        "yfpdczj7",
        "fxcpfysr",
        "xg2vom0w",
        "rqdpylmi",
        "s3dxavbq",
        "2zf86reb",
        "1247ezti",
        "kd2pd8rs",
        "9drbwvhg",
        "09srbijj",
        "1qynjykb",
        "02ors1ov",
        "vxgyr2du",
        "x6ho5md2",
        "0pz90ly4",
        "ybm0jqi4",
        "kn0f5a3v",
        "2pdk23cz",
        "3kkf33w6",
        "9fn4l6rf",
        "9pdmmk1r",
        "o1vnl1yp",
    ]

    for run_id in in1k_matryoshka_relu_run_ids:
        cfgs.append({
            "run": os.path.join(runs_root, run_id),
            "save": False,
            "data": {"shards": dinov3_vitl_in1k_val, "layer": 23},
        })

    return cfgs
