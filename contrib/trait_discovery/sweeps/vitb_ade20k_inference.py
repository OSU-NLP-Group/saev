def make_cfgs():
    import os.path

    cfgs = []

    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    dinov3_vitb_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/fd5781e8"
    dinov3_vitb_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/66a5d2c1"

    # Pareto run IDs for ViT-S/16 trained on IN1K with 256 patches.
    run_ids: dict[int, list[str]] = {
        6: [],
        7: [],
        8: [],
        9: [],
        10: [
            "eqht2edc",
            "bzmeiyat",
            "1hjlnu1s",
            "ssoshhfv",
            "oc5jcdu8",
            "jpnwfh3w",
            "2we45xxf",
            "bv1h09se",
            "0akkhcjf",
            "yb185c6g",
            "jjewtqwp",
        ],
        11: [
            "n1xwev0z",
            "ef657fwa",
            "qoc1660r",
            "6crsj9gj",
            "d4v8aruu",
            "7mpdhd0n",
            "abhe5g2j",
            "22p3bnt8",
        ],
    }

    for layer, ids in run_ids.items():
        for id in ids:
            for shards in [dinov3_vitb_ade20k_train, dinov3_vitb_ade20k_val]:
                cfgs.append({
                    "run": os.path.join(run_root, id),
                    "data": {"shards": shards, "layer": layer},
                })

    return cfgs
