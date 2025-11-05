def make_cfgs():
    import os.path

    cfgs = []

    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    dinov3_vitb_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/fd5781e8"
    dinov3_vitb_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/66a5d2c1"

    # Pareto run IDs for ViT-S/16 trained on IN1K with 256 patches.
    run_ids: dict[int, list[str]] = {
        6: [
            # "y7uk853s",
            # "odd8ogb4",
            # "db4qjvkf",
            # "e4i667sz",
            # "extgn8yv",
            # "ku0rwwex",
            # "8hyzbyht",
            # "vsvkrqfg",
            # "t2xcozei",
            "gyyfc054",
            "lppv40ws",
        ],
        7: [
            # "iyb7ec1w",
            # "rxjh04w7",
            # "688bm8ht",
            # "f6i2ow8w",
            # "uiqc8e2f",
            # "40j71aj2",
            # "xucm378k",
            # "21trz0ik",
            # "knz7yndg",
            # "1hcm0oqu",
            "6yhupj05",
        ],
        8: [
            # "wgh9hgih",
            # "poe1kh3i",
            # "ttghd72n",
            # "sabum27l",
            # "q6pg7hl9",
            # "r3opp7dy",
            # "2obnw9ky",
            # "bk7iwhfu",
            # "o4cheohl",
            "dk6k8hc0",
            "rc82kpln",
        ],
        9: [
            # "cozptrw2",
            # "cynce806",
            # "1aod3v62",
            # "5g0ez3ix",
            # "zvx4qkov",
            # "na2k2dyp",
            # "6h9n14t3",
            # "g6ga929x",
            # "tqk8igwb",
            "hs18j6i2",
            "t0qdoi9u",
            "893a4vol",
        ],
        10: [
            # "eqht2edc",
            # "bzmeiyat",
            # "1hjlnu1s",
            # "ssoshhfv",
            # "oc5jcdu8",
            # "jpnwfh3w",
            # "2we45xxf",
            # "bv1h09se",
            # "0akkhcjf",
            # "yb185c6g",
            # "jjewtqwp",
        ],
        11: [
            # "n1xwev0z",
            # "ef657fwa",
            # "qoc1660r",
            # "6crsj9gj",
            # "d4v8aruu",
            # "7mpdhd0n",
            # "abhe5g2j",
            # "22p3bnt8",
        ],
    }

    for _, ids in run_ids.items():
        for id in ids:
            cfgs.append({
                "run": os.path.join(run_root, id),
                "train_shards": dinov3_vitb_ade20k_train,
                "test_shards": dinov3_vitb_ade20k_val,
                "debug": True,
            })

    return cfgs
