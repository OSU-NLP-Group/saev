# uv run scripts/launch.py baseline::inference --run /fs/ess/PAS2136/samuelstevens/tdiscovery/runs/ewbn9nix/ --data.shards /fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0 --data.layer 15 --device cpu


def make_cfgs() -> list[dict]:
    import os.path

    cfgs = []

    runs_root = "/fs/ess/PAS2136/samuelstevens/tdiscovery/saev/runs"
    dinov3_vitl_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0"
    dinov3_vitl_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3802cb66"

    kmeans_runs = [
        # ("kmeans", "jfhj4d4n", 13),
        # ("kmeans", "rwhky9rk", 15),
        # ("kmeans", "jqab2fgy", 17),
        # ("kmeans", "9m9pc35o", 19),
        # ("kmeans", "yez15rhk", 21),
        # ("kmeans", "myy5btgw", 23),
    ]

    pca_runs = [
        # ("pca", "rzevbhc3", 1, 13),
        # ("pca", "hbe6rr6a", 4, 13),
        # ("pca", "g0ps4h0k", 64, 13),
        # ("pca", "87ubodsc", 16, 13),
        ("pca", "5w350qw7", 256, 13),
        ("pca", "scdbam8o", 1024, 13),
        # ("pca", "q5juz55c", 1, 15),
        # ("pca", "023nzlm4", 4, 15),
        # ("pca", "ip5w5ke1", 16, 15),
        # ("pca", "psg5580o", 64, 15),
        ("pca", "1wge1z92", 256, 15),
        ("pca", "0synj8yg", 1024, 15),
        # ("pca", "t0et0ytw", 4, 17),
        # ("pca", "3s4m6j4n", 1, 17),
        # ("pca", "hn7jnlb2", 16, 17),
        # ("pca", "ggf4ymbk", 64, 17),
        ("pca", "5wb5omma", 256, 17),
        ("pca", "pqu34yoc", 1024, 17),
        # ("pca", "yo8dt9cx", 1, 19),
        # ("pca", "2jpk3jpt", 4, 19),
        # ("pca", "km2rfvs2", 16, 19),
        # ("pca", "pz5vzm7f", 64, 19),
        ("pca", "yeodde3p", 256, 19),
        ("pca", "i5b4uioq", 1024, 19),
        # ("pca", "ri7jwid0", 1, 21),
        # ("pca", "f30fd62w", 4, 21),
        # ("pca", "5p71gh7f", 16, 21),
        # ("pca", "iwdhujg7", 64, 21),
        ("pca", "20emivvl", 256, 21),
        ("pca", "zblgdebr", 1024, 21),
        # ("pca", "qmbo5jxw", 1, 23),
        # ("pca", "kwh4twl0", 4, 23),
        # ("pca", "za1xuhhn", 16, 23),
        # ("pca", "a1x1laxm", 64, 23),
        ("pca", "unu6dbfb", 256, 23),
        ("pca", "dzv7ha4u", 1024, 23),
    ]

    for shards in [dinov3_vitl_ade20k_train, dinov3_vitl_ade20k_val]:
        for method, run_id, layer in kmeans_runs:
            cfgs.append({
                "run": os.path.join(runs_root, run_id),
                "data": {"shards": shards, "layer": layer},
            })
        for method, run_id, k, layer in pca_runs:
            cfgs.append({
                "run": os.path.join(runs_root, run_id),
                "data": {"shards": shards, "layer": layer},
            })

    return cfgs
