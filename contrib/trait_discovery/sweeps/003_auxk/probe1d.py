def make_cfgs():
    import os.path

    cfgs = []
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    dinov3_fishvista_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/5dcb2f75"
    dinov3_fishvista_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/8692dfa9"

    dinov3_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0"
    dinov3_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3802cb66"

    fishvista_run_ids = {
        13: [
            # "nz4kbwrg",
            # "7mutzotf",
            # "14byzh33",
            # "u5w7spva",
            # "swkzf6h4",
            # "1liq98bm",
        ],
        15: [
            # "k0mpsqkk",
            # "np180i6p",
            # "5x0q8dp8",
            # "qft1m6z1",
            # "q9v1m71c",
            # "y2qs0ei9",
        ],
        17: [
            # "220wrvvr",
            # "uqn5b99y",
            # "jmvt870b",
            # "roc94btc",
            # "q7qmy8w1",
            # "odyp6oke",
        ],
        19: [
            # "aql7nsgb",
            # "umh8wxrg",
            # "rh0amlbo",
            # "jdxh601v",
            # "9gi0f31g",
            # "757wfw38",
        ],
        21: [
            # "b7xtriqt",
            # "xkow5w0y",
            # "im0wwtm8",
            # "hh35eaw9",
            # "5f2p2pes",
            # "pouei803",
        ],
        23: [
            # "gi8l3pk1",
            # "x0mkt04y",
            # "c3yusm40",
            # "um6hbn05",
            # "r0m5713d",
            # "i6z66rly",
        ],
    }
    in1k_run_ids = {
        13: [
            # "3ld8ilmo",
            # "l03epvhu",
            # "co7dpa0w",
            # "kpadjov4",
            # "2edpn91i",
            # "1up044nl",
        ],
        15: [
            # "6r92o6t6",
            # "e4w7u0np",
            # "jsr327fs",
            # "emz255bp",
            # "ffqb9b3n",
            # "3hzenf5e",
        ],
        17: [
            # "tkdd41tq",
            # "4g4lbmgs",
            # "h8nfg6ci",
            # "2hsh4w50",
            # "jjz6a7ja",
            # "huzxe3hu",
        ],
        19: [
            # "0c4mlnn7",
            # "6x4t5t76",
            # "xk0a9w3g",
            # "cdu13t6j",
            # "hh7d7yop",
            # "32zm1zcd",
        ],
        21: [
            # "rez38zbu",
            # "jxxje744",
            "2k6kq9f2",
            # "jttb6ijl",
            # "s5srn2q7",
            # "qurkdz1r",
        ],
        23: [
            # "a95jzikd",
            # "elwq2g19",
            "ztnu4ml1",
            "flqkcam7",
            "s3pqewz1",
            "l8hooa3r",
        ],
    }

    for layer, ids in fishvista_run_ids.items():
        for id in ids:
            cfgs.append({
                "run": os.path.join(run_root, id),
                "train_shards": dinov3_fishvista_train,
                "test_shards": dinov3_fishvista_val,
                "debug": True,
            })
    for layer, ids in in1k_run_ids.items():
        for id in ids:
            cfgs.append({
                "run": os.path.join(run_root, id),
                "train_shards": dinov3_ade20k_train,
                "test_shards": dinov3_ade20k_val,
                "debug": True,
            })

    return cfgs
