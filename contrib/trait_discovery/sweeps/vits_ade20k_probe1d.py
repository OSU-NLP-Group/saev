# uv run scripts/launch.py probe1d \
#   --run /fs/ess/PAS2136/samuelstevens/saev/runs/xc0h7cq7/ \
#   --train-shards /fs/scratch/PAS2136/samuelstevens/saev/shards/781f8739 \
#   --test-shards /fs/scratch/PAS2136/samuelstevens/saev/shards/5e195bbf \
#   --slurm-acct PAS2136 \
#   --slurm-partition nextgen \
#   --n-hours 4 \
#   --device cuda


def make_cfgs():
    import os.path

    cfgs = []

    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    dinov3_vits_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/781f8739"
    dinov3_vits_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/5e195bbf"

    # Pareto run IDs for ViT-S/16 trained on IN1K with 256 patches.
    run_ids: dict[int, list[str]] = {
        6: [
            "3zih0tpa",
            "ct5w12zx",
            "f5qog0he",
            "l6da5024",
            # "xc0h7cq7",
            # "u5aqv7t7",
            # "usiwpodi",
            # "gcpbnr9n",
            # "nbqbvh45",
            # "r2sf1w19",
            "sfpco1tn",
        ],
        7: [
            "l9stkmwt",
            "tbfdr3cc",
            "c7w1t9jc",
            "xvxn1ed1",
            # "zaxl9nqu",
            # "r2g7cj5v",
            # "u6r4jsdm",
            # "d5ej9yuh",
            # "pvtt26ky",
            "hu77o1op",
            "36538viq",
        ],
        8: [
            "p0z7t1ci",
            "125xh1t9",
            # "cx2g9omb",
            # "q2i5lq7h",
            # "qwf07reo",
            # "tzwgex0i",
            # "y563o43r",
            "5gjy7lwi",
            "33oh6osq",
            "bvsb2257",
        ],
        9: [
            # "qt0fmmxm",
            # "fj8b9r5o",
            # "1o4uc5bf",
            # "z1qvy51u",
            # "1ihxsv0i",
            # "euoj6wv0",
            # "flfplqsa",
            # "5o0mby2h",
            # "ickedctl",
        ],
        10: [
            # "chn5wi3x",
            # "219r3phu",
            # "knglrhzb",
            # "21d1kgyk",
            # "jt45lucm",
            # "6hrok1al",
            # "qrjtyj70",
            # "3j06kxdt",
            # "g4dexqq1",
        ],
        11: [
            # "jgu19fzx",
            # "8yd05vxi",
            # "utmjp20e",
            # "gc6iqrf2",
            # "5ewxrjg4",
            # "x7e75z6t",
            # "hyda2tk7",
            "36ztscy4",
        ],
    }
    for _, ids in run_ids.items():
        for id in ids:
            cfgs.append({
                "run": os.path.join(run_root, id),
                "train_shards": dinov3_vits_ade20k_train,
                "test_shards": dinov3_vits_ade20k_val,
                "debug": True,
            })

    return cfgs
