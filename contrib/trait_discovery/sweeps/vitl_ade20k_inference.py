def make_cfgs():
    import os.path

    cfgs = []

    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    dinov3_vitl_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0"
    dinov3_vitl_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3802cb66"

    # Pareto run IDs for ViT-S/16 trained on IN1K with 256 patches.
    matryoshka_run_ids: dict[int, list[str]] = {
        13: [
            "jsqj2arm",
            "3hr3d3w0",
            "lq18pdy9",
            "kk60aru4",
            "fh0jta0t",
            "y8q60ohz",
            "u7tz0xii",
            "ag8agm56",
            "ja4zp5kn",
            "220r8j1q",
            "fjravp6a",
            "yhu9d2z9",
            "fkl5sxba",
        ],
        15: [
            "fi7qafny",
            "txmrh5nd",
            "qtvsac3e",
            "aq8vvjub",
            "xfgouwrz",
            "rsmrpkly",
            "e9oeml82",
        ],
        17: [
            "di427rrs",
            "edx9q34f",
            "pn1f9cge",
            "n7pv6rkj",
            "4rhpmk3f",
            "syuerpif",
            "egid27oa",
            "jqx6qdxv",
            "vrepu5ey",
            "av2qk4oj",
            "vkdu21ck",
        ],
        19: [
            "y6osup5x",
            "yi5zik0k",
            "aa30r3nm",
            "sq1ccr13",
            "0tj48gqd",
            "7dr58kwn",
            "2uqtzyv6",
            "s96104bm",
        ],
        21: [
            "qcyausyf",
            "i6pxw0q9",
            "zyj9edre",
            "x7py290w",
            "v4pyroov",
            "71u6kzuq",
            "t1ip1brk",
            "pz4up9fd",
            "36al8yw7",
            "y8vhxwya",
        ],
        23: [
            "lnleoyf6",
            "ibt2fgta",
            "6l12fjm9",
            "rfic94if",
            "t1vh0qy1",
            "mccrm7u8",
            "t88ez13w",
            "eosnewqp",
            "fxcpfysr",
            "kd2pd8rs",
            "9drbwvhg",
            "1qynjykb",
            "0pz90ly4",
            "ybm0jqi4",
            "2pdk23cz",
            "9fn4l6rf",
        ],
    }

    vanilla_run_ids: dict[int, list[str]] = {
        23: [
            "afyydka9",
            "wrnz7h7h",
            "xayzq0hf",
            "0pdum8cq",
            "t1na9yxo",
            "extl56w1",
            "6iom0amk",
            "i1ujcsi6",
            "ho86h0gp",
            "as770651",
            "yt2roil5",
            "dt1y8m94",
        ]
    }

    for layer, ids in list(matryoshka_run_ids.items()) + list(vanilla_run_ids.items()):
        for id in ids:
            for shards in [dinov3_vitl_ade20k_train, dinov3_vitl_ade20k_val]:
                cfgs.append({
                    "run": os.path.join(run_root, id),
                    "data": {"shards": shards, "layer": layer},
                })

    return cfgs
