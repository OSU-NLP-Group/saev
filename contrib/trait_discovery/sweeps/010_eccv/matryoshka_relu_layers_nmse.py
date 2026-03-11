def make_cfgs() -> list[dict]:
    import os.path

    cfgs: list[dict] = []

    runs_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    dinov3_vitl_in1k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3e27794f"

    # Source: contrib/trait_discovery/notebooks/figures.py plot_layer_comparison RunSpec IDs.
    # Filtered to runs missing inference/3e27794f/metrics.json.
    run_ids_by_layer = {
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
            "u33xj1ig",
            "qtvsac3e",
            "aq8vvjub",
            "xfgouwrz",
            "t56bqbvp",
            "rsmrpkly",
            "obctvep6",
            "e9oeml82",
            "ateood6p",
            "god3i07b",
            "na632sj5",
            "rhjhoav8",
        ],
        17: [
            "di427rrs",
            "xvcht4ti",
            "edx9q34f",
            "pn1f9cge",
            "0r1iy3es",
            "n7pv6rkj",
            "k9i6zi1v",
            "4rhpmk3f",
            "syuerpif",
            "x90r98th",
            "egid27oa",
            "jqx6qdxv",
            "pevusep7",
            "vrepu5ey",
            "pfjnrjjq",
            "7rg0o6tk",
            "av2qk4oj",
            "vkdu21ck",
            "2o9yaiuo",
            "8i936qx0",
        ],
        19: [
            "p5sppjgl",
            "y6osup5x",
            "yi5zik0k",
            "aa30r3nm",
            "sq1ccr13",
            "0tj48gqd",
            "c94z9ib1",
            "7dr58kwn",
            "2uqtzyv6",
            "s96104bm",
            "kbiotiaj",
        ],
        21: [
            "7w24prz9",
            "uus4op1x",
            "4cqe9fha",
            "qcyausyf",
            "i6pxw0q9",
            "zyj9edre",
            "jul72wj6",
            "ophe0g6m",
            "x7py290w",
            "wakiyun9",
            "71u6kzuq",
            "t1ip1brk",
            "ajvxj1a6",
            "pz4up9fd",
            "jlyegk4k",
            "36al8yw7",
            "n5b6p8du",
            "9qywvc6q",
            "a01f97t0",
        ],
    }

    for layer, run_ids in run_ids_by_layer.items():
        for run_id in run_ids:
            cfgs.append({
                "run": os.path.join(runs_root, run_id),
                "save": False,
                "data": {"shards": dinov3_vitl_in1k_val, "layer": layer},
            })

    return cfgs
