def make_cfgs():
    import os.path

    cfgs = []
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    # DINOv3
    dinov3_fishvista_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/5dcb2f75"
    dinov3_fishvista_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/8692dfa9"

    relu_run_ids = {
        13: [
            "55tmixmp",
            "jc6bvmb9",
            "gzztt8ek",
            "06jcsgfm",
            "y302s3tm",
            "g1ytyldf",
            "5l9ydsu3",
        ],
        15: [
            "lj4c0wgd",
            "i21265ow",
            "7tb7b2sj",
            "k0mxsevo",
            "x8vbn8fn",
            "6ijtpaso",
            "civwwlk1",
            "rul3110d",
            "phr0pl1e",
            "vz6xafm7",
        ],
        17: [
            "19r50l7f",
            "b9os49ea",
            "9xr57269",
            "d2bw9wd6",
            "1d94jkej",
            "4hblxpqf",
            "695opfha",
            "jeofo5xy",
            "cn5lr1qd",
            "lg70kyzy",
        ],
        19: [
            "99uteczr",
            "fpkvcr1c",
            "ak9ol0mo",
            "u0w9q3x7",
            "4yur6h2r",
            "fxm9686f",
        ],
        21: [
            "as9s06dp",
            "p26n5fap",
            "apmvrrql",
            "jjjopw01",
            "mtlv44kr",
            "c8a9jb2s",
            "v2mahh73",
            "lvl8t56b",
            "dimc14vp",
            "a9s7vw72",
            "4cdj4d43",
            "3b1hido7",
        ],
        23: [
            "xa40sswx",
            "jdjiwh1d",
            "d4zpduji",
            "syv70wec",
            "fprsgfvb",
            "px89mieu",
            "rmez38o6",
            "9ie2vdto",
            "xr8ctx57",
            "0uv8xr1h",
            "vx254wah",
            "84k6x68j",
            "d054l9y9",
        ],
    }
    topk_run_ids = {
        13: ["o8eu450v", "ztft3fd3", "utogd78d"],
        15: ["03mlknd1", "e9265of6", "rw66adat"],
        17: ["oi2yermn", "curs60zb", "2b3z0oul"],
        19: ["cmmx8bmo", "wmzxt57z", "o9euqapx"],
        21: ["18z0sbmo", "88hpsuw2", "exo2nkrk"],
        23: ["8pkh7jsv", "tokrebo1", "jljl7cfe"],
    }

    for layer, ids in list(relu_run_ids.items()) + list(topk_run_ids.items()):
        for id in ids:
            cfgs.append({
                "run": os.path.join(run_root, id),
                "train_shards": dinov3_fishvista_train,
                "test_shards": dinov3_fishvista_val,
                "debug": True,
            })

    return cfgs
