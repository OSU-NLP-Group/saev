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
            "q7gsjgjw",
            "06jcsgfm",
            "3cztz1e2",
            "aznajx0a",
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
            "uhjb92s0",
            "rul3110d",
            "phr0pl1e",
            "vz6xafm7",
            "p8vnx1o4",
        ],
        17: [
            "19r50l7f",
            "b9os49ea",
            "9xr57269",
            "d2bw9wd6",
            "1d94jkej",
            "4hblxpqf",
            "695opfha",
            "fkdjmzl9",
            "jeofo5xy",
            "cn5lr1qd",
            "k2zta2sv",
            "ev44ldso",
            "lg70kyzy",
            "px7yedws",
            "z1lcxt8p",
            "ebz3sgy0",
        ],
        19: [
            "99uteczr",
            "fpkvcr1c",
            "ak9ol0mo",
            "l9581v9g",
            "zl0dc758",
            "muoij4x9",
            "u0w9q3x7",
            "4yur6h2r",
            "tgeapbmb",
            "fxm9686f",
            "iakth8gg",
            "0x1mmlg1",
        ],
        21: [
            "as9s06dp",
            "p26n5fap",
            "apmvrrql",
            "jjjopw01",
            "u0kb2h5r",
            "mtlv44kr",
            "c8a9jb2s",
            "dk5qn14g",
            "v2mahh73",
            "lvl8t56b",
            "dimc14vp",
            "frzu6p9x",
            "i0f9eemi",
            "a9s7vw72",
            "4cdj4d43",
            "3b1hido7",
        ],
        23: [
            "xa40sswx",
            "jdjiwh1d",
            "3dwnlqf7",
            "54zw9rwz",
            "fprsgfvb",
            "gjl1ser9",
            "ahi0jb27",
            "vkjelijv",
            "rmez38o6",
            "ksxv8s0j",
            "9ie2vdto",
            "xr8ctx57",
            "0uv8xr1h",
            "0b4fgrkw",
            "vx254wah",
            "keadjq2e",
            "84k6x68j",
            "d054l9y9",
            "brr6pkl8",
        ],
    }
    topk_run_ids = {
        13: ["4qceyi1o", "f9yxwe45", "jy6azbts"],
        15: ["likts7ds", "pp6hqb6u", "yjul04yh"],
        17: ["gllm2myc", "13vg4iui", "wy1p56cq"],
        19: ["pai0qve7", "ob2sbnk6", "o9euqapx"],
        21: ["1gik8q7q", "jtoh0nnc", "bnzotm7j"],
        23: ["lshbfxm2", "7kiamy1u", "db04ccac"],
    }

    for layer, ids in list(relu_run_ids.items()) + list(topk_run_ids.items()):
        for id in ids:
            for shards in [dinov3_fishvista_train, dinov3_fishvista_val]:
                cfgs.append({
                    "run": os.path.join(run_root, id),
                    "data": {"shards": shards, "layer": layer},
                })

    return cfgs
