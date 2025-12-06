def make_cfgs():
    import os.path

    cfgs = []
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    dinov3_fishvista_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/5dcb2f75"
    dinov3_fishvista_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/8692dfa9"

    dinov3_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0"
    dinov3_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3802cb66"

    fishvista_run_ids = [
        # "a5bpj8yd",
        # "r2q7d4fe",
        # "2mmqgv8c",
        # "zz0bnvrx",
        # "j6jkpr0e",
        # "b15hxbl6",
        # "na664hrj",
        # "e5l8mq2v",
        # "897avdav",
        # "h5q0ta4s",
        # "lvxu51uh",
        # "j9rqnmrp",
        # "r4m2q7sj",
        # "3bbf6vol",
        # "1b3wldm1",
        # "99rcaevz",
        # "d5nm5jep",
        # "cpwpt43q",
        # "y4kmefi2",
        # "yv8cjbxs",
        # "tz3n0glw",
        # "2kek5j0o",
        # "1g2znfa0",
        # "wf11niof",
        # "d6scd1h2",
        # "3ww2qwv9",
        # "a64s5e6c",
        # "fy2ax39q",
        # "bdflmpr4",
        # "mkj210qa",
        # "62v3xgpn",
        # "sfogu4f5",
        # "2h4wb4p4",
        # "egqf52o2",
        # "z586ah21",
        # "l7jghyo8",
        # "u8l17uxa",
        # "1akvk57c",
    ]
    in1k_run_ids = [
        # "mhm4yn0a",
        # "mw6hifr0",
        # "a1iwhild",
        # "hofpivjj",
        # "kfuntjnw",
        # "4vuk4k7k",
        # "jq0p4t5d",
        # "0peq0v64",
        "urj2oz2b",
        "9nm7bjan",
        "9icq784s",
        # "hzycamw2",
        # "upao4mcq",
        # "5otkn3mm",
        # "i4zbglfh",
        # "hqdrpp9g",
        # "l81k4bnz",
        # "f7aowx0k",
        # "1ancs7nb",
        "3lzel1gr",
        "8qxolknr",
        "msjn0it3",
        # "ptqk2m8f",
        # "r7plffxc",
        # "iq5sef52",
        # "wv0r8ywz",
        # "okbjkgcc",
        # "k2yetftn",
        # "u3ogyz1m",
        # "3jpnri0j",
        # "r1ayls0t",
        # "v1c0d5ir",
        # "iv3olxgp",
        # "3vj2kfdu",
        # "2owewqmr",
        # "jpmb4ceg",
    ]

    for id in fishvista_run_ids:
        cfgs.append({
            "run": os.path.join(run_root, id),
            "train_shards": dinov3_fishvista_train,
            "test_shards": dinov3_fishvista_val,
            "debug": True,
        })
    for id in in1k_run_ids:
        cfgs.append({
            "run": os.path.join(run_root, id),
            "train_shards": dinov3_ade20k_train,
            "test_shards": dinov3_ade20k_val,
            "debug": True,
        })

    return cfgs
