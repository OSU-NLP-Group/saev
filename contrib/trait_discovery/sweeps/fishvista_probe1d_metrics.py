def make_cfgs():
    import os.path

    cfgs = []
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    # DINOv3
    dinov3_fishvista_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/5dcb2f75"
    dinov3_fishvista_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/8692dfa9"

    dinov3_run_ids = {
        19: [
            "4fjom2gv",
            "1xmbz0au",
            "ymni4y0e",
            "ap20rd49",
            "ng04xi95",
            "93akoer3",
            "nfs1rd4q",
            "gmjhlwnd",
            "k72pmlxn",
            "karg99yp",
        ],
        21: [
            "me0dkn2i",
            "dg407oo6",
            "x3vpnyuy",
            "o2kme9p0",
            "7koi3mlh",
            "d1fhe1bf",
            "e1inytl8",
            "pf42xwcl",
            "lcmrzj62",
            "yjw34qb5",
        ],
        23: [
            "0u69syhr",
            "7wk4je0m",
            "vx0agl3s",
            "0nssnczi",
            "yr1bnupl",
            "y1vmcaib",
            "9rrslm9e",
        ],
    }

    bioclip2_fishvista_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/c8abf6e8"
    bioclip2_fishvista_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/1bc9cc5d"
    bioclip2_run_ids = {
        19: [
            "de3za3x0",
            "4lrrcpzn",
            "bc93u3fd",
            "7r4o47c8",
            "d4p8aucf",
            "o7686j7v",
            "s744xvcu",
            "02ao05dz",
            "fo016xvs",
            "lepaquzb",
            "xyf3efj2",
            "94rpyymy",
            "2bvglmig",
            "1ba3z434",
        ],
        21: [
            "n8zp5wya",
            "7bpc915f",
            "vtmjukxb",
            "t9j4gial",
            "3175go23",
            "bdnx1rsx",
            "9q2pwwjh",
            "hbwvwpsq",
            "1xn3sebc",
            "sgx540fy",
            "eqkc2xqq",
            "xtqa90n3",
        ],
        23: [
            "ga8m7h7m",
            "bjn5t1vw",
            "upleu6tp",
            "ycm2insd",
            "pc5a6zuq",
            "zgm7np3e",
            "3y5od8ng",
            "yaurei68",
            "nktqy28u",
            "nhv26dg9",
            "s1ogw4nh",
        ],
    }

    for run_ids, train_shards, val_shards in (
        (dinov3_run_ids, dinov3_fishvista_train, dinov3_fishvista_val),
        (bioclip2_run_ids, bioclip2_fishvista_train, bioclip2_fishvista_val),
    ):
        for _, ids in run_ids.items():
            for id in ids:
                cfgs.append({
                    "run": os.path.join(run_root, id),
                    "train_shards": train_shards,
                    "test_shards": val_shards,
                    "debug": True,
                })

    return cfgs
