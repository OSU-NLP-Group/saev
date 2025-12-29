def make_cfgs():
    import os.path

    cfgs = []
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    dinov3_vitl_butterflies_imgfolder = (
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/8bb48fc6"
    )

    run_ids = {
        13: ["k857x9vw", "7r6cnknt", "y7alnu4l", "19n9kpt1", "86tfnhvm"],
        15: ["asw9j64g", "vfmfdq8x", "nmtt7r1w", "ldlog2fk", "gi5ycyok"],
        17: ["il1z8faw", "8xx144hf", "nmr4b7gt", "9dqg7qk2", "evhi174e"],
        19: ["kxvk6pn0", "hzce99jt", "6jbvhosl", "o2faiso0", "e8vqm83t"],
        21: ["047mline", "9cx2x0tz", "ur0y9l04", "fxd7l9dk", "guebah9r"],
        23: ["o0q33ojt", "0qzdqnw0", "s503zwer", "ad8x5nj6", "1q66exeb"],
    }

    for layer, ids in run_ids.items():
        for id in ids:
            cfgs.append({
                "run": os.path.join(run_root, id),
                "data": {"shards": dinov3_vitl_butterflies_imgfolder, "layer": layer},
            })

    return cfgs
