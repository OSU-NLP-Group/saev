def make_cfgs():
    import os.path

    cfgs = []
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    # PE-spatial ADE20K shards
    pe_spatial_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/4b279034"
    pe_spatial_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/245e4d61"

    # PE-spatial Pareto checkpoint IDs (from 008_pe notebook)
    pe_spatial_run_ids = {
        21: ["j0hmtzkh", "nv55sqo1", "3hqhkxqf", "sp94dh0t", "cqfjwsif"],
        23: ["3u4wh1m9", "barw9k1k", "ecbauxoc", "5okeekou", "0xt63hu0"],
    }

    for layer, ids in pe_spatial_run_ids.items():
        for shards in [pe_spatial_ade20k_train, pe_spatial_ade20k_val]:
            for run_id in ids:
                cfgs.append({
                    "run": os.path.join(run_root, run_id),
                    "data": {"shards": shards, "layer": layer},
                })

    return cfgs
