def make_cfgs():
    import os.path

    cfgs = []
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    # PE-core ADE20K shards (save activations for probe1d)
    pe_core_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/fa2b7ff0"
    pe_core_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/80219cbf"

    # PE-core IN1K val shards (metrics only, no save)
    pe_core_in1k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/a7f78fe3"

    # PE-core Pareto checkpoint IDs (from 009_pe_core notebook)
    pe_core_run_ids = {
        21: ["6ed9ojrt", "ang7atm3", "ogpjtuij", "xq1zfqh1", "9u9ny8nm"],
        23: ["h4gy7fke", "ywydn3z5", "omk5qhxf", "f3a9b41q", "r69kzt74"],
    }

    for layer, ids in pe_core_run_ids.items():
        # ADE20K train + val: save activations for probe1d
        for shards in [pe_core_ade20k_train, pe_core_ade20k_val]:
            for run_id in ids:
                cfgs.append({
                    "run": os.path.join(run_root, run_id),
                    "data": {"shards": shards, "layer": layer},
                    "save": True,
                })

        # IN1K val: metrics only (NMSE, L0)
        for run_id in ids:
            cfgs.append({
                "run": os.path.join(run_root, run_id),
                "data": {"shards": pe_core_in1k_val, "layer": layer},
                "save": False,
            })

    return cfgs
