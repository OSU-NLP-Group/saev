def make_cfgs():
    import os.path

    from tdiscovery.classification import DecisionTree, PatchAgg

    cfgs = []
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    train_shards = "/fs/scratch/PAS2136/samuelstevens/saev/shards/e65cf404"
    test_shards = "/fs/scratch/PAS2136/samuelstevens/saev/shards/b8a9ff56"

    # SAE runs: layer -> [run_ids]
    fishvista_run_ids = {
        13: ["vjyiz6qo", "4j4cpxpj", "ltpubtmx", "xcqixn3v", "ut004yhy"],
        15: ["3lcqylos", "du0p6063", "bldfz1qi", "qz686gdn", "hp93bxwi"],
        17: ["ihvb8175", "rx0y07bl", "ctftp72w", "qh9mnelt", "48op2zys"],
        19: ["jnl93dlg", "1gywxpjg", "cvjrkpo1", "qnze2wzc", "dwnwbjo9"],
        21: ["fpgvte58", "9ol8p6x7", "u6b884y1", "g2mkhipq", "nuekzgyn"],
        23: ["pdikj9bl", "hfpct5ae", "s465wgg4", "dc86xg8z", "bpz34d80"],
    }

    # Sweep dimensions
    agg_methods = [PatchAgg.MEAN, PatchAgg.MAX]
    max_depths = [3, 5, 7, -1]  # -1 means unlimited
    target_cols = ["habitat", "family"]

    # Each run_id is the best SAE for a different value of k (top-k SAE)
    for layer, ids in fishvista_run_ids.items():
        for run_id in ids:
            for agg in agg_methods:
                for max_depth in max_depths:
                    for target_col in target_cols:
                        cfgs.append({
                            "run": os.path.join(run_root, run_id),
                            "train_shards": train_shards,
                            "test_shards": test_shards,
                            "patch_agg": agg,
                            "target_col": target_col,
                            "cls": DecisionTree(max_depth=max_depth),
                        })

    return cfgs
