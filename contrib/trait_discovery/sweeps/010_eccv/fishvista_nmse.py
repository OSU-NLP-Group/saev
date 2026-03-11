def make_cfgs() -> list[dict]:
    import os.path

    cfgs: list[dict] = []

    runs_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    fishvista_train_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/5dcb2f75"
    fishvista_probe_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/8692dfa9"

    # Source: contrib/trait_discovery/sweeps/006_proposal_audit/cls_train.py
    # fishvista_run_ids[23] (DINOv3 ViT-L/16, Matryoshka+TopK).
    run_ids = ["pdikj9bl", "hfpct5ae", "s465wgg4", "dc86xg8z", "bpz34d80"]

    for run_id in run_ids:
        for shards in [fishvista_train_val, fishvista_probe_val]:
            cfgs.append({
                "run": os.path.join(runs_root, run_id),
                "data": {"shards": shards, "layer": 23},
                "save": False,
                "force_recompute": True,
            })

    return cfgs
