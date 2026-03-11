"""Sweep config for evaluating classifier checkpoints with proposal-audit framework."""

import pathlib


def make_cfgs():
    run_root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")

    # FishVista val shards (segmentation labels for audit)
    dinov3_fishvista_val = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/b8a9ff56"
    )

    # ADE20K val shards
    dinov3_ade20k_val = pathlib.Path(
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/3802cb66"
    )

    # Run IDs from cls_train.py
    fishvista_run_ids = {
        13: ["vjyiz6qo", "4j4cpxpj", "ltpubtmx", "xcqixn3v", "ut004yhy"],
        15: ["3lcqylos", "du0p6063", "bldfz1qi", "qz686gdn", "hp93bxwi"],
        17: ["ihvb8175", "rx0y07bl", "ctftp72w", "qh9mnelt", "48op2zys"],
        19: ["jnl93dlg", "1gywxpjg", "cvjrkpo1", "qnze2wzc", "dwnwbjo9"],
        21: ["fpgvte58", "9ol8p6x7", "u6b884y1", "g2mkhipq", "nuekzgyn"],
        23: ["pdikj9bl", "hfpct5ae", "s465wgg4", "dc86xg8z", "bpz34d80"],
    }
    in1k_run_ids = {
        13: ["3ld8ilmo", "l03epvhu", "co7dpa0w", "kpadjov4", "2edpn91i", "1up044nl"],
        15: ["6r92o6t6", "e4w7u0np", "jsr327fs", "emz255bp", "ffqb9b3n", "3hzenf5e"],
        17: ["tkdd41tq", "4g4lbmgs", "h8nfg6ci", "2hsh4w50", "jjz6a7ja", "huzxe3hu"],
        19: ["0c4mlnn7", "6x4t5t76", "xk0a9w3g", "cdu13t6j", "hh7d7yop", "32zm1zcd"],
        21: ["rez38zbu", "jxxje744", "2k6kq9f2", "jttb6ijl", "s5srn2q7", "qurkdz1r"],
        23: ["a95jzikd", "elwq2g19", "ztnu4ml1", "flqkcam7", "s3pqewz1", "l8hooa3r"],
    }

    # Classifier patterns from cls_train.py
    fishvista_tasks = [
        "habitat",
        "cruisers_vs_maneuverers",
        "pelagic_vs_demersal",
        "shallow_vs_deep",
    ]
    ade20k_tasks = ["scene_top50"]
    cls_strs = [
        "C0.001",
        "C0.01",
        "C0.1",
        "depth-1",
        "depth2",
        "depth3",
        "depth5",
        "depth8",
    ]

    cfgs = []

    # FishVista SAE runs
    for _layer, run_ids in fishvista_run_ids.items():
        for run_id in run_ids:
            run = run_root / run_id
            inference_dpath = run / "inference" / dinov3_fishvista_val.name

            # Collect all classifier checkpoints for this run
            checkpoints = []
            for task in fishvista_tasks:
                for cls_str in cls_strs:
                    ckpt = inference_dpath / f"cls_{task}_max_{cls_str}.pkl"
                    if ckpt.exists():
                        checkpoints.append(ckpt)

            if not checkpoints:
                continue

            cfgs.append({
                "run": run,
                "test_shards": dinov3_fishvista_val,
                "cls_checkpoints": tuple(checkpoints),
                "max_budget": 1000,
                "tau": 0.3,
                "budgets": (3, 10, 30, 100, 300, 1000),
                "ignore_label_ids": (0,),
            })

    # IN1K SAE runs (evaluated on ADE20K)
    for _layer, run_ids in in1k_run_ids.items():
        for run_id in run_ids:
            run = run_root / run_id
            inference_dpath = run / "inference" / dinov3_ade20k_val.name

            # Collect all classifier checkpoints for this run
            checkpoints = []
            for task in ade20k_tasks:
                for cls_str in cls_strs:
                    ckpt = inference_dpath / f"cls_{task}_max_{cls_str}.pkl"
                    if ckpt.exists():
                        checkpoints.append(ckpt)

            if not checkpoints:
                continue

            cfgs.append({
                "run": run,
                "test_shards": dinov3_ade20k_val,
                "cls_checkpoints": tuple(checkpoints),
                "max_budget": 1000,
                "tau": 0.3,
                "budgets": (3, 10, 30, 100, 300, 1000),
                "ignore_label_ids": (0, 255),  # ADE20K has void=255
            })

    return cfgs
