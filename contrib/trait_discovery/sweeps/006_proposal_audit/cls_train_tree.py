def make_cfgs():
    import os.path

    from tdiscovery.classification import DecisionTree, LabelGrouping, PatchAgg

    cfgs = []
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    dinov3_fishvista_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/e65cf404"
    dinov3_fishvista_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/b8a9ff56"

    dinov3_ade20k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0"
    dinov3_ade20k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3802cb66"

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

    ade20k_top50 = [
        "street",
        "misc",
        "bedroom",
        "living_room",
        "bathroom",
        "kitchen",
        "dining_room",
        "skyscraper",
        "highway",
        "building_facade",
        "conference_room",
        "hotel_room",
        "office",
        "mountain_snowy",
        "corridor",
        "airport_terminal",
        "game_room",
        "waiting_room",
        "poolroom_home",
        "home_office",
        "art_studio",
        "attic",
        "broadleaf",
        "park",
        "mountain",
        "exterior",
        "coast",
        "alley",
        "parlor",
        "closet",
        "beach",
        "childs_room",
        "art_gallery",
        "apartment_building_outdoor",
        "staircase",
        "castle",
        "pasture",
        "dorm_room",
        "nursery",
        "natural",
        "lobby",
        "garage_indoor",
        "reception",
        "needleleaf",
        "bar",
        "warehouse_indoor",
        "shop",
        "roundabout",
        "house",
        "casino_indoor",
    ]
    ade20k_task = LabelGrouping(
        name="scene_top50",
        source_col="scene",
        groups={label: [label] for label in ade20k_top50},
    )
    fishvista_tasks = [
        LabelGrouping(name="habitat", source_col="habitat"),
        LabelGrouping(
            name="cruisers_vs_maneuverers",
            source_col="habitat",
            groups={
                "cruisers": ["pelagic-oceanic", "pelagic-neritic", "pelagic"],
                "maneuverers": ["reef-associated"],
            },
        ),
        LabelGrouping(
            name="pelagic_vs_demersal",
            source_col="habitat",
            groups={
                "pelagic": [
                    "pelagic-oceanic",
                    "pelagic-neritic",
                    "pelagic",
                    "epipelagic",
                ],
                "demersal": ["demersal", "bathydemersal", "benthopelagic"],
            },
        ),
        LabelGrouping(
            name="shallow_vs_deep",
            source_col="habitat",
            groups={
                "shallow": ["epipelagic", "reef-associated", "pelagic-neritic"],
                "deep": [
                    "mesopelagic",
                    "bathypelagic",
                    "abyssopelagic",
                    "bathydemersal",
                ],
            },
        ),
    ]

    max_depths = [2, 3, 5, 8, -1]

    for _layer, ids in fishvista_run_ids.items():
        for run_id in ids:
            for max_depth in max_depths:
                for task in fishvista_tasks:
                    cfgs.append({
                        "run": os.path.join(run_root, run_id),
                        "train_shards": dinov3_fishvista_train,
                        "test_shards": dinov3_fishvista_val,
                        "patch_agg": PatchAgg.MAX,
                        "task": {
                            "name": task.name,
                            "source_col": task.source_col,
                            "groups": task.groups,
                        },
                        "cls": DecisionTree(max_depth=max_depth),
                    })

    for _layer, ids in in1k_run_ids.items():
        for run_id in ids:
            for max_depth in max_depths:
                cfgs.append({
                    "run": os.path.join(run_root, run_id),
                    "train_shards": dinov3_ade20k_train,
                    "test_shards": dinov3_ade20k_val,
                    "patch_agg": PatchAgg.MAX,
                    "task": {
                        "name": ade20k_task.name,
                        "source_col": ade20k_task.source_col,
                        "groups": ade20k_task.groups,
                    },
                    "cls": DecisionTree(max_depth=max_depth),
                })

    return cfgs
