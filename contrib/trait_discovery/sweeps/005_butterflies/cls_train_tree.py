def make_cfgs():
    import os.path

    from tdiscovery.classification import DecisionTree, LabelGrouping, PatchAgg

    cfgs = []
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    butterflies_shards = "/fs/scratch/PAS2136/samuelstevens/saev/shards/8bb48fc6"

    # SAE runs: layer -> [run_ids]
    run_ids = {
        13: ["k857x9vw", "7r6cnknt", "y7alnu4l", "19n9kpt1", "86tfnhvm"],
        15: ["asw9j64g", "vfmfdq8x", "nmtt7r1w", "ldlog2fk", "gi5ycyok"],
        17: ["il1z8faw", "8xx144hf", "nmr4b7gt", "9dqg7qk2", "evhi174e"],
        19: ["kxvk6pn0", "hzce99jt", "6jbvhosl", "o2faiso0", "e8vqm83t"],
        21: ["047mline", "9cx2x0tz", "ur0y9l04", "fxd7l9dk", "guebah9r"],
        23: ["o0q33ojt", "0qzdqnw0", "s503zwer", "ad8x5nj6", "1q66exeb"],
    }

    mimic_pairs = [
        ("lativitta", "malleti"),
        ("cyrbia", "cythera"),
        ("notabilis", "plesseni"),
        ("hydara", "melpomene"),
        ("venus", "vulcanus"),
        ("demophoon", "rosina"),
        ("phyllis", "nanna"),
        ("erato", "thelxiopeia"),
    ]

    tasks = []
    for erato_ssp, melp_ssp in mimic_pairs:
        tasks.append(
            LabelGrouping(
                name=f"{erato_ssp}_vs_{melp_ssp}",
                source_col="class",
                groups={
                    "erato": [f"Heliconius erato ssp. {erato_ssp}"],
                    "melpomene": [f"Heliconius melpomene ssp. {melp_ssp}"],
                },
            )
        )

    for layer, ids in run_ids.items():
        for run_id in ids:
            for agg in [PatchAgg.MAX]:
                for max_depth in [1, 2, 3, 5]:
                    for task in tasks:
                        cfgs.append({
                            "run": os.path.join(run_root, run_id),
                            "train_shards": butterflies_shards,
                            "test_shards": butterflies_shards,
                            "patch_agg": agg,
                            "task": {
                                "name": task.name,
                                "source_col": task.source_col,
                                "groups": task.groups,
                            },
                            "cls": DecisionTree(max_depth=max_depth),
                        })

    return cfgs
