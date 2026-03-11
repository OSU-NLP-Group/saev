def make_cfgs() -> list[dict]:
    import os.path
    import typing as tp

    from tdiscovery.classification import LabelGrouping, PatchAgg

    class VersionSpec(tp.TypedDict):
        version: str
        shards_by_n_patches: dict[int, str]
        run_ids_by_layer: dict[int, list[str]]

    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

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
    views = ["dorsal", "ventral"]
    tasks = [
        LabelGrouping(
            name=f"{erato_ssp}_{view}_vs_{melp_ssp}_{view}",
            source_col="subspecies_view",
            groups={
                "erato": [f"{erato_ssp}_{view}"],
                "melpomene": [f"{melp_ssp}_{view}"],
            },
        )
        for erato_ssp, melp_ssp in mimic_pairs
        for view in views
    ]

    # Append a new dict here when v1.7 shards/runs are ready.
    versions: list[VersionSpec] = [
        {
            "version": "v1.6",
            "shards_by_n_patches": {
                640: "/fs/scratch/PAS2136/samuelstevens/saev/shards/79239bdd"
            },
            "run_ids_by_layer": {
                21: ["zhul9opa", "gz2dikb3", "3rqci2h1", "r27w7pmf", "x4n29kua"],
                23: ["pnsi8yhe", "onqqe859", "rd8wc24d", "vends70d", "pa5cu0mf"],
            },
        }
    ]

    cfgs = []
    C_values = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    for spec in versions:
        for n_patches, shards_dpath in spec["shards_by_n_patches"].items():
            for layer, run_ids in spec["run_ids_by_layer"].items():
                for run_id in run_ids:
                    for C in C_values:
                        for task in tasks:
                            cfgs.append({
                                "run": os.path.join(run_root, run_id),
                                "train_shards": shards_dpath,
                                "test_shards": shards_dpath,
                                "patch_agg": PatchAgg.MAX,
                                "task": {
                                    "name": task.name,
                                    "source_col": task.source_col,
                                    "groups": task.groups,
                                },
                                "cls": {"C": C},
                            })

    return cfgs
