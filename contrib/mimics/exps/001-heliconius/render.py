"""Sweep configs for Heliconius mimic-pair rendering."""


def make_cfgs() -> list[dict]:
    run_ids = ["zhul9opa", "gz2dikb3", "3rqci2h1", "r27w7pmf", "x4n29kua"]

    pair_specs = [
        "lativitta:malleti",
        "cyrbia:cythera",
        "notabilis:plesseni",
        "hydara:melpomene",
        "venus:vulcanus",
        "demophoon:rosina",
        "phyllis:nanna",
        "erato:thelxiopeia",
    ]
    views = ["dorsal", "ventral"]

    task_names = []
    for pair_spec in pair_specs:
        erato_ssp, melp_ssp = pair_spec.split(":", maxsplit=1)
        for view in views:
            task_names.append(f"{erato_ssp}_{view}_vs_{melp_ssp}_{view}")

    base = {
        "run_ids": run_ids,
        "top_k_ckpts": 20,
        "n_features_min": 1,
        "n_features_max": 30,
        "n_per_class": 8,
        "feature_chunk_size": 64,
    }

    cfgs = []
    for task_name in task_names:
        cfgs.append({
            **base,
            "task_name": task_name,
        })
    return cfgs
