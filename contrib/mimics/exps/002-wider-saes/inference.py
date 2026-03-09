"""Inference for Pareto-optimal 16K/32K SAEs on Cambridge butterflies (384p, v1.6)."""


def make_cfgs():
    import os.path

    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    shards_384 = "/fs/scratch/PAS2136/samuelstevens/saev/shards/a6be28a1"

    # Pareto-optimal run IDs from notebook.py, keyed by (layer, d_sae).
    run_ids = {
        (21, 16384): ["u9i54ybv", "4gi47eoy", "iifhcp9z", "1pwpq6ue"],
        (21, 32768): ["oqm8s8sq", "zgsifqrc", "kmlavddy", "qohobmx1"],
        (23, 16384): ["uu4l9a1c", "qpsn8p39", "yo4x94aj", "nn46e0xn"],
        (23, 32768): ["7yhyyxim", "7ywv33hl", "37yshx8v", "d9wkqlds"],
    }

    cfgs = []
    for (layer, _), ids in run_ids.items():
        for run_id in ids:
            cfgs.append({
                "run": os.path.join(run_root, run_id),
                "data": {"shards": shards_384, "layer": layer},
            })
    return cfgs
