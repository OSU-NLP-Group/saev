"""Sweep configs for Cambridge mimic-pair rendering.

Usage:
    uv run python contrib/mimics/launch.py render --sweep contrib/mimics/sweeps/render_cambridge_pairs.py
"""


def make_cfgs() -> list[dict]:
    run_ids = ["zhul9opa", "gz2dikb3", "3rqci2h1", "r27w7pmf", "x4n29kua"]
    base = {
        "run_ids": run_ids,
        "top_k_ckpts": 20,
        "n_features_min": 1,
        "n_features_max": 30,
        "n_per_class": 8,
    }

    task_names = [
        "lativitta_dorsal_vs_malleti_dorsal",
        "lativitta_ventral_vs_malleti_ventral",
        "cyrbia_dorsal_vs_cythera_dorsal",
        "cyrbia_ventral_vs_cythera_ventral",
    ]

    cfgs = []
    for task_name in task_names:
        cfgs.append({
            **base,
            "task_name": task_name,
        })
    return cfgs
