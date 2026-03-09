"""Scoring sweep for Pareto-optimal 16K/32K SAEs on Cambridge butterflies (384p, v1.6)."""


def make_cfgs() -> list[dict]:
    # Pareto-optimal run IDs from notebook.py, keyed by (layer, d_sae).
    run_ids = [
        # layer 21, 16K
        "u9i54ybv",
        "4gi47eoy",
        "iifhcp9z",
        "1pwpq6ue",
        # layer 21, 32K
        "oqm8s8sq",
        "zgsifqrc",
        "kmlavddy",
        "qohobmx1",
        # layer 23, 16K
        "uu4l9a1c",
        "qpsn8p39",
        "yo4x94aj",
        "nn46e0xn",
        # layer 23, 32K
        "7yhyyxim",
        "7ywv33hl",
        "37yshx8v",
        "d9wkqlds",
    ]

    return [{"run_ids": run_ids}]
