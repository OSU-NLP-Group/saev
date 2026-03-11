"""
Sweep file for PE-spatial probe1d metrics jobs on ADE20K shards.

Usage:
    uv run python contrib/trait_discovery/scripts/launch.py metrics \
        --sweep contrib/trait_discovery/sweeps/008_pe/probe1d_metrics.py \
        --slurm-acct PAS2136 \
        --slurm-partition nextgen \
        --n-hours 6 \
        --mem-gb 320
"""


def make_cfgs() -> list[dict]:
    import os.path

    cfgs = []
    run_root_dpath = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    # PE-spatial ADE20K shards
    pe_spatial_ade20k_train_dpath = (
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/4b279034"
    )
    pe_spatial_ade20k_val_dpath = (
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/245e4d61"
    )

    # PE-spatial Pareto checkpoint IDs (from 008_pe notebook)
    pe_spatial_run_ids: dict[int, list[str]] = {
        21: [
            "j0hmtzkh",
            "nv55sqo1",
            "3hqhkxqf",
            "sp94dh0t",
            "cqfjwsif",
        ],
        23: [
            "3u4wh1m9",
            "barw9k1k",
            "ecbauxoc",
            "5okeekou",
            "0xt63hu0",
        ],
    }

    for _layer, run_ids in pe_spatial_run_ids.items():
        for run_id in run_ids:
            cfgs.append({
                "run": os.path.join(run_root_dpath, run_id),
                "train_shards": pe_spatial_ade20k_train_dpath,
                "test_shards": pe_spatial_ade20k_val_dpath,
            })

    return cfgs
