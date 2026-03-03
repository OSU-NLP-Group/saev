"""
Sweep file for PE-core probe1d metrics jobs on ADE20K shards.

Usage:
    uv run python contrib/trait_discovery/scripts/launch.py metrics \
        --sweep contrib/trait_discovery/sweeps/009_pe_core/probe1d_metrics.py \
        --slurm-acct PAS2136 \
        --slurm-partition nextgen \
        --n-hours 6 \
        --mem-gb 320
"""


def make_cfgs() -> list[dict]:
    import os.path

    cfgs = []
    run_root_dpath = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    # PE-core ADE20K shards
    pe_core_ade20k_train_dpath = (
        "/fs/scratch/PAS2136/samuelstevens/saev/shards/fa2b7ff0"
    )
    pe_core_ade20k_val_dpath = "/fs/scratch/PAS2136/samuelstevens/saev/shards/80219cbf"

    # PE-core Pareto checkpoint IDs (from 009_pe_core notebook)
    pe_core_run_ids: dict[int, list[str]] = {
        21: ["6ed9ojrt", "ang7atm3", "ogpjtuij", "xq1zfqh1", "9u9ny8nm"],
        23: ["h4gy7fke", "ywydn3z5", "omk5qhxf", "f3a9b41q", "r69kzt74"],
    }

    for _layer, run_ids in pe_core_run_ids.items():
        for run_id in run_ids:
            cfgs.append({
                "run": os.path.join(run_root_dpath, run_id),
                "train_shards": pe_core_ade20k_train_dpath,
                "test_shards": pe_core_ade20k_val_dpath,
            })

    return cfgs
