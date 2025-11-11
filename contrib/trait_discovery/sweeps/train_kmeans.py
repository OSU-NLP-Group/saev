# uv run scripts/launch.py baseline::train --method kmeans --sweep sweeps/train_kmeans.py --runs-root /fs/ess/PAS2136/samuelstevens/tdiscovery/runs --train-data.shards /fs/scratch/PAS2136/samuelstevens/saev/shards/51567c6c --train-data.min-buffer-fill 0.2 --val-data.shards /fs/scratch/PAS2136/samuelstevens/saev/shards/3e27794f
def make_cfgs() -> list[dict]:
    cfgs = []
    runs_root = "/fs/ess/PAS2136/samuelstevens/tdiscovery/runs"
    dinov3_vitl_in1k_train = "/fs/scratch/PAS2136/samuelstevens/saev/shards/51567c6c"
    dinov3_vitl_in1k_val = "/fs/scratch/PAS2136/samuelstevens/saev/shards/3e27794f"

    for layer in [13, 15, 17, 19, 21, 23]:
        cfgs.append({
            "runs_root": runs_root,
            "train_data": {
                "shards": dinov3_vitl_in1k_train,
                "layer": layer,
                "min_buffer_fill": 0.2,
            },
            "val_data": {"shards": dinov3_vitl_in1k_val, "layer": layer},
        })

    return cfgs
