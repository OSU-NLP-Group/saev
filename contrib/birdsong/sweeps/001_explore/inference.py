def make_cfgs() -> list[dict]:
    import os.path

    runs_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"
    bird_mae_large_birdclef = "/fs/scratch/PAS2136/samuelstevens/saev/shards/5e37a03c"

    # Paste run IDs here
    run_ids = {
        13: ["iwud7hc6", "jhq9go80", "fckf537x", "5dc12bfq", "12ojkfal"],
        15: ["qpogdqi8", "36kn5p1b", "sl71whfz", "rg01sk49", "5rpabyz3"],
        17: ["ew1x0tvd", "5o45mgda", "owf14062", "yj3dxky7", "vze93jo7"],
        19: ["qd4s9puz", "j0x7a7go", "snz0qpn6", "2y3x4yk7", "psjeodyq"],
        21: ["lrtcvr9z", "o3hztzhd", "x5m3a1pw", "xako4v6f", "4b7xvfr2"],
        23: ["xgsn4zsp", "ywstcv8f", "kxqtrt0s", "0y6vhggq", "11eeb71a"],
    }

    cfgs = []
    for layer, ids in run_ids.items():
        for id in ids:
            cfgs.append({
                "run": os.path.join(runs_root, id),
                "data": {"shards": bird_mae_large_birdclef, "layer": layer},
            })
    return cfgs
