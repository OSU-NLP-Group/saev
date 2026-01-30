"""
Sweep file for visual generation jobs for Cambridge butterfly SAE checkpoints.

Usage:
    uv run python contrib/trait_discovery/scripts/launch.py visuals \
        --sweep contrib/trait_discovery/sweeps/007_cambridge_butterflies/visuals.py \
        --slurm-acct PAS2136 --slurm-partition nextgen --n-hours 2
"""


def make_cfgs() -> list[dict]:
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    # Cambridge butterfly shards (DINOv3 ViT-L-16, prefer-fg)
    # v1.0 shards
    shards_v10 = {
        256: "/fs/scratch/PAS2136/samuelstevens/saev/shards/592615d0",
        384: "/fs/scratch/PAS2136/samuelstevens/saev/shards/bf629dcb",
        512: "/fs/scratch/PAS2136/samuelstevens/saev/shards/efbdb4d2",
        640: "/fs/scratch/PAS2136/samuelstevens/saev/shards/69e6d6fd",
    }
    # v1.2 shards
    shards_v12 = {
        384: "/fs/scratch/PAS2136/samuelstevens/saev/shards/7c2ba646",
        640: "/fs/scratch/PAS2136/samuelstevens/saev/shards/71ba8292",
    }

    # Pareto-optimal run IDs by (n_patches, layer, version)
    # v1.0 runs (Pareto-optimal from notebook)
    run_ids_v10 = {
        (256, 21): ["v04m3tc9", "0wxbpz03", "1s86of18", "eubhjiy8", "jdjm31hy"],
        (256, 23): ["42op32o2", "feq16pxe", "knzgya5p", "vhi2238u"],
        (384, 21): ["b3sqicjz", "9wdbh073", "utp8qgq2", "mq8b1zm1", "o2ki3i2k"],
        (384, 23): ["rwdvdgvw", "ct4qhbll", "38wcb0le", "cyz4ktf9", "j59nqapp"],
        (512, 21): ["plsrs8lm", "gc6gailn", "awbqhn90", "vxgrlxbj", "0osrvj4i"],
        (512, 23): ["jiv299mm", "9tkaifc5", "1g9yj7v0", "cyojv8el", "ib65fepe"],
        (640, 21): ["g87a1mbq", "2agzx7an", "kp67vgir", "z5gx0mdt", "pguk1eps"],
        (640, 23): ["qtq2yjir", "ymk1lxo3", "si7wxssf", "yow1j0du", "l3h554l2"],
    }
    # v1.2 runs (Pareto-optimal from notebook)
    run_ids_v12 = {
        (384, 21): ["j93k4u2g", "hxyh41r1", "hdugk4nc", "7mp71fbo", "fmaplh5x"],
        (384, 23): ["77snhj2n", "5e4dt1x6", "frm6bg6o", "vttegf5e", "om59gfwg"],
        (640, 21): ["75asjb3v", "mz1dsdbe", "3bo0k2mh", "qwaxl0tb", "urjt69gb"],
        (640, 23): ["7cmnd5ib", "ez4ntgik", "ujj22ci0", "ey7aqcqi", "cqwk6eoo"],
    }

    # First 30 latents + 10 random
    latents = list(range(30))

    cfgs = []

    # v1.0 configs (commented out - already done)
    # for (n_patches, _layer), ids in run_ids_v10.items():
    #     for run_id in ids:
    #         cfgs.append({
    #             "run": f"{run_root}/{run_id}",
    #             "shards": shards_v10[n_patches],
    #             "latents": latents,
    #             "n_latents": 10,
    #             "ignore_labels": [0],
    #             "device": "cuda",
    #         })

    # v1.2 configs
    for (n_patches, _layer), ids in run_ids_v12.items():
        for run_id in ids:
            cfgs.append({
                "run": f"{run_root}/{run_id}",
                "shards": shards_v12[n_patches],
                "latents": latents,
                "n_latents": 10,
                "ignore_labels": [0],
                "device": "cuda",
            })

    return cfgs
