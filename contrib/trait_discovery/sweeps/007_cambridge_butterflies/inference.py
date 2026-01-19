def make_cfgs():
    import os.path

    cfgs = []
    run_root = "/fs/ess/PAS2136/samuelstevens/saev/runs"

    # Cambridge butterfly shards (DINOv3 ViT-L-16, prefer-fg)
    shards = {
        256: "/fs/scratch/PAS2136/samuelstevens/saev/shards/592615d0",
        384: "/fs/scratch/PAS2136/samuelstevens/saev/shards/bf629dcb",
        512: "/fs/scratch/PAS2136/samuelstevens/saev/shards/efbdb4d2",
        640: "/fs/scratch/PAS2136/samuelstevens/saev/shards/69e6d6fd",
    }

    # Pareto-optimal run IDs by (n_patches, layer)
    run_ids = {
        (256, 21): ["v04m3tc9", "0wxbpz03", "1s86of18", "eubhjiy8", "jdjm31hy"],
        (256, 23): ["42op32o2", "feq16pxe", "knzgya5p", "vhi2238u"],
        (384, 21): ["b3sqicjz", "9wdbh073", "utp8qgq2", "mq8b1zm1", "o2ki3i2k"],
        (384, 23): ["rwdvdgvw", "ct4qhbll", "38wcb0le", "cyz4ktf9", "j59nqapp"],
        (512, 21): ["plsrs8lm", "gc6gailn", "awbqhn90", "vxgrlxbj", "0osrvj4i"],
        (512, 23): ["jiv299mm", "9tkaifc5", "1g9yj7v0", "cyojv8el", "ib65fepe"],
        (640, 21): ["wrqs99c3", "irjpor5y", "trqdvdzo", "1s9dmofh", "1scfjhq4"],
        (640, 23): ["a5ahw0m7", "k7560pnu", "si7wxssf", "yow1j0du", "l3h554l2"],
    }

    for (n_patches, layer), ids in run_ids.items():
        for id in ids:
            cfgs.append({
                "run": os.path.join(run_root, id),
                "data": {"shards": shards[n_patches], "layer": layer},
            })

    return cfgs
