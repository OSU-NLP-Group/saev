# Performance

SAEs are mostly disk-bound.
Gemma Scope (Google SAE paper) aimed for 1 GB/s to keep their GPUS brrr'ing.
This is pretty hard even with sequential reads, much less random access.

I run all my experiments on [OSC](https://www.osc.edu/) and their scratch filesystem `/fs/scratch` has sequential read speeds of around 800 MB/s and random access speeds around 22 MB/s.

I got these numbers with:

```sh
fio --name=net --filename=/fs/scratch/PAS2136/samuelstevens/cache/saev/366017a10220b85014ae0a594276b25f6ea3d756b74d1d3218da1e34ffcf32e9/acts000000.bin --rw=read --bs=1M --direct=1 --iodepth=16 --runtime=30 --time_based
```

and

```sh
fio --name=net --filename=/fs/scratch/PAS2136/samuelstevens/cache/saev/366017a10220b85014ae0a594276b25f6ea3d756b74d1d3218da1e34ffcf32e9/acts000000.bin --rw=randread --bs=4K --direct=1 --iodepth=16 --runtime=30 --time_based
```

These two commands reported, respectively:

```
READ: bw=796MiB/s (835MB/s), 796MiB/s-796MiB/s (835MB/s-835MB/s), io=23.3GiB (25.0GB), run=30001-30001msec
```

and

```
READ: bw=22.9MiB/s (24.0MB/s), 22.9MiB/s-22.9MiB/s (24.0MB/s-24.0MB/s), io=687MiB (721MB), run=30001-30001msec
```

My naive pytorch-style dataset that uses multiple processes to feed a dataloader did purely random reads and was too slow.
It reports around 500 examples/s:

![Performance plot showing that naive random access dataloading maxes out around 500 examples/s.](assets/benchmarking/ee86c12134a89ea819b129bcce0d1abbda5143c4/plot.png)

I've implemented a dataloader that tries to do sequential reads rather than random reads in `saev/data/iterable.py`.
It's much faster (around 4.5K examples/s) on OSC.

![Performance plot showing that my first attempt at a sequential dataloader maxes out around 4500 examples/s.](assets/benchmarking/4e9b2faf065ffb21e635633a2ee485bd699b0941/plot.png)

I know that it should be even faster; the dataset of 128M examples is 2.9TB, my sequential disk read speed is 800 MB/s, so it should take ~1 hr.
For 128M examples at 4.5K examples/s, it should take 7.9 hours.
You can see this on a [wandb run here](https://wandb.ai/samuelstevens/saev/runs/okm4fv8j?nw=nwusersamuelstevens&panelDisplayName=Disk+Utilization+%28%25%29&panelSectionName=System) which reports 14.6% disk utilization.
Certainly that can be higher.

> *Not sure if this is the correct way to think about it, but: 100 / 14.6 = 6.8, close to 7.9 hours.*

