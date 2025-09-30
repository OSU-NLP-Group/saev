# Storage & Run Manifest Spec (v1)

There are two main locations:

1. `$SAEV_SCRATCH/saev`: where we store transformer activations.
2. `$SAEV_NFS/saev`: where we store checkpoints and other computed intermediate stuff like example images, probe1d results, etc.

Visually, these are:

```
$SAEV_SCRATCH/saev/
  shards/
    <shard_hash>/
      metadata.json
      shards.json
      acts000000.bin
      acts000001.bin
      ...
      labels.bin
```

and

```
$SAEV_NFS/saev/
  runs/
    <run_id>/
      checkpoint/           # output of train.py on <shard_hash>
        sae.pt
        config.json
      links/                # Symlinks
        shards              # $SCRATCH/saev/shards/<shard_hash>
        dataset             # Whatever the original image dataset was
      inference/            # outputs from dump.py
        <shard_hash>/
          config.json
          patch_acts.npz
          visuals/          # output of visuals.py
```

Each `$SAEV_SCRATCH/shards/<shard_hash>/` MUST include:

* `metadata.json` (UTF-8, canonical spec; see `protocol.md`)
* `shards.json` (UTF-8, shard index and sizes; see `protocol.md`)
* `acts*.bin` (binary shards; format in `protocol.md`)
* `labels.bin` (binary patch labels aligned to shards; format in `protocol.md`)

!!! note
    **Immutability:** Files under `saev/shards/<shard_hash>/` MUST be treated as read-only after publication. Any change yields a new `shard_hash`.

All CLI entrypoints should accept a single `--run <path>` argument.
Every other path MUST be resolved from the run root:

* ViT activations: `links/shards` &rarr; `saev/shards/<shard_hash>`
* Dataset: `links/dataset` &rarr; Dataset root, wherever it is on disk.
* SAE checkpoint: `checkpoint/sae.pt`

**Example resolution:**

```python
run = pathlib.Path(cfg.run)
shards_root = (run / "links" / "shards").resolve()
dataset_root = (run / "links" / "dataset").resolve()
ckpt = run / "checkpoint" / "sae.pt"
labels = vit_root / "labels.bin"
```

* `$SAEV_SCRATCH` and `$SAEV_NFS` should be set for all users/processes running saev tools.

## FAQs

* **Where do patch labels live?** Next to `acts*.bin` in `$SAEV_SCRATCH/shards/<shard_hash>/labels.bin`. Scripts discover them via `links/shards/labels.bin`.

* **Can I put datasets directly in `$SAEV_SCRATCH`?** Sure, but not in `$SAEV_SCRATCH/shards`.

<!-- * `saev vit index` &rarr; computes `metadata.json`, `vit_hash`, writes `shards.json`. -->
<!-- * `saev run init --vit <vit_hash> --dataset butterflies@v0 --run-id 6wnspewc` &rarr; scaffolds a run (creates `links/*`, `manifest.toml`). -->
<!-- * `saev cache stage vit <vit_hash> [--to /scratch/... ]` &rarr; rsyncs and retargets `links/vit`. -->
<!-- * `saev run doctor <run_dir>` &rarr; validates symlinks and required files. -->


