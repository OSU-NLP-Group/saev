# 001-heliconius

Local render sweep for Heliconius mimic-pair tasks.

## Run

```bash
uv run launch.py render --sweep exps/001-heliconius/render.py
```

## Dry Run

```bash
uv run python contrib/mimics/launch.py render --sweep contrib/mimics/exps/001-heliconius/render.py --dry-run
```

## Consistency

```bash
uv run python contrib/mimics/launch.py consistency --task-filter dorsal
```

Or score only one run:

```bash
uv run python contrib/mimics/launch.py consistency --run-ids 3rqci2h1
```

## Viewer

```bash
uv run marimo edit contrib/mimics/notebooks/viewer.py
```

## Notes

- The sweep defines one config per task name (`pair x view`).
- Use CLI args to override the sweep defaults (`--top-k-ckpts`, `--max-pooled-features`, etc.).
