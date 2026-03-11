# Missing Local Runs (010_eccv)

This note tracks run IDs that exist in W&B but are missing under the local runs root:

- expected root: `/fs/ess/PAS2136/samuelstevens/saev/runs`
- checked sweep: `matryoshka_relu_nmse.py`

## Missing IDs

- `cqu1diye`
- `m1zmqln0`

## Current status

- Both run IDs exist in W&B (`samuelstevens/saev`) and are `finished`.
- Both are currently commented out in `matryoshka_relu_nmse.py` so the sweep can submit cleanly.
- The submitted array therefore includes 21 jobs instead of 23.

## Follow-up

- If these run directories are restored/synced to the local runs root, uncomment the two IDs in `matryoshka_relu_nmse.py` and resubmit the sweep.
