# Special Token Dataloaders

## Problem

`IndexedDataset` already supports `Config.tokens = "special"` and returns one activation per example for the CLS token. `OrderedDataLoader` and `ShuffledDataLoader` still reject that mode, even though the public docs and downstream plans already assume CLS-only loading is available.

This blocks a simple workflow for training or analyzing models on only special tokens, for example training an SAE on CLS activations.

## Goal

Add `tokens = "special"` support to the ordered and shuffled activation loaders for a fixed transformer layer.

## Non-goals

- No `tokens = "all"` support in either loader.
- No support for `layer = "all"` in either loader.
- No patch-label filtering for special tokens.
- No change to on-disk shard layout or indexing semantics.

## Requirements

### Functional

1. `OrderedConfig.tokens` must accept `"special"` as well as `"content"`.

2. `OrderedDataLoader` and `ShuffledDataLoader` must both accept:
   - `tokens = "special"`
   - `layer = <int>` where the layer is present in metadata

3. In special-token mode, each yielded sample corresponds to exactly one example:
   - `example_idx` is the example index
   - `token_idx` is `-1`
   - `act` is the activation stored at token position `0` in the shard

4. Epoch sizes in special-token mode:
   - `n_samples == metadata.n_examples`
   - `len(loader)` follows the existing `batch_size` and `drop_last` logic

5. Ordered loader ordering in special-token mode:
   - samples are yielded in increasing `example_idx`
   - `token_idx` is always `-1`

6. Shuffled loader semantics in special-token mode:
   - each example appears once per epoch
   - order remains deterministic for a fixed seed
   - batches still expose the same keys: `act`, `example_idx`, `token_idx`

7. Token labels:
   - ordered loader must not attach `token_labels` for special tokens, even if `labels.bin` exists
   - shuffled loader must reject `ignore_labels` when `tokens != "content"`

### Non-functional

1. Reuse the existing shard protocol. Special-token mode must read token position `0` from each example when `metadata.cls_token` is true.

2. Keep the implementation small. The existing content-token path should remain unchanged except where a shared branch is cleaner than duplicate code.

3. Preserve the current meaning of `token_idx = -1` for special tokens so loader outputs match `IndexedDataset` and `shards.IndexMap`.

## Design

### Ordered loader

Continue to use `shards.IndexMap` to translate a global sample index into a shard location. This already knows that special tokens map to:

- `content_token_idx = -1`
- `token_idx_in_shard = 0`

The ordered manager only needs two behavior changes:

1. permit `tokens = "special"` in the fixed-layer path
2. skip label lookup when `content_token_idx < 0`

### Shuffled loader

The shuffled loader currently iterates over every content token in a shard chunk. In special-token mode it should instead emit exactly one activation per example in the chunk:

- activation source: `mmap[start:end, layer_i, 0]`
- metadata:
  - column 0: global example indices
  - column 1: `-1`

`ignore_labels` remains content-token-only because `labels.bin` is defined over content tokens.

## Test Plan

Add red tests before implementation:

1. Ordered loader special-token smoke test on fake shards:
   - batch iterates successfully
   - all `token_idx == -1`
   - first batch has sequential `example_idx`

2. Ordered loader matches `IndexedDataset` in special-token mode.

3. Shuffled loader special-token epoch test on fake shards:
   - all `token_idx == -1`
   - every example appears exactly once in a full epoch
   - activations match `IndexedDataset` for the sampled `example_idx`

4. Shuffled loader rejects `ignore_labels` when `tokens = "special"`.

## Acceptance Criteria

- The new tests fail before the implementation change.
- The new tests pass after the implementation change.
- Existing content-token tests continue to pass.
