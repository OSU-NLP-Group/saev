# saev Sharded Activation File Protocol

saev caches activations to disk rather than run ViT or LLM inference when training SAEs.
Gemma Scope makes this decision as well (see Section 3.3.2 of https://arxiv.org/pdf/2408.05147).
`saev.data` has a specific protocol to support this in on [OSC](https://www.osc.edu), a super computer center, and take advantage of OSC's specific disk performance. 

Goal: loss-lessly persist very large Transformer (ViT or LLM) activations in a form that is:

* mem-mappable
* Parameterized solely by the *experiment configuration* (`scripts/shards.py:Config`)
* Referenced by a content-hash, so identical configs collide, divergent ones never do
* Can be read quickly in a random order for training, and can be read (slowly) with random-access for visuals.

This document is the single normative source. Any divergence in code is a **bug**.

---

## 1. Directory layout

```
<dump_to>/<HASH>/
    metadata.json    # UTF-8 JSON, human-readable, describes data-generating config
    shards.json      # UTF-8 JSON, human-readable, describes shards.
    acts000000.bin   # shard 0
    acts000001.bin   # shard 1
    ...
    actsNNNNNN.bin   # shard NNNNNN  (zero-padded width=6)
    labels.bin       # patch labels (optional)
```

*`HASH` = `sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode('utf-8'))`*
Guards against silent config drift.

---

## 2. JSON file schemas

### 2.1. `metadata.json`

| field                | type   | semantic                                   |
| -------------------- | ------ | ------------------------------------------ |
| `family`             | string | `"clip" \| "siglip" \| "dinov2"`           |
| `ckpt`               | string | model identifier (OpenCLIP, HF, etc.)      |
| `layers`             | int[]  | ViT residual‐block indices recorded        |
| `patches_per_ex`     | int    | **example patches only** (excludes CLS)    |
| `cls_token`          | bool   | `true` -> patch 0 is CLS, else no CLS      |
| `d_model`            | int    | activation dimensionality                  |
| `n_examples`         | int    | total examples in dataset                  |
| `patches_per_shard`  | int    | **logical** activations per shard (see #3) |
| `data`               | object | opaque dataset description                 |
| `dataset`            | string | absolute path to original dataset root     |
| `dtype`              | string | numpy dtype. Fixed `"float32"` for now.    |
| `protocol`           | string | `"2.1"` (shards after big refactor)        |

The `data` object is `base64.b64encode(pickle.dumps(img_ds)).decode('utf8')`.

The `dataset` field stores the absolute path to the root directory of the original image dataset, allowing runs to create symlinks back to the source images for visualization and analysis.

### 2.2. `shards.json`

A single array of `shard` objects, each of which has the following fields:

| field      | type   | semantic                             |
| ---------- | ------ | ------------------------------------ |
| name       | string | shard filename (`acts000000.bin`).   |
| n_examples | int    | the number of examples in the shard. |

---

## 3. Shard sizing maths

```python
tokens_per_ex = patches_per_ex + (1 if cls_token else 0)

examples_per_shard = floor(patches_per_shard / (tokens_per_ex * len(layers)))

shape_per_shard = (
    examples_per_shard, len(layers), tokens_per_ex, d_model,
)
```

*`patches_per_shard` is a **budget** (default ~2.4 M) chosen so a shard is approximately 10 GiB for Float32 @ `d_model = 1024`.*

*The last shard will have a smaller value for `examples_per_shard`; this value is documented in `n_examples` in `shards.json`*

---

## 4. Data Layout and Global Indexing

The entire dataset of activations is treated as a single logical 4D tensor with the shape `(n_examples, len(layers), tokens_per_ex, d_model)`. This logical tensor is C-contiguous with axes ordered `[Example, Layer, Token, Dimension]`.

Physically, this tensor is split along the first axis (`Example`) into multiple shards, where each shard is a single binary file. The number of examples in each shard is constant, except for the final shard, which may be smaller.

To locate an arbitrary activation vector, a reader must convert a logical coordinate (`global_ex_idx`, `layer_value`, `token_idx`) into a file path and an offset within that file.

### 4.1 Definitions

Let the parameters from `metadata.json` be:

* L = `len(layers)`
* P = `patches_per_ex`
* T = `P + (1 if cls_token else 0)` (Total tokens per example)
* D = `d_model`
* S = `n_examples` from `shards.json` or `examples_per_shard` from Section 3 (shard sizing).

### 4.2 Coordinate Transformations

Given a logical coordinate:

* `global_ex_idx`: integer, with `0 <= global_ex_idx < n_examples`
* `layer`: integer, must be an element of `layers`
* `token_idx`: integer, `0 <= token_idx < T`

The physical location is found as follows:

1.  **Identify Shard:**
    * `shard_idx = global_ex_idx // S`
    * `ex_in_shard = global_ex_idx % S`
    The target file is `acts{shard_idx:06d}.bin`.

2.  **Identify Layer Index:** The stored data contains a subset of the ViT's layers. The logical `layer_value` must be mapped to its index in the stored `layers` array.
    * `layer_idx = layers.index(layer)`
    A reader must raise an error if `layer` is not in `layers`.

3.  **Calculate Offset:** The data within a shard is a 4D tensor of shape `(S, L, T, D)`. The offset to the first byte of the desired activation vector `[ex_in_shard, layer_idx , token_idx]` is:
    * `offset_in_vectors = (ex_in_shard * L * T) + (layer_idx * T) + token_idx`
    * `offset_in_bytes = offset_in_vectors * D * 4` (assuming 4 bytes for `float32`)

A reader can then seek to `offset_in_bytes` and read $D \times 4$ bytes to retrieve the vector.

*Alternatively, rather than calculate the offset, readers can memmap the shard, then use Numpy indexing to get the activation vector.*

### 4.3 Token Axis Layout

The `token` axis of length $T$ is ordered as follows:
* If `cls_token` is `true`:
    * Index `0`: [CLS] token activation
    * Indices `1` to $P$: Patch token activations
* If `cls_token` is `false`:
    * Indices `0` to $P-1$: Patch token activations

The relative order of patch tokens is preserved exactly as produced by the upstream Vision Transformer.

---

## 5 Versioning & compatibility

* **Major changes** (shape reorder, dtype switch, new required JSON keys) increment the major protocol version number at the top of this document and must emit a *breaking* warning in loader code.
* **Minor, backward-compatible additions** (new optional JSON key) merely update this doc and the minor protocol version number.

---

That's it.
Anything else you find in code that contradicts this document, fix the code or update the spec.
