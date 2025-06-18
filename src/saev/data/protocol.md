# SAEV Sharded-Activation File Protocol v1 (2025-06-17)

saev caches activations to disk rather than run ViT or LLM inference when training SAEs.
Gemma Scope makes this decision as well (see Section 3.3.2 of https://arxiv.org/pdf/2408.05147).
`saev.data` has a specific protocol to support this in on [OSC](https://www.osc.edu), a super computer center, and take advantage of OSC's specific disk performance. 

Goal: loss-lessly persist very large Transformer (ViT or LLM) activations in a form that is:

* mem-mappable
* Parameterized solely by the *experiment configuration* (`writers.Config`)
* Referenced by a content-hash, so identical configs collide, divergent ones never do
* Can be read quickly in a random order for training, and can be read (slowly) with random-access for visuals.

This document is the single normative source. Any divergence in code is a **bug**.

---

## 1. Directory layout

```
<dump_to>/<HASH>/
    metadata.json              # UTF-8 JSON, human-readable, describes data-generating config
    shards.json                # UTF-8 JSON, human-readable, describes shards.
    acts000000.bin             # shard 0
    acts000001.bin             # shard 1
    ...
    actsNNNNNN.bin             # shard NNNNNN  (zero-padded width=6)
```

*`HASH` = `sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode('utf-8'))`*
Guards against silent config drift.

---

## 2. JSON file schemas

### 2.1. `metadata.json`

| field                   | type   | semantic                                     |
| ------------------------| ------ | -------------------------------------------- |
| `vit_family`            | string | `"clip" \| "siglip" \| "dinov2"`             |
| `vit_ckpt`              | string | model identifier (OpenCLIP, HF, etc.)        |
| `layers`                | int[]  | ViT residual‐block indices recorded          |
| `n_patches_per_img`     | int    | **image patches only** (excludes CLS)        |
| `cls_token`             | bool   | `true` -> patch 0 is CLS, else no CLS        |
| `d_vit`                 | int    | activation dimensionality                    |
| `n_imgs`                | int    | total images in dataset                      |
| `max_patches_per_shard` | int    | **logical** activations per shard (see #3)   |
| `data`                  | string | opaque dataset description (`str(cfg.data)`) |
| `dtype` | string | The numpy dtype used. Fixed at float32 for now. |


### 2.2. `shards.json`

A single array of `shard` objects, each of which has the following fields:

| field  | type   | semantic                           |
| ------ | ------ | ---------------------------------- |
| name   | string | shard filename (`acts000000.bin`). |
| n_imgs | int    | the number of images in the shard. |

---

## 3 Shard sizing maths

```python
n_tokens_per_img = n_patches_per_img + (1 if cls_token else 0)

n_imgs_per_shard = floor(max_patches_per_shard / n_tokens_per_img / len(layers))

shape_per_shard = (
    n_imgs_per_shard, len(layers), n_tokens_per_img, d_vit,
)
```

*`max_patches_per_shard` is a **budget** (default ~2.4 M) chosen so a shard is approximately 10 GiB for Float32 @ `d_vit = 1024`.*

*The last shard will have a smaller value for `n_imgs_per_shard`; this value is documented in `n_imgs` in `shards.json`*

---

## 4. Data Layout and Global Indexing

The entire dataset of activations is treated as a single logical 4D tensor with the shape `(n_imgs, len(layers), n_tokens_per_img, d_vit)`. This logical tensor is C-contiguous with axes ordered `[Image, Layer, Token, Dimension]`.

Physically, this tensor is split along the first axis (`Image`) into multiple shards, where each shard is a single binary file. The number of images in each shard is constant, except for the final shard, which may be smaller.

To locate an arbitrary activation vector, a reader must convert a logical coordinate (`global_img_idx`, `layer_value`, `token_idx`) into a file path and an offset within that file.

### 5.1 Definitions

Let the parameters from `metadata.json` be:

* L = `len(layers)`
* P = `n_patches_per_img`
* T = `P + (1 if cls_token else 0)` (Total tokens per image)
* D = `d_vit`
* S = `n_imgs` from `shards.json` or Section 3 (shard sizing).

### 5.2 Coordinate Transformations

Given a logical coordinate:

* `global_img_idx`: integer, with `0 <= global_img_idx < n_imgs`
* `layer`: integer, must be an element of `layers`
* `token_idx`: integer, `0 <= token_idx < T`

The physical location is found as follows:

1.  **Identify Shard:**
    * `shard_idx = global_img_idx // S`
    * `img_in_shard = global_img_idx % S`
    The target file is `acts{shard_idx:06d}.bin`.

2.  **Identify Layer Index:** The stored data contains a subset of the ViT's layers. The logical `layer_value` must be mapped to its index in the stored `layers` array.
    * `layer_idx = layers.index(layer)`
    A reader must raise an error if `layer` is not in `layers`.

3.  **Calculate Offset:** The data within a shard is a 4D tensor of shape `(S, L, T, D)`. The offset to the first byte of the desired activation vector `[img_in_shard, layer_in_list_idx, token_idx]` is:
    * `offset_in_vectors = (img_in_shard * L * T) + (layer_in_list_idx * T) + token_idx`
    * `offset_in_bytes = offset_in_vectors * D * 4` (assuming 4 bytes for `float32`)

A reader can then seek to `offset_in_bytes` and read $D \times 4$ bytes to retrieve the vector.

*Alternatively, rather than calculate the offset, readers can memmap the shard, then use Numpy indexing to get the activation vector.*

### 5.3 Token Axis Layout

The `token` axis of length $T$ is ordered as follows:
* If `cls_token` is `true`:
    * Index `0`: [CLS] token activation
    * Indices `1` to $P$: Patch token activations
* If `cls_token` is `false`:
    * Indices `0` to $P-1$: Patch token activations

The relative order of patch tokens is preserved exactly as produced by the upstream Vision Transformer.

---

## 6 Versioning & compatibility

* **Major changes** (shape reorder, dtype switch, new required JSON keys) increment the protocol version number at the top of this document and must emit a *breaking* warning in loader code.
* **Minor, backward-compatible additions** (new optional JSON key) merely update this doc.

---

That's the whole deal.
No hidden invariants.
Anything else you find in code that contradicts this sheet, fix the code or update the spec.

