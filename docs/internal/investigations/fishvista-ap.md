# 09/01/2025 12PM

How can we batch calculating average precision across a huge training set? Is that possible? Or not?

Specifically, I have 32K possible linear probes. I have 10 classes in my classification problem, and 2.7M examples. I want to pick the best linear probe for each of the 10 classes by using average precision. So I basically have:
```
scores_nk: Float[Tensor, "n k"]
labels_nc: Int[Tensor, "n c"]
```
Where n = 2.7M, c = 10, k = 32K.

And I want a final pair of tensors of shape c = 10, the best AP for each class c_i, and the index of the best prototype for each class c_i.

How can I do this without blowing up my computer? I prefer simplicity over complexity, and am willing to accept some tradeoffs in precision/exactness in order to maintain simplicity.

# 09/01/2025 1PM

I don't ever plan to use 2.7M examples to pick the optimal linear probe/prototype via AP. Instead, I want to sample ~100K samples.

However, sampling 100K is pretty hard. Look at this function in eval_fishvista.py:

```py
def compute_patch_scores(
    scorer: baselines.Scorer,
    acts_dl: saev.data.OrderedDataLoader,
    imgs_dl: torch.utils.data.DataLoader,
    device: str,
) -> tuple[Float[Tensor, "N K"], Int[Tensor, " N"]]:
    """
    Compute prototype scores for all patches in the dataset.

    Returns:
        scores: Prototype scores for each patch (N images × P patches × K prototypes)
        labels: Ground truth segmentation labels for each patch (N images × P patches)
    """
    n_patches = acts_dl.metadata.n_imgs * acts_dl.metadata.n_patches_per_img
    scores_nk = torch.full((n_patches, scorer.n_prototypes), -torch.inf)
    labels_n = torch.full((n_patches,), -1, dtype=int)

    scorer = scorer.to(device)
    scorer.eval()

    for acts, imgs in zip(
        saev.helpers.progress(acts_dl, desc="Computing scores"), imgs_dl
    ):
        image_i_b = imgs["index"].repeat_interleave(acts_dl.metadata.n_patches_per_img)
        assert (image_i_b == acts["image_i"]).all()

        patch_i_b = torch.arange(acts_dl.metadata.n_patches_per_img).repeat(
            len(imgs["index"])
        )
        assert (patch_i_b == acts["patch_i"]).all()

        acts_bd = acts["act"].to(device)
        # Open a fresh jaxtyping dynamic shape context so B rebinds per batch; without this, the outer @jaxtyped pins B from earlier iterations (e.g., 16000) and a smaller final batch (e.g., 10240) triggers a shape violation.
        with jaxtyped("context"):
            scores_bk = scorer(acts_bd)

        i_b = image_i_b * acts_dl.metadata.n_patches_per_img + patch_i_b
        scores_nk[i_b] = scores_bk
        labels_n[i_b] = einops.rearrange(
            imgs["patch_labels"], "imgs patches -> (imgs patches)"
        )

    return scores_nk, labels_n
```

Some of the challenges include setting up the acts_dl and the imgs_dl so that they are always matching. I could use another reservoir buffer strategy, where we always iterate through the entire 2.7M samples (cheap from a time perspective), but only ever keep at most 100K samples.

Gather some context and recommend a course of action.

# 09/01/2025 1:30PM

We can avoid hashing or reservoir logic entirely and get an exact‐size, deterministic subsample with a seeded `randperm`. Precompute a global boolean mask over all patches once, and index it inside the existing loop. This keeps `acts_dl` and `imgs_dl` perfectly aligned and works across any number of passes.

Sketch:

```py
import torch

def make_keep_mask(n_imgs: int, n_patches_per_img: int, k_keep: int, seed: int) -> torch.Tensor:
    """Return a boolean mask of length N that keeps exactly k indices, uniformly without replacement, deterministically given `seed`. N = n_imgs * n_patches_per_img.

    Memory: N=2.7M ⇒ ~2.7 MB as torch.bool.
    """
    n_total = n_imgs * n_patches_per_img
    assert 0 < k_keep <= n_total
    g = torch.Generator(device="cpu").manual_seed(seed)
    keep_idx_k = torch.randperm(n_total, generator=g)[:k_keep]
    keep_mask_n = torch.zeros(n_total, dtype=torch.bool)
    keep_mask_n[keep_idx_k] = True
    return keep_mask_n

# Precompute once per experiment (and optionally save to disk to reuse across runs)
# Example:
# keep_mask_n = make_keep_mask(acts_dl.metadata.n_imgs, acts_dl.metadata.n_patches_per_img, k_keep=100_000, seed=0xC0FFEE)
# torch.save(keep_mask_n, "cache/fishvista_keep_mask.pt")

# Then, inside the existing loop in eval_fishvista.py:
for acts, imgs in zip(saev.helpers.progress(acts_dl, desc="Computing scores"), imgs_dl):
    image_i_b = imgs["index"].repeat_interleave(acts_dl.metadata.n_patches_per_img)
    assert (image_i_b == acts["image_i"]).all()

    patch_i_b = torch.arange(acts_dl.metadata.n_patches_per_img).repeat(len(imgs["index"]))
    assert (patch_i_b == acts["patch_i"]).all()

    # Flattened patch index i_b ∈ [0, N)
    i_b = image_i_b * acts_dl.metadata.n_patches_per_img + patch_i_b

    # Deterministic, exact subsample
    keep_b = keep_mask_n[i_b]
    if not keep_b.any():
        continue

    # Only score kept patches
    acts_bd = acts["act"][keep_b].to(device)
    with jaxtyped("context"):
        scores_bk = scorer(acts_bd)

    # Downstream: either accumulate per‐prototype stats or write into preallocated arrays at positions i_b[keep_b]
    i_keep_b = i_b[keep_b]
    scores_nk[i_keep_b] = scores_bk
    labels_n[i_keep_b] = einops.rearrange(imgs["patch_labels"], "imgs patches -> (imgs patches)")[keep_b]
```

Notes

- Exactness: `randperm` yields a uniform sample without replacement of exact size `k_keep` given a fixed `seed`.
- Reproducibility: Store `keep_mask_n` to disk and reload to reuse across stages/passes.
- Simplicity: All decisions are local to a batch via `keep_b = keep_mask_n[i_b]`; no global state or thresholding.
- Memory: For N ≈ 2.7M, the mask is ~2.7 MB; the index vector `keep_idx_k` is ~0.8 MB for 100k (int64). Either is fine; the mask gives O(1) batch checks.
