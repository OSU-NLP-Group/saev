# 05/26/2026

I ran the random vector baseline with 8,192 prototypes on CUB 200.
Overall, the results were stronger than I predicted.
As a reminder, here's the experimental procedure again.

Goal: Evaluate different methods of choosing protoypes in ViT activation space for finding traits in images.

1. Pick a set of protoypes (vectors). These prototypes must satisfy this `Scorer` interface:

```python
class Scorer:
    def __call__(self, activations: Float[Tensor, "B D"]) -> Float[Tensor, "B K"]: ...

    @property
    def n_prototypes(self) -> int: ...
```

2. Choose the best prototype for each trait (312 binary attributes in CUB 200) with the highest average precision (AP) over the ~6K training images. All patches are scored via the above interface, then we take the max score over all patches in an image as an image-level score.
3. Measure each trait-specific prototype's score for each the ~6K test images. From this we can calculate each trait's AP. We can also calculate mean AP (mAP) across all 312 traits.

Using a random set of prototypes led to very strong results.

With 8192 prototypes (1024 x 8), we get a mAP of 0.864.
With 1024 prototypes, we get a mAP of 0.858.
With 128 prototypes, we get a mAP of 0.855.

Color and pattern attributes have very high AP.
Wing shape attributes are low AP.

A possible explanation: colors and patterns are pretty obvious features and might show up in only one patch.
Max-pooling over the image makes it easy.
Shapes require multiple patches, so random patches are unlikely to find the specific direction.
It's a weak explanation but it's what we have so far.

Possible explanations for such strong results:

1. 8192 is a big pool. For only 312 traits, that's a lot of options.
2. 6K training images is enough to prevent overfitting on the train set.
3. The traits are too easy: color is very simple and it gets quite high scores.

We can test hypotheses 1 and 2 quite easily.

|   Prototypes |   Train |      mAP |       5% |      95% |
|--------------|---------|----------|----------|----------|
|          128 |      16 | 0.838336 | 0.584672 | 0.925063 |
|          128 |      64 | 0.84328  | 0.61134  | 0.927155 |
|          128 |     256 | 0.844236 | 0.617318 | 0.92509  |
|          128 |    1024 | 0.849561 | 0.616914 | 0.932712 |
|          128 |    5794 | 0.864806 | 0.660373 | 0.936375 |
|          512 |      16 | 0.835533 | 0.57979  | 0.921742 |
|          512 |      64 | 0.841849 | 0.570539 | 0.929806 |
|          512 |     256 | 0.845989 | 0.602945 | 0.930535 |
|          512 |    1024 | 0.858065 | 0.654299 | 0.937016 |
|          512 |    5794 | 0.865186 | 0.650653 | 0.94162  |
|         2048 |      16 | 0.839093 | 0.574827 | 0.929895 |
|         2048 |      64 | 0.84648  | 0.656736 | 0.920138 |
|         2048 |     256 | 0.850884 | 0.591706 | 0.932471 |
|         2048 |    1024 | 0.859395 | 0.636532 | 0.934989 |
|         2048 |    5794 | 0.867621 | 0.671396 | 0.938671 |
|         8192 |      16 | 0.841456 | 0.63949  | 0.923007 |
|         8192 |      64 | 0.843862 | 0.593469 | 0.937366 |
|         8192 |     256 | 0.841481 | 0.602946 | 0.926301 |
|         8192 |    1024 | 0.864679 | 0.605401 | 0.934542 |

Then we can summarize these results with a chart.

# 06/27/2025

My code for loading traits was very wrong.
Very very wrong.

```py
cub_root = "/fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder"
test_y_true_NT = cub200.data.load_attrs(cub_root, is_train=False).numpy()
n_test, n_traits = test_y_true_NT.shape

const_ap_T = np.array([sklearn.metrics.average_precision_score(test_y_true_NT[:,i], np.zeros(n_test)) for i in range(n_traits)])
print(const_ap_T.mean())  # -> 0.83
```

I fixed this in c489e4a.

With this update, then our table looks like:

|   Prototypes |   Train |      mAP |         5% |      95% |
|--------------|---------|----------|------------|----------|
|          128 |      16 | 0.103469 | 0.00422779 | 0.357801 |
|          128 |      64 | 0.133899 | 0.00410075 | 0.488794 |
|          128 |     256 | 0.169785 | 0.00364236 | 0.530993 |
|          128 |    1024 | 0.153601 | 0.00392851 | 0.458499 |
|          128 |    5794 | 0.150209 | 0.00425978 | 0.48719  |
|          512 |      16 | 0.10293  | 0.00343484 | 0.36127  |
|          512 |      64 | 0.127226 | 0.00347708 | 0.443757 |
|          512 |     256 | 0.167894 | 0.00435084 | 0.537646 |
|          512 |    1024 | 0.181973 | 0.00397084 | 0.535357 |
|          512 |    5794 | 0.206456 | 0.0034702  | 0.525066 |
|         2048 |      16 | 0.106846 | 0.00325554 | 0.332599 |
|         2048 |      64 | 0.15091  | 0.0037288  | 0.533692 |
|         2048 |     256 | 0.186111 | 0.00387836 | 0.535279 |
|         2048 |    1024 | 0.208857 | 0.00446532 | 0.553357 |
|         2048 |    5794 | 0.225127 | 0.00421674 | 0.61467  |
|         8192 |      16 | 0.104258 | 0.00363135 | 0.38363  |
|         8192 |      64 | 0.141801 | 0.00343469 | 0.463187 |
|         8192 |     256 | 0.185463 | 0.00436711 | 0.570816 |
|         8192 |    1024 | 0.223924 | 0.00522286 | 0.591821 |
|         8192 |    5794 | 0.24071  | 0.00356492 | 0.613314 |

Way better.

## To Dos

1. Fill in `baselines.KMeans.train()`
2. Train KMeans checkpoints on iNat21 SigLIP ViT-L/14 activations (layer 23) and save them to a shared location.
3. Fill in `baselines.PCA.train()`.
4. Train PCA checkpoints on iNat21 SigLIP ViT-L/14 activations (layer 23) and save them to a shared location.

# 06/29/2025

I'm going to travel for a couple weeks and I want to make sure I'm taking advantage of the compute hours while I'm sleeping.
So I want to run experiments frequently, with the goal of learning something new, even if the checkpoints are never used.

What do I want to learn?

1. Are there differences in the learned features at different layers?
2. Does ViT scale lead to different learned features?
3. Do SAEs find meaningfully different features in BioCLIP vs other more general models? Or can you replicate it with data, rather than model?

Let's investigate layers.

I started a job with layer 23 and a job with layer 13.

uv run train.py --slurm-acct PAS2136 --slurm-partition nextgen --n-hours 12 --sweep configs/preprint/baseline.toml --data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/f9deaa8a07786087e8071f39a695200ff6713ee02b25e7a7b4a6d5ac1ad968db --data.patches image --data.layer 23 sae:relu --sae.d-vit 1024

uv run train.py --slurm-acct PAS2136 --slurm-partition nextgen --n-hours 12 --sweep configs/preprint/baseline.toml --data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/f9deaa8a07786087e8071f39a695200ff6713ee02b25e7a7b4a6d5ac1ad968db --data.patches image --data.layer 13 sae:relu --sae.d-vit 1024

Only difference is --data.layer.

uv run visuals.py --ckpt checkpoints/53fl3ysv/sae.pt --dump-to /fs/scratch/PAS2136/samuelstevens/saev/visuals/53fl3ysv --log-freq-range -3 -1 --log-value-range -1 3 --data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/f9deaa8a07786087e8071f39a695200ff6713ee02b25e7a7b4a6d5ac1ad968db/ --data.layer 13 images:image-folder --images.root /fs/ess/PAS2136/foundation_model/inat21/raw/train_mini/

However, I need to add slurm to the visuals.py script.

# 07/24/2025

I have successfully added slurm to the visuals.py script.
Now it's time to add some metrics logging so that I can compare the effect of different hyperparameters.


Reconstruction Mean-Squared Error (MSE)

Why: basic sanity check that the decoder is learning to reproduce the ViT activation.

$\text{MSE} = \frac{\lVert z - \hat z \rVert_2^2}{D}$

where $z$ is the original activation, $\hat z$ is the reconstruction, and $D$ is the dimensionality of $z$.

```python
mse = (z_hat - z).pow(2).mean()
```

Explained Variance

Why: scale-invariant measure of reconstruction quality; lets you compare runs with different sparsity penalties.

$\text{ExplVar} = 1 - \frac{\operatorname{Var}(z - \hat z)}{\operatorname{Var}(z)}$

where the variances are computed element-wise over the batch.

```python
expl_var = 1 - (z_hat - z).var() / z.var()
```

Mean L0 (fraction of active units)

Why: tracks sparsity level; you usually aim for 1–5 % active units.

$\text{L0} = \frac{1}{B K}\sum_{b=1}^{B}\sum_{k=1}^{K}\mathbf{1}\!\bigl(|f_{bk}|>\epsilon\bigr)$

with $\epsilon \approx 10^{-8}$.

```python
l0 = (f.abs() > 1e-8).float().mean()
```

Mean L1 (average absolute activation)

Why: complementary sparsity signal; falls as the sparsity penalty λ rises.

$\text{L1} = \frac{1}{B K}\sum_{b=1}^{B}\sum_{k=1}^{K}|f_{bk}|$

```python
l1 = f.abs().mean()
```

Dead-Unit Percentage

Why: measures wasted capacity; too many dead units suggest λ is too high or K is too large.

$\text{DeadPct} = \frac{1}{K}\sum_{k=1}^{K}\mathbf{1}\!\Bigl(\max_{b}|f_{bk}| = 0\Bigr)$

```python
dead_pct = ((f.abs() > 1e-8).sum(0) == 0).float().mean()
```

Mean Absolute Off-Diagonal Correlation

Why: lower values indicate more independent (less redundant) codes.

$\text{MeanAbsCorr} = \frac{1}{K(K-1)}\sum_{i\ne j}\bigl|\rho_{ij}\bigr|$

where $\rho_{ij}$ is the Pearson correlation between units $i$ and $j$.

```python
corr = torch.corrcoef(f.T)
mean_abs_corr = corr.fill_diagonal_(0).abs().mean()
```

Dictionary Coherence

Why: detects duplicate decoder atoms; lower is better.

$$$\text{Coherence} = \max_{i\ne j}\left|\left\langle\hat w_i,\hat w_j\right\rangle\right|,\quad
\hat w_i = \frac{w_i}{\lVert w_i\rVert_2}$$

```python
W = decoder.weight                       # (D, K)
W_norm = W / W.norm(dim=0, keepdim=True) # column-normalise
coherence = (W_norm.T @ W_norm).abs().triu(1).max()
```

Decoder Column ℓ₂ Norm (average)

Why: catches exploding decoder weights, often a sign of bad learning rate or λ.

$$\text{AvgWColNorm} = \frac{1}{K}\sum_{k=1}^{K}\lVert w_k\rVert_2$$

```python
avg_w_col_norm = decoder.weight.norm(dim=0).mean()
```

Gradient Norm

Why: spikes reveal optimisation issues or FP16 under/overflow.

$$\lVert\nabla\theta\rVert_2$$

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
```

# 07/25/2025

I have kicked off a training job for a lot of SAEs across many different layers of SigLIP 2 on iNat21 train-mini (job ID: 1856101).

I'm also going to save activations for DINOv2 and BioCLIP-2.

BioCLIP 2:
```sh
uv run python -m saev.data --slurm-acct PAS2136 --n-hours 48 --slurm-partition nextgen --vit-family clip --vit-ckpt hf-hub:imageomics/bioclip-2 --d-vit 1024 --n-patches-per-img 256 --vit-layers 13 15 17 19 21 23 --dump-to /fs/scratch/PAS2136/samuelstevens/cache/saev/ --max-patches-per-shard 500_000 data:image-folder --data.root /fs/ess/PAS2136/foundation_model/inat21/raw/train_mini/
```
Job ID: 1856237

DINOv2
```sh
uv run python -m saev.data --slurm-acct PAS2136 --n-hours 48 --slurm-partition nextgen --vit-family dinov2 --vit-ckpt dinov2_vitl14_reg --d-vit 1024 --n-patches-per-img 256 --vit-layers 13 15 17 19 21 23 --dump-to /fs/scratch/PAS2136/samuelstevens/cache/saev/ --max-patches-per-shard 500_000 data:image-folder --data.root /fs/ess/PAS2136/foundation_model/inat21/raw/train_mini/
```
Job ID: 1856263

# 07/31/2025

Great. I trained some SAEs on SigLIP2 and DINOv2 models.
I should now:

1. Get some visuals from early and late layers for both models (4x visuals scripts).
2. Merge Jake's code so that I can also train these models with at least BatchTopK which is highly likely to be correct.
