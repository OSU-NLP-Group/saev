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
2. Merge Jake's code so that I can also train these models with at least BatchTopK which is highly likely to be correct. [DONE]

# 08/01/2025

- What do my different metrics mean? I should try to understand them better so that I can build intuition around which checkpoints are better.
- What is the absolute core of my story?

# 08/04/2025

## 8d67ty75 (DINOv2 Layer 23/23)

10612 - Bird breasts
11446 - Berry calyx
1117 - Edge of a flower's stigma, with a bee on it
15206 - Butterfly wings
8297 - Scorpion legs, in both UV and RGB
13358 - sky, but specifically with birds flying?
2645 - shells
4459 - lizard legs
9146 - dragonfly wings
13277 - white mrking on outside of bird wings
6614 - backs of creepy crawly insects
10964 - tops of bird wings
14514 - flower stigmas
15797 - outside border of fungis
9365 - grasshopper anntenae
4735 - flower stems
5473 - bird wings
11846 - white bird breast
1270 - animal ears (mostly mammals, one bird)
5115 - white flower petals
1562 - camera lens
3832 - cactus flowers
13189 - beetle legs
6973 - dolphin and orca heads/noses
4884 - reptile/amphimbian noses
12717 - whiskers, including a catfish's whiskers


## c77k5ay3 (DINOv2 Layer 13/23)

2716 - Animal eyes, including birds and mammals

# 08/06/2025

I need to compare the SigLIP checkpoints again the random vectors baseline.

To do this, I need to:

1. Train SAEs. [done]
2. Fit random vector baselines. [done]
3. Score the SAE checkpoints. [done]
4. Score the random vector baselines. [done]


Unfortunately, the results are not good.

Here are the SAE scores. Instead of varying the number of prototypes, we vary the L0-MSE tradeoff.

| Checkpoint   |   Train |       mAP |         5% |      95% |
|--------------|---------|-----------|------------|----------|
| 72mwb0m4     |      16 | 0.0969196 | 0.00310666 | 0.322744 |
| 72mwb0m4     |      64 | 0.149365  | 0.00310666 | 0.585383 |
| 72mwb0m4     |     256 | 0.190386  | 0.00310666 | 0.589916 |
| 72mwb0m4     |    1024 | 0.228591  | 0.00355049 | 0.606393 |
| 72mwb0m4     |    4096 | 0.244641  | 0.00345185 | 0.606697 |
| 72mwb0m4     |    5794 | 0.245494  | 0.00362444 | 0.606697 |
| 9ml97uhc     |      16 | 0.100118  | 0.00340351 | 0.350569 |
| 9ml97uhc     |      64 | 0.123519  | 0.00385649 | 0.463584 |
| 9ml97uhc     |     256 | 0.182071  | 0.00335379 | 0.593917 |
| 9ml97uhc     |    1024 | 0.223868  | 0.00453283 | 0.597392 |
| 9ml97uhc     |    4096 | 0.240573  | 0.00362444 | 0.602611 |
| 9ml97uhc     |    5794 | 0.24232   | 0.00383253 | 0.602611 |
| n8gs5x83     |      16 | 0.0995626 | 0.00345185 | 0.363675 |
| n8gs5x83     |      64 | 0.126214  | 0.00345185 | 0.422263 |
| n8gs5x83     |     256 | 0.169573  | 0.00352377 | 0.581443 |
| n8gs5x83     |    1024 | 0.21915   | 0.0041529  | 0.601269 |
| n8gs5x83     |    4096 | 0.233236  | 0.00441054 | 0.601269 |
| n8gs5x83     |    5794 | 0.234051  | 0.004107   | 0.601269 |
| ssnkul19     |      16 | 0.0998286 | 0.00310666 | 0.339477 |
| ssnkul19     |      64 | 0.134707  | 0.00345185 | 0.522674 |
| ssnkul19     |     256 | 0.192325  | 0.00310666 | 0.597629 |
| ssnkul19     |    1024 | 0.229777  | 0.00335085 | 0.620064 |
| ssnkul19     |    4096 | 0.246071  | 0.0037407  | 0.620155 |
| ssnkul19     |    5794 | 0.248653  | 0.00379703 | 0.620155 |
| u6age4v9     |      16 | 0.0963895 | 0.00310666 | 0.326862 |
| u6age4v9     |      64 | 0.12535   | 0.00310666 | 0.453362 |
| u6age4v9     |     256 | 0.172496  | 0.00321293 | 0.571335 |
| u6age4v9     |    1024 | 0.217222  | 0.00346826 | 0.590206 |
| u6age4v9     |    4096 | 0.23483   | 0.0036451  | 0.594146 |
| u6age4v9     |    5794 | 0.236083  | 0.00431481 | 0.594146 |
| us2299gs     |      16 | 0.100607  | 0.00310666 | 0.334417 |
| us2299gs     |      64 | 0.117316  | 0.00293407 | 0.41762  |
| us2299gs     |     256 | 0.187055  | 0.00426654 | 0.579715 |
| us2299gs     |    1024 | 0.234947  | 0.00352162 | 0.624854 |
| us2299gs     |    4096 | 0.252846  | 0.00352162 | 0.615807 |
| us2299gs     |    5794 | 0.255218  | 0.00365674 | 0.624854 |
| xt0wu8cf     |      16 | 0.108987  | 0.00345185 | 0.366548 |
| xt0wu8cf     |      64 | 0.129467  | 0.00345185 | 0.447736 |
| xt0wu8cf     |     256 | 0.186483  | 0.00345185 | 0.575409 |
| xt0wu8cf     |    1024 | 0.232144  | 0.00372551 | 0.60116  |
| xt0wu8cf     |    4096 | 0.245799  | 0.00328641 | 0.601495 |
| xt0wu8cf     |    5794 | 0.249019  | 0.00344427 | 0.601495 |

And here are the random vector results.
They're equally strong.

|   Prototypes |   Train |      mAP |         5% |      95% |
|--------------|---------|----------|------------|----------|
|          128 |      16 | 0.10932  | 0.00524497 | 0.386835 |
|          128 |      64 | 0.127726 | 0.00418055 | 0.461893 |
|          128 |     256 | 0.135391 | 0.00418767 | 0.466669 |
|          128 |    1024 | 0.146155 | 0.00339619 | 0.473166 |
|          128 |    4096 | 0.154615 | 0.0032345  | 0.474334 |
|          128 |    5794 | 0.155311 | 0.0032345  | 0.474334 |
|          512 |      16 | 0.121419 | 0.00342993 | 0.411526 |
|          512 |      64 | 0.130049 | 0.00332486 | 0.498455 |
|          512 |     256 | 0.167661 | 0.00439419 | 0.532045 |
|          512 |    1024 | 0.196644 | 0.0037761  | 0.553886 |
|          512 |    4096 | 0.205265 | 0.00399592 | 0.554241 |
|          512 |    5794 | 0.20622  | 0.00393551 | 0.553886 |
|         2048 |      16 | 0.112203 | 0.00356719 | 0.390693 |
|         2048 |      64 | 0.142566 | 0.00389514 | 0.465463 |
|         2048 |     256 | 0.184404 | 0.00313376 | 0.551546 |
|         2048 |    1024 | 0.208458 | 0.00412844 | 0.569776 |
|         2048 |    4096 | 0.215505 | 0.00449296 | 0.570287 |
|         2048 |    5794 | 0.216999 | 0.0046339  | 0.570287 |
|         8224 |      16 | 0.107898 | 0.00341703 | 0.377234 |
|         8224 |      64 | 0.149802 | 0.00328508 | 0.505668 |
|         8224 |     256 | 0.193558 | 0.00341703 | 0.595349 |
|         8224 |    1024 | 0.226368 | 0.00407066 | 0.618476 |
|         8224 |    4096 | 0.23801  | 0.00396418 | 0.62261  |
|         8224 |    5794 | 0.239192 | 0.00436143 | 0.62261  |
|        32768 |      16 | 0.108925 | 0.00333207 | 0.402901 |
|        32768 |      64 | 0.141772 | 0.00361151 | 0.510398 |
|        32768 |     256 | 0.197465 | 0.00386315 | 0.584396 |
|        32768 |    1024 | 0.237713 | 0.00345829 | 0.623653 |
|        32768 |    4096 | 0.251146 | 0.00548655 | 0.621171 |
|        32768 |    5794 | 0.254775 | 0.00689308 | 0.624733 |

I think we need to:

1. DINOv2 instead of SigLIP?
2. Earlier layers?
3. Better SAEs (activation functions, objectives)?
4. Train SAEs on bird-only images?

For debugging, we also need to see which features are chose for a given trait.

Thus, I need to:

1. Record DINOv2 activations on CUB train & test.
2. Sweep both random baselines and SAEs on all layers trained for various activations.
In plain language, I want to:

Train and evaluate random vectors perform on all layers of DINOv2, SiglIP and BioCLIP 2 activations.
As output, I want an exploded table with

| CUB Attr | Average Precision | N Prototypes | N Train | ViT | Layer |

Then I can calculate all the metrics I want using polars.

The hard parts:

1. Should this be one script? Or split up into two scripts, like it is right now? (train_baselines.py and dump_cub200_scores.py)
2. Should whatever we choose also work for SAEs? Do other methods need more hyperparameters? Should those be included in the table?

The outputs for the SAEs should be:

| CUB Attr | Average Precision | N Train | ViT | Layer | Eval L0 | Eval MSE | Sparsity Coeff | Objective | D_sae | Learning Rate |

Then, again, I can calculate all the metrics I want using polars.

So between train_baselines.py, dump_cub200_scores.py, src/saev/interactive/metrics.py and notebooks/results.py, I think I have all the pieces.

I typically do:
```sh
uv run train_baseline.py --sweep sweeps/train-cub200-baselines.toml --method random --slurm-acct PAS2136 --slurm-partition nextgen --n-hours 12 --data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/900da851ddfb6085f76db3c7a75a62c2f6c4ee60ca64556cb6eefa47f7cd6c6e/ --data.layer 13 --data.debug

uv run dump_cub200_scores.py --train-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/900da851ddfb6085f76db3c7a75a62c2f6c4ee60ca64556cb6eefa47f7cd6c6e/ --train-data.layer 23 --test-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/9c29c95d5663c77b69069dc55bbb72e1de9ec0cbc9392f067d57b41f4b769980/ --test-data.layer 23 --cub-root /fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder --slurm-acct PAS2136 --slurm-partition nextgen --sweep sweeps/eval-cub200-baselines.toml

uv run dump_cub200_scores.py --train-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/900da851ddfb6085f76db3c7a75a62c2f6c4ee60ca64556cb6eefa47f7cd6c6e/ --train-data.layer 23 --test-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/9c29c95d5663c77b69069dc55bbb72e1de9ec0cbc9392f067d57b41f4b769980/ --test-data.layer 23 --cub-root /fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder --slurm-acct PAS2136 --slurm-partition nextgen --sweep sweeps/eval-cub200-saes.toml

uv run marimo edit  # navigate to and run notebooks/results.py.
```

But I would like to run something like

```sh
uv run magic.py \
    --sweep/magic-broom.toml \
    --method random \
    --slurm-acct PAS2136 \
    --slurm-partition nextgen \
    --n-hours 12 \
    --train-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/900da851ddfb6085f76db3c7a75a62c2f6c4ee60ca64556cb6eefa47f7cd6c6e/ \
    --test-data.shard-root /.../ \
    --images.root /.../ \
```

And then get a nice table as JSON or CSV in `results/magic.json` or such, which I can then load into a polars dataframe in marimo

```sh
uv run evaluate_trait_discovery.py --sweep sweeps/cub200-baselines.toml --cub-root /fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder --train-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/900da851ddfb6085f76db3c7a75a62c2f6c4ee60ca64556cb6eefa47f7cd6c6e --test-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/9c29c95d5663c77b69069dc55bbb72e1de9ec0cbc9392f067d57b41f4b769980/ --slurm-acct PAS2136 --slurm-partition nextgen
```

Okay. This seemed to have worked.
However, layer 13 < layer 17 > layer 21 < layer 23.
And approximately, layer 13 == layer 21 and layer 17 == layer 23.
This makes me think that we have a bug in our code somewhere.

Here are the raw results (for LLMs, agents, etc):

|   Prototypes |   Train |   Layer |       mAP |         5% |      95% |
|--------------|---------|---------|-----------|------------|----------|
|          128 |      16 |      13 | 0.099902  | 0.00330546 | 0.337178 |
|          128 |      16 |      17 | 0.106822  | 0.00308492 | 0.365345 |
|          128 |      16 |      21 | 0.0951607 | 0.00348695 | 0.344674 |
|          128 |      16 |      23 | 0.105443  | 0.00344247 | 0.370793 |
|          128 |      64 |      13 | 0.104357  | 0.00328143 | 0.356923 |
|          128 |      64 |      17 | 0.121689  | 0.00352573 | 0.423669 |
|          128 |      64 |      21 | 0.0988818 | 0.00364708 | 0.348666 |
|          128 |      64 |      23 | 0.11504   | 0.00319519 | 0.390296 |
|          128 |     256 |      13 | 0.112886  | 0.00322249 | 0.388678 |
|          128 |     256 |      17 | 0.149445  | 0.00362755 | 0.486611 |
|          128 |     256 |      21 | 0.109768  | 0.00374001 | 0.373068 |
|          128 |     256 |      23 | 0.140324  | 0.00378009 | 0.465516 |
|          128 |    1024 |      13 | 0.121219  | 0.00375318 | 0.405894 |
|          128 |    1024 |      17 | 0.161234  | 0.00335293 | 0.517461 |
|          128 |    1024 |      21 | 0.110127  | 0.00371413 | 0.386314 |
|          128 |    1024 |      23 | 0.162851  | 0.00370978 | 0.51155  |
|          128 |    4096 |      13 | 0.124581  | 0.00329968 | 0.423978 |
|          128 |    4096 |      17 | 0.165506  | 0.003624   | 0.518951 |
|          128 |    4096 |      21 | 0.118299  | 0.00370199 | 0.432019 |
|          128 |    4096 |      23 | 0.166815  | 0.00374301 | 0.523304 |
|          128 |    5994 |      13 | 0.128721  | 0.00373144 | 0.421465 |
|          128 |    5994 |      17 | 0.165717  | 0.0035344  | 0.508889 |
|          128 |    5994 |      21 | 0.121071  | 0.00374879 | 0.440073 |
|          128 |    5994 |      23 | 0.171517  | 0.00360793 | 0.493421 |
|          512 |      16 |      13 | 0.100406  | 0.00356321 | 0.34778  |
|          512 |      16 |      17 | 0.106342  | 0.00326544 | 0.364687 |
|          512 |      16 |      21 | 0.0998768 | 0.00332343 | 0.354251 |
|          512 |      16 |      23 | 0.104305  | 0.00339516 | 0.359524 |
|          512 |      64 |      13 | 0.109504  | 0.00363827 | 0.384808 |
|          512 |      64 |      17 | 0.133538  | 0.00341353 | 0.436992 |
|          512 |      64 |      21 | 0.108298  | 0.00348944 | 0.36835  |
|          512 |      64 |      23 | 0.120439  | 0.00334609 | 0.42308  |
|          512 |     256 |      13 | 0.127516  | 0.00341652 | 0.465929 |
|          512 |     256 |      17 | 0.170479  | 0.00356695 | 0.532403 |
|          512 |     256 |      21 | 0.128753  | 0.00350896 | 0.412343 |
|          512 |     256 |      23 | 0.167813  | 0.00366484 | 0.519662 |
|          512 |    1024 |      13 | 0.137014  | 0.00386583 | 0.456923 |
|          512 |    1024 |      17 | 0.191602  | 0.00364351 | 0.545216 |
|          512 |    1024 |      21 | 0.135838  | 0.00355732 | 0.459386 |
|          512 |    1024 |      23 | 0.186345  | 0.00367242 | 0.544843 |
|          512 |    4096 |      13 | 0.141775  | 0.00337709 | 0.460786 |
|          512 |    4096 |      17 | 0.19881   | 0.00351026 | 0.543437 |
|          512 |    4096 |      21 | 0.144787  | 0.00371475 | 0.457885 |
|          512 |    4096 |      23 | 0.200397  | 0.00368842 | 0.572783 |
|          512 |    5994 |      13 | 0.13874   | 0.0035669  | 0.464499 |
|          512 |    5994 |      17 | 0.207367  | 0.00361785 | 0.578806 |
|          512 |    5994 |      21 | 0.140021  | 0.00371336 | 0.4478   |
|          512 |    5994 |      23 | 0.198996  | 0.00394433 | 0.547743 |
|         2048 |      16 |      13 | 0.100315  | 0.00353794 | 0.34614  |
|         2048 |      16 |      17 | 0.104898  | 0.00334024 | 0.346225 |
|         2048 |      16 |      21 | 0.102542  | 0.00339493 | 0.35882  |
|         2048 |      16 |      23 | 0.110083  | 0.00333054 | 0.390993 |
|         2048 |      64 |      13 | 0.114867  | 0.00327129 | 0.411356 |
|         2048 |      64 |      17 | 0.139635  | 0.00349516 | 0.498049 |
|         2048 |      64 |      21 | 0.11879   | 0.00376159 | 0.409469 |
|         2048 |      64 |      23 | 0.134941  | 0.00355747 | 0.454934 |
|         2048 |     256 |      13 | 0.138103  | 0.00333705 | 0.476023 |
|         2048 |     256 |      17 | 0.192353  | 0.00401555 | 0.583648 |
|         2048 |     256 |      21 | 0.136766  | 0.00363226 | 0.451837 |
|         2048 |     256 |      23 | 0.185044  | 0.00344053 | 0.56459  |
|         2048 |    1024 |      13 | 0.15537   | 0.0036975  | 0.490913 |
|         2048 |    1024 |      17 | 0.222745  | 0.0037053  | 0.581673 |
|         2048 |    1024 |      21 | 0.151831  | 0.00385387 | 0.471914 |
|         2048 |    1024 |      23 | 0.209594  | 0.00342191 | 0.586507 |
|         2048 |    4096 |      13 | 0.165254  | 0.00355355 | 0.508869 |
|         2048 |    4096 |      17 | 0.23784   | 0.00372401 | 0.581079 |
|         2048 |    4096 |      21 | 0.164814  | 0.00382632 | 0.495539 |
|         2048 |    4096 |      23 | 0.226707  | 0.0040119  | 0.590405 |
|         2048 |    5994 |      13 | 0.165567  | 0.00370824 | 0.497689 |
|         2048 |    5994 |      17 | 0.236885  | 0.0037925  | 0.594213 |
|         2048 |    5994 |      21 | 0.164975  | 0.00423534 | 0.49209  |
|         2048 |    5994 |      23 | 0.226336  | 0.00415778 | 0.586389 |
|         8224 |      16 |      13 | 0.101383  | 0.0034287  | 0.336689 |
|         8224 |      16 |      17 | 0.10959   | 0.00363317 | 0.359425 |
|         8224 |      16 |      21 | 0.105519  | 0.00363595 | 0.366047 |
|         8224 |      16 |      23 | 0.106864  | 0.0036852  | 0.365608 |
|         8224 |      64 |      13 | 0.119965  | 0.00365385 | 0.420588 |
|         8224 |      64 |      17 | 0.143741  | 0.00309537 | 0.494022 |
|         8224 |      64 |      21 | 0.124667  | 0.00382045 | 0.425296 |
|         8224 |      64 |      23 | 0.142287  | 0.00351817 | 0.505698 |
|         8224 |     256 |      13 | 0.148118  | 0.00330956 | 0.520434 |
|         8224 |     256 |      17 | 0.203606  | 0.00370064 | 0.587269 |
|         8224 |     256 |      21 | 0.150596  | 0.00383384 | 0.488548 |
|         8224 |     256 |      23 | 0.198211  | 0.00386236 | 0.58296  |
|         8224 |    1024 |      13 | 0.172325  | 0.00363658 | 0.523506 |
|         8224 |    1024 |      17 | 0.235498  | 0.00373675 | 0.609907 |
|         8224 |    1024 |      21 | 0.168956  | 0.00376967 | 0.506099 |
|         8224 |    1024 |      23 | 0.221139  | 0.00362223 | 0.603788 |
|         8224 |    4096 |      13 | 0.185445  | 0.00379274 | 0.545261 |
|         8224 |    4096 |      17 | 0.251893  | 0.00414814 | 0.613367 |
|         8224 |    4096 |      21 | 0.174193  | 0.00442431 | 0.502884 |
|         8224 |    4096 |      23 | 0.239761  | 0.0035579  | 0.606663 |
|         8224 |    5994 |      13 | 0.190149  | 0.00368306 | 0.549831 |
|         8224 |    5994 |      17 | 0.257553  | 0.00454221 | 0.609672 |
|         8224 |    5994 |      21 | 0.178852  | 0.00398184 | 0.501153 |
|         8224 |    5994 |      23 | 0.240209  | 0.00496067 | 0.613996 |
|        32768 |      16 |      13 | 0.101156  | 0.00349897 | 0.334814 |
|        32768 |      16 |      17 | 0.104902  | 0.00314443 | 0.357403 |
|        32768 |      16 |      21 | 0.105595  | 0.00348333 | 0.351396 |
|        32768 |      16 |      23 | 0.104998  | 0.00370875 | 0.357553 |
|        32768 |      64 |      13 | 0.120855  | 0.00333491 | 0.433436 |
|        32768 |      64 |      17 | 0.147019  | 0.00343807 | 0.518568 |
|        32768 |      64 |      21 | 0.127113  | 0.00343559 | 0.465773 |
|        32768 |      64 |      23 | 0.145418  | 0.00394493 | 0.519853 |
|        32768 |     256 |      13 | 0.159191  | 0.00334751 | 0.523756 |
|        32768 |     256 |      17 | 0.210674  | 0.00345727 | 0.604252 |
|        32768 |     256 |      21 | 0.162428  | 0.00360099 | 0.514294 |
|        32768 |     256 |      23 | 0.205253  | 0.00384745 | 0.602335 |
|        32768 |    1024 |      13 | 0.184816  | 0.00358291 | 0.549556 |
|        32768 |    1024 |      17 | 0.248688  | 0.00394058 | 0.634789 |
|        32768 |    1024 |      21 | 0.18122   | 0.00378961 | 0.528377 |
|        32768 |    1024 |      23 | 0.239894  | 0.00342861 | 0.616644 |
|        32768 |    4096 |      13 | 0.204362  | 0.00351442 | 0.562282 |
|        32768 |    4096 |      17 | 0.265153  | 0.00401285 | 0.622709 |
|        32768 |    4096 |      21 | 0.188627  | 0.00363691 | 0.519043 |
|        32768 |    4096 |      23 | 0.254377  | 0.00436139 | 0.626323 |
|        32768 |    5994 |      13 | 0.205562  | 0.00378191 | 0.571589 |
|        32768 |    5994 |      17 | 0.266949  | 0.00461905 | 0.631839 |
|        32768 |    5994 |      21 | 0.18853   | 0.00377243 | 0.52858  |
|        32768 |    5994 |      23 | 0.257002  | 0.00510297 | 0.619554 |

Jake suggested trying all of the last 12 layers.
Since the CUB vectors are small (only 12K images, about 72GB for 12 layers), we can simply store all of the last 12 layers.

# 08/07/2025

```sh
# Run 
uv run eval_cub200.py --sweep sweeps/cub200-baselines.toml --cub-root /fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder --train-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/5d7021d1fef171427b0a165c89fae8cbae5af7e91080ca9bafccbadb5318c9d9 --test-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/706cafabc5a038769ace3c5025c021cac1b3259c7b3ffb80759eab521bedaf04/ --slurm-acct PAS2136 --slurm-partition nextgen
```

Okay. I think I got it figured out for DINOv2.
It seems to follow smoother trends than SigLIP2.
I think I want to regenerate the SigLIP2 activations for CUB and try these experiments again.
But I need to keep these old results around.

So now I need to:

1. Regenerate the SigLIP 2 activations for CUB, all layers (12-23) [done]
2. Move the existing siglip results to `results.backup`. [done]
3. Re-run the random vector experiment with the new siglip2 activations. [done]
4. Compare against DINOv2. [done]
5. Compare against SAEs for both DINOv2 and SigLIP2.

If this leads to meaningfully different results, then I need to update the siglip shards for iNat21 as well.

```sh
# Train split
uv run python -m saev.data --slurm-acct PAS2136 --n-hours 48 --slurm-partition nextgen --vit-family siglip --vit-ckpt hf-hub:timm/ViT-L-16-SigLIP2-256 --d-vit 1024 --n-patches-per-img 256 --vit-layers 12 13 14 15 16 17 18 19 20 21 22 23 --dump-to /fs/scratch/PAS2136/samuelstevens/cache/saev/ --no-cls-token --max-patches-per-shard 500_000 data:image-folder --data.root /fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder/train/

# Test split
uv run python -m saev.data --slurm-acct PAS2136 --n-hours 48 --slurm-partition nextgen --vit-family siglip --vit-ckpt hf-hub:timm/ViT-L-16-SigLIP2-256 --d-vit 1024 --n-patches-per-img 256 --vit-layers 12 13 14 15 16 17 18 19 20 21 22 23 --dump-to /fs/scratch/PAS2136/samuelstevens/cache/saev/ --no-cls-token --max-patches-per-shard 500_000 data:image-folder --data.root /fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder/test/
```


NIH ChestX-ray14, 5.5K citations

https://nihcc.app.box.com/v/ChestXray-NIHCC
https://arxiv.org/pdf/1705.02315

112,120 total images with size 1024 x 1024
8 possible labels (multi-label)
880 also include bounding boxes for four findings

Not really like CUB, where birds have a set of components. On average, each image has 1.2 pathologies.


VinDr-CXR: An open dataset of chest X-rays with radiologist’s annotations (400+ citation)

> Out of this raw data, we release 18,000 images that were manually annotated by a total of 17 experienced radiologists with 22 local labels of rectangles surrounding abnormalities and 6 global labels of suspected diseases.

15K training, 3K test with 22 localizel labels, 6 global/image-level labels.


MIMIC-CXR (Scientific Data, used in SAE-Rad)

377K images


HAM10000
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FDBW86T

I'm not sure what annotations are provided. But it seems that there are pixel-level annotations. These could be used to produce bounding boxes or something like that.

It's similar to semantic segmentation.


```sh
# Run SigLIP 2 random vectors
uv run eval_cub200.py --sweep sweeps/cub200-baselines.toml --cub-root /fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder --train-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/d111f368ba4ae9515c9efd7f75f568e40a13072a48f88702166a76bc458628b5 --test-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/0d868f6ea242faebb5c167d7131358e6dd1a4d9ebbd0268b43667f27c8bcbd75/ --slurm-acct PAS2136 --slurm-partition nextgen
```

# 08/12/2025

1. DINOv2 instead of SigLIP?
2. Earlier layers?
3. Better SAEs (activation functions, objectives)?
4. Train SAEs on bird-only images?

- Jake will figure out BatchTopK/TopK aux + threshold during inference
- I will train a linear classifier for each of the traits.
- I will actually compare the SAEs to random vectors.
- Ask Rayeed about kmeans/pca; if those don't improve then we're in real trouble.

# 08/26/2025

- DINOv3
- scientific discovery test sets
- Writing
- CLS token for linear probing on CUB 200
- FishVista

I have sort of figured out a native resolution trick where we convert every image into 640 (or N, for any N) tokens.
This preserves the aspect ratio as much as possible.

Now we need to feed these patches into DINov3.
This involves updating the dataloader.
It's PyTorch, so the resize is dynamic and depends on the image.
We can write that custom transform ourselves.

The other trick is to make sure it's clear which position the activations have when they're saved to shards.
Pretty tricky.

# 08/28/2024

```sh
uv run contrib/trait_discovery/scripts/format_fishvista.py --fv-root /fs/scratch/PAS2136/samuelstevens/datasets/fish-vista/ --dump-to /fs/scratch/PAS2136/samuelstevens/datasets/fish-vista-segfolder
```

```sh
uv run python -m saev.data --dump-to /fs/scratch/PAS2136/samuelstevens/cache/saev --vit-layers 13 15 17 19 21 23 --vit-family dinov3 --vit-ckpt /fs/ess/PAS2136/samuelstevens/models/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth --slurm-acct PAS2136 --slurm-partition nextgen --n-patches-per-img 640 --d-vit 1024 --vit-batch-size 256 data:seg-folder --data.root /fs/scratch/PAS2136/samuelstevens/datasets/fish-vista-segfolder/ --data.img-label-fname image_labels.txt
```

I need to evaluate a linear classifier on DINOv3 traits.
This is kind of hard because of the way the aspect-aware resizing works with segmentation masks.
So it goes.
But I just have to keep trucking, with the intent of measuring classifier accuracy on traits at a very coarse level.
And we have to visualize the trait labels in a notebook to make sure they're not shit.

# 08/29/2025

I did visualize the trait labels in a notebook.
You can see these results in contrib/trait_discovery/notebooks/fishvista-patches.ipynb.

So now we just have to keep going and train a linear model to get the upper bound.

I have submitted a job.
It doesn't do validation because I didn't want to work out interpolation in order to calculate mAP.
It also doesn't save checkpoints.
But at least this way we can see what the performance of a linear probe is.
Then we can visualize some predictions.
After that, we will be forced to do the baselines again and get some freaking results.

In parallel, I could also work on:

1. Writing
2. Beetles?
3. Butterflies?
4. Equids?

Certainly I need to start with butterflies, since that was the original motiviation of the project.
I need to read that paper about SAEs for discovery.
I need to measure mAP for these linear probes.

Okay, since working on all these tasks in parallel will probably make me lose my mind, let's simply organize them sequentially and work on them one at a time.

1. [done] Use the prefer-fg pixel aggregation on FishVista.
2. [done] Include patch-level validation accuracy on FishVista (re-train).
3. [done] Include validation mAP on FishVista (re-train).
4. Save linear checkpoints
5. Apply RandomVectors to FishVista.
6. Butterflies, beetles, equids, writing, etc.

Now we need to take the RandomVector thing and apply it to the FishVista dataset using DINOv3.

For CUB-200, I used the max across all patches to score each image, because for each trait, I only had image-level annotations. For FishVista, I have patch-level annotations (technically pixel-level, but we operate at patches). I want to train/evaluate the RandomVector Scorer on FishVista. Make a plan for how you would implement that.

I also have saved FishVista shards, so we can use the same set of dataloaders from CUB200 for FishVista.
The main thing I would like to simplify is in CUB200, we had a training script, then an evaluate script.
For SAEs, there is no training (people will use train.py from the main repo root).
For linear probes, we have supervised_fishvista.py.
For RandomVectors, PCA, KMeans, I think training is short enough that we don't need to separate the scripts.
So I think we can put stuff that's common to cub200 and fishvista in src/tdiscovery/*.py.


```sh
uv run eval_fishvista.py --sweep sweeps/fishvista-baselines.toml
uv run supervised_fishvista.py --sweep sweeps/fishvista-supervised.toml
```

# 09/03/2025

1. [done] Evaluate DINOv3-trained SAEs on FishVista.
2. Try training Matryoshka SAEs on DINOv3+iNat21 if it doesn't work.
3. [in progress] Vanilla SAEs on FishVista
4. Subset of ToL-200M for fish - talk to Matt + Net.

Later:

- Butterflies
- Beetles
- Equids

I also need to visualize the learned DINOv3 features on iNat21.

The SAE is really bad at fitting the DINOv3 features on iNat21.
That is, the MSE/L0 tradeoff is really bad.
Why is this?

- Increased dimensionality (768 -> 1024) -> probably not, we trained on SigLIP 2 and DINOv2 ViT-Ls.
- Increased feature richness (DINOv2 -> DINOv3)
- FlexResize (16x16 patches to HxW patches)
- Something else?

For reference, here's how we did on ImageNet-1K with CLIP and DINOv2 ViT-B:

| Model | MSE | L0 | Dead | Dense|
|---|---|---|---|
| CLIP | 0.0761 | 412.7 | 0 | 11,858|
| DINOv2 | 0.0697 | 728.7 | 1 | 19,735|

And here's a sweep on layer 11/12

| Model |	L0 | MSE | lambda | Learning Rate |
|---|---|---|---|---|
| DINOv2 ViT‑B/14 |	3003.89 | 0.03821 | 0.0004 |0.0003 |
|  |	1138.11 | 0.07542 | 0.0008 |0.0003 |
|  |	419.05 | 0.12867 | 0.0016 |0.0003 |
|  |	1614.23 | 0.04765 | 0.0004 |0.0010 |
|  |	664.96 | 0.08113 | 0.0008 |0.0010 |
|  |	249.24 | 0.13107 | 0.0016 |0.0010 |
|  |	728.71 | 0.07024 | 0.0004 |0.0030 |
|  |	320.43 | 0.10416 | 0.0008 |0.0030 |
|  |	138.78 | 0.15187 | 0.0016 |0.0030 |
| CLIP ViT‑B/16 |	2109.84 | 0.02967 | 0.0004 |0.0003 |
|  |	1011.87 | 0.06381 | 0.0008 |0.0003 |
|  |	289.24 | 0.12350 | 0.0016 |0.0003 |
|  |	1585.54 | 0.03219 | 0.0004 |0.0010 |
|  |	694.05 | 0.06570 | 0.0008 |0.0010 |
|  |	183.17 | 0.11830 | 0.0016 |0.0010 |
|  |	776.41 | 0.03137 | 0.0004 |0.0030 |
|  |	412.70 | 0.07598 | 0.0008 |0.0030 |
|  |	124.65 | 0.12318 | 0.0016 |0.0030 |


Notes from Tanya

Think about how a combination of SAE features can map to a given class or node in the tree of life.

Can we visualize a heatmap of the 16K SAE features over the 60K fish images?
Then, can we make it interactive where we sort by the species, genus, etc?
Can we order the 16K features by co-occurence?
This could really lead to data-driven discovery.
If the co-occurences are pure with respect to a node in the tree of life, then we can say a set of features is both sufficient and necessary to distinguish/classify a node in the tree?
We could do this for beetles too and go beyond just species label--connect to the other metadata.

---

I need to train some STUPID SAES ON MY DATA.
Let's make it as easy as possible.

First, train an SAE on FishVista training data. Use all the images, not just the segmented ones. [in progress]

# 09/05/2025

I trained some SAEs on the FishVista classification images.
I also trained some Matryoshka SAEs on the FishVista classification images.

What do I still need to do?

I think we might want to do class-level segmentation.
Even with the best matryoshka SAE to prevent feature splitting, we might not ever get a pure feature across all species.
But we clearly are getting very nice fish features.

So to get good quantitative results, we might need to break the fishes up a little bit.

# 09/07/2025

Butterflies! Today is all about butterflies!

1. Train Matryoshka SAE on Heliconious images
2.


iNat21 training has

- 374,161 Lepitdoptera images.
- 63980 Nymphalidae images
- 600 Heliconious images 

Even if we just train on the Nymphalidae images, that triples our 21.9K in the Jiggins dataset.

| Dataset | Images  | Patches |
|---------|---------|---------|
| Jiggins |  21,298 |   13.6M |
| Lepit.  | 374,161 |  239.4M |
| Nymphs  |  63,980 |   40.9M |

But I think even just 50M patches will be sufficient (Jiggins + Nymphs).
We can train for multiple epochs.

Experimental plan:

1. [done] Train Matryoshka on Jiggins images
2. Train Matryoshka on Jiggins+Nymphs

- Validation split??
- Visuals??
- Heatmap??

What do I want? I want to train an SAE on the Jiggins images and measure the validation accuracy. Do I? Is it worth the missing training data?


# 09/09/2025

1. Save activations for butterfly datasets.
2. Sort features by image activations measures.

Here are some thoughts:

1. F1 is not perfect, because we typically want classes with at least 5 examples.
2. We end up with a lot of spurious correlations. For example,

I think we need even smaller patches, so higher resolution images.

Additional metrics:

- Classification accuracy/F1 for mimic *pairs*.
- AP?
- SAE latent entropy doesn't seem to work well either. Maybe we need to filter based on the number of unique images in a latent's top-k?

I think we need a separate visuals script + notebook for trait discovery.
Show:

- Mimic pair comparison
- Ventral vs dorsal?


# 09/10/2025

I have a lot of different experiments in different stages.

1. FishVista
2. Butterflies
3. Beetles

For each of them, I'll lay out the progress, the blockers, and potential ways to get around the blockers. Then I need to decide which experiment to prioritize.

FishVista

- I trained Matryoshka SAEs on DINOv3 FishVista classification images (56.3K images) with 640 patches per image.
- Then I measured whether we could find SAE features/latents that reliably detected the 9 different traits in FishVista (head, eye, tail, pelvic fin, pectoral fin, etc).
- I compared this to just picking random vectors and then filtering those random vectors using a training set to pick out the 9 best vectors.
- The SAEs underperform the random vectors. I probably could do more detailed error analysis to figure out why, but my impression is that we get good SAE features, but they might be specific to a given subnode in the tree of life. So maybe not a pelvic fin for all fish in FishVista, but a pelvic fin for all sunfish.
- However, breaking up the classes is complex and makes the experiment harder to explain, which is bad for reviews. Furthermore, FishVista is just a quantitative experiment to satisfy reviewers. It's not the main goal of this paper, but it can be used as a method to measure SAE progress.

Butterflies

- I trained Matryoshka SAEs on DINOv3 Jiggins images (21.8K images) with 640 patches per image.
- Then I looked at SAE features to see if we could find features that reliably trigger on one mimic pair but not another. I used thresholded SAE values (>= 95th percentile per SAE feature = positive) to pick out features that maximized F1, precision and recall for each of the below pairs, and inspected example feature images from an SAE from layer 14/24 and an SAE from layer 22/24.
- I found a couple (~3-4) possible new discoveries, but I need to talk to Dan/Neil (our subject matter experts).
- Mostly I found a lot of spurious correlations that could be removed by only training on butterfly patches. Unfortunately, I would have to use [SST](https://github.com/DavidCarlyn/SST) to do that, and I don't think it will be trivial.
- I need to measure average precision, which is a threshold-free.
- I need to measure F1/precision/recall on the binary task of mimic A vs mimic B.

| A | # of A | B | # of B |
|---|---|---|---|
| lativitta | 2640 | malleti | 1620  |
| cyrbia | 1397 | cythera | 110  |
| notabilis | 573 | plesseni | 337  |
| hydara | 280 | melpomene | 462  |
| venus | 237 | vulcanus | 158  |
| demophoon | 222 | rosina | 132  |
| phyllis | 194 | nanna | 77  |
| erato | 67 | thelxiopeia | 2  |

Beetles

In addition to fish and butterflies, I have ~40K images of ground beetles that I could train SAEs on.
These beetles DO have segmentation masks already, so I could train SAEs specifically on the beetles themselves, not the background.
I could then use the part of body segmentations to do something similar to FishVista.
However, we lack a good ecological context for finding something *new* about the beetles. Mimics are a straightforward, obvious context.
I'm not sure about beetles.

What should I do, if I want to submit to ICLR (10 days)?

# 09/11/2025

Ideas:

- SAEs on segmented butterflies.
- SAEs on segmented beetles
- Track mimic a vs b accuracy for SAE features
- Higher resolution SAEs
- Better optimization techniques
- More images (in situ vs specimen)

So I need to do the segmentation myself.
There are 130 different classes, and I feel like many of the classes have mostly consistent imaging protocols.
Maybe actually I can use some of the other metadata to figure out the imaging protocol, then use that to cluster the images.
Then I can do SST inference for each group within a given imaging protocol.

# 09/16/2025

I think I have segmentations working for butterflies.
So I want to train some Matryoshka SAEs on just the butterfly images.

Some additional metrics:

- Randomness of each batch, as measured by entropy over image indices and patch indices

1. Get all activations
2. Train SAEs on the different datasets. Use Matryoshka. We have been getting okay results so far.
3. Visualize some experiments. Specifically, it needs to be the butterflies and fish. We should compare against high res butterflies and high res butterflies with filtering.

Quality of life improvements:

- Include an optional validation split (ADE20K)

Prior work on later layers being harder to model?
Is 16x enough?

# 09/18/2025

- Repo usability
- Repo code quality
- More unit tests
- Code structure: can you do jaxtyping + submitit in same file? different file?


1. Save .X data for each SAE.
2. Use that to pick out images for top k images per latent

# 09/19/2025

To get the visuals right for butterflies, we have to debug every step.
Something still isn't working because my visuals are firing on non-bg patches.

# 09/20/2025

To evaluate semantic segmentation, we need to measure class-level mIoU.
There are two problems with that:

1. We need to tune the classification value for each combination of segmentation class and latent. For ADE20K, that's 150 x 16K = 2.4M tuned classification values.
2. Sparse autoencoders are not evaluated by their ability to do segmentation! Instead, we use probing tasks like from Gao et al.

Alternatively, we can use the probing evaluation from Gao et al.
That doesn't require manually tuning the activation level; instead, it requires a LGBFS search (which I've never done before).

---

I think we need some tests for dataloaders and patch filtering.
Then we also need a test for loading shards from huggingface and making sure they still work.
We can use ADE20K.

# 09/22/2025

I am going to put the butterfly visualizations on hold for a bit and focus on quantitative probing evaluation of ADE20K and FishVista.

1. Check compute feasibility of a sparse Newton-Raphson for both ADE20K and FishVista
2. Implement a sparse Newton-Raphson
3. Compare my implementation against scipy.optimize and torch.optim.LBFGS via unit tests.
4. Evaluate a random baseline on both ADE20K and FishVista.
5. Evaluate a sparse autoencoder.

How would I define success?
I think I want a probe1d.py script that reports a loss and accuracy for a given SAE evaluated on a given dataset.
It should be trivial to sweep a lot of different sparse autoencoders for a given dataset/layer using a `--sweep` arg.


## Compute Feasibility

Layer 21 has L0 from 100 to 350.
Layer 13 is 30 to 200.
Layer 23 is 50 to 100.

I think a max L0 of 400 is super reasonable for estimating compute.

So I think it's possible.

For ADE20K:

22K images x 256 patches/img x 400 non-zero values per patch x (8 byte int64 index + 4 byte fp32 float value) = 27GB.
Then we will have 32K latents x 151 classes x (2 params + ~12 numbers per param) x 4 bytes = 270.6 MB.

For FishVista:

6.1K images x 640 patches/image x 400 non-zero values per patch x (8 byte int64 index + 4 byte fp32 float value) = 18.7 GB.
Then we will have 32K latents x 10 classes x (2 params + ~12 numbers per param) x 4 bytes = 17.9 MB.

This all fits on a 40GB GPU.
Thus, we will be able to fit probes on every combination of (latent x semantic class) in parallel on a single 40GB A100.


## Implement a Sparse Newton Raphson

Time to write a spec for my coding buddy! I hate math! :)


# 09/23/2025

Okay, Claude Code wrote a decent sparse implementation.
[GPT-5 Pro thinks it will blow up if we try to do all classes in parallel.](https://chatgpt.com/share/68d2b939-a638-8003-bdca-341eb4a125e2)
I actually agree.

There are a couple things that I've learned that I want to remember tomorrow:

1. The above; we need to iterate over classes (151 for ADE20K, 10 for FishVista)
2. dump.py saves sparse matrices instead of image-level max activations. visuals.py will need to be updated to deal with this.
3. Job 2525911 (`tail -f logs/2525911_0_log.out`) might fail because of a dtype issue. We have 29126 images x 1920 patches/image x 16K latents/patch = 894.75B values. 2 ^ 31 = 2.14B and 2 ^ 63 = 9.22e18. So while the column indices (`.indices`) can be int32, the row pointers (`.indptr`) *MUST* be int64. Supposedly hstack will do the conversion. Luckily, this has a negligble impact on size (22MB for int32 indptr to 44MB for int64 intptr).
4. Once we have these activations, then we can try to measure ADE20K accuracy. But we need to dump the activations for the FishVista or ADE20K SAEs, not the butterfly SAEs.

Thus, tomorrow, I need:

1. SAEs trained on FishVista and ADE20K.
2. Dumped activations from these SAEs.

Then I can work on evaluating class-level probes in a memory-efficient manner.


# 09/24/2025

I have activations. Let's try to evaluate ADE20K probes.

# 09/30/2025

I'm doing a big refactor.
I need to update the tests.
Then I want to add the next two commands below:

```sh
uv run scripts/launch.py shards  # runs scripts/shards.py
```

```sh
uv run scripts/launch.py doctor <run_dir>   # validates symlinks and required files.
```

# 10/02/205

Boringly, I simply need to train an SAE on ADE20K.
This implies a training and a validation split.
This also implies that we have the correct disk layout according to the disk-layout.md document.

```sh
uv run train.py --sweep contrib/trait_discovery/sweeps/train-saes.toml --tag ade20k-v0.1 --n-train 100_000_000 --slurm-acct PAS2136 --slurm-partition nextgen --train-data.shards /fs/scratch/PAS2136/samuelstevens/saev/shards/f34a6053594b4f70b9b7fe3cdcdb03b4852ed1a4e01fb1c1d2e270037ad5cbbf/ --val-data.shards /fs/scratch/PAS2136/samuelstevens/saev/shards/771db1317b40582dc64fe552f01aef1f76be444ca5188aa16dca3a9848e1417f/ sae.activation:relu objective:matryoshka
```

# 10/08/2025

I think I finished the refactor.
This landed:

- Python-based sweeps
- A new disk layout to simplify experiments
- Renaming a bunch of variables to make saev extensible to non-vision transformers
- Way better user docs

So now I will train an ADE20K DINOv3 SAE.

# 10/11/2025

Rather than store a reference to a dataset path (which *might* be useful to humans), you can just load the dataset itself in code (which is useful to scripts).

```py
import importlib
import sys


importlib.import_module(act_ds.md.__class__.__module__)  # <module 'saev.data.shards' from '/users/PAS1576/samuelstevens/projects/saev/src/saev/data/shards.py'>
class_obj = getattr(sys.modules[act_ds.md.__class__.__module__], act_ds.md.__class__.__name__)
class_obj  # <class 'saev.data.shards.Metadata'>
```

# 10/13/2025

What the heck am I doing?

I need to train a DINOv3 SAE on ADE20K. I need to see some visuals. I need to do the same for FishVista. I need to use the semantic patch labels to evaluate the quality of the SAE probes. And I need to write.

What is the goal? Why do we care about this?

# 10/16/2025

I think my sparse probe method works. It appears to be quite slow, but it matches all the reference tests.
So now I need to actually evaluate some stupid SAEs on this godforsaken task.
Luckily, I think I have a script to do that.
Holy shit I think it worked.

# 10/17/2025

Some things to work on:

1. train/val split for probe
2. report normalized cross entropy

I would like to compare a bunch of different metrics for these different SAEs.
What questions do I want to answer?

- Are vanilla or matryoshka better?
- How does the probe's explained variance correlate with other metrics?

In summary 
Trained 16K x 151 1D logistic probes on the ADE20K training split.
Took the latent with the minimum loss for each of the 151 classes, then compared the mean loss (binary cross entropy) against the mean loss if you just use the class prevalence as a baseline (basically bias term only, no weight term).
This is explained variance, and is R = 1 - CE / CE_baseline.
If CE is 0, then R is 1.
If CE is the same as CE_baseline, then R is 0.
If CE is worse than CE_baseline, then R is negative.
So R is in the range (-inf, 1].

An 16K latent matryoshka SAE trained on DINOv3 layer 13 with ADE20K training data gets R = 0.104.

Now I just need to:

- Do the probe on train/test, instead of fitting it on train and then evaluating on train as well
- Compare against a bunch of other different SAEs (more/fewer latents, matryoshka vs vanilla, layer 13 vs layer 21, etc)

It turns out my parameters are still pretty bad.

# 10/21/2025

Issues:

1. Sweep 2669919 had a number of failing runs. I suspect this is due to the use of psutil with the dataloader processes. For instance, logs/2669919_86_0_log.out throws an exception on `rb = p_dataloader.io_counters().read_bytes`. We need to fix that in order to run the full sweeps. This is a good conceptual issue for Codex to draw up a design for. It's hard to write unit tests, however (*unless maybe Codex can write a good unit test?*).
2. We still need to speed up the 1D probing. contrib/trait_discovery/logs/2671676_0_log.out has a complete run. None of the runs converge. We also need to deal with less sparse matrices being an issue--what if our matrix is nearly dense? How do we deal with that?

I thought that codex had fixed #1 but it seems not, because the Monitor is not in the train.py file. You can see it in logs/2675297_5_0_log.out.


# 10/23/2025

I think my sparse solver is pretty much done.
There are some code quality issues, where there's a lot of moving things from device and CPU, and a lot of `.to(self.dtype)`.
I am also not sure how it will handle larger arrays that can't go straight o the GPU.
Fially, I don't think it ever breaks early.
One of the goals was to actually stop early if all the gradients are good enough, even on ill-conditioned data.

# 10/24/2025

I actually don't think we need to fix a specific layer. We can train multiple SAEs in parallel. We just need to write the specific sets of model/data/layer combinations to save activations for.

20K images with DINOv3 ViT-L/16, 256 patches/image, 1024 dim/patch, 6 layers is 119GB. So ImageNet-1K has 1.2M images, and would be 60x larger. That's 7.1TB. Luckily I have 30TB, so it is fine. 

I think it's:

1. DINOv3 ViT-L/16, ImageNet-1K training split, 256 patches/image, 6 layers (7.1TB
2. DINOv3 ViT-B/16, ImageNet-1K training split, 256 patches/image, 6 layers (5.3TB)
3. DINOv3 ViT-S/16, ImageNet-1K training split, 256 patches/image, 6 layers (2.6TB)

Then we also need all of these settings for the ADE20K train/val splits. The biggest of these 119GB, so say that all of ADE20K for all the models is AT MOST ~500GB (0.5TB).

So I would train SAEs on every combination of DINOv3 model, layer, with different sparsity lambdas for 100M tokens. This is super cheap. Then I can evaluate probe loss on ADE20K by fitting probes train, measuring loss on val, and only doing this for the probes that are pareto optimal on MSE/L0. I'll only do this for Matryoshka because I'm basically sure that it's best.


# 10/25/2025

Still dealing with probe issues. However, since I am waiting on the SAE training with the dataloader monitor, I think it's fine to keep iterating on the probe. Once I get some imagenet activations finished, then I can train some SAEs.

For my sweeps, I need to have a script to parse and summarize the results. Then I can decide on the final format of the results in the paper.
This should be a notebook that's specific to this paper.

What do I want to know? It probably is going to be a big dataframe that I can make many different charts with quickly to express lots of different relationships.

# 10/26/2025

Now I need to describe my ablation experiments precisely.
This is the core experiment, so it should be straightforward to described.
