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

```sh
# Run 
uv run eval_cub200.py --sweep sweeps/cub200-baselines.toml --cub-root /fs/ess/PAS2136/cub2011/CUB_200_2011_ImageFolder --train-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/5d7021d1fef171427b0a165c89fae8cbae5af7e91080ca9bafccbadb5318c9d9 --test-data.shard-root /fs/scratch/PAS2136/samuelstevens/cache/saev/706cafabc5a038769ace3c5025c021cac1b3259c7b3ffb80759eab521bedaf04/ --slurm-acct PAS2136 --slurm-partition nextgen

