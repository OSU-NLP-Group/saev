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
