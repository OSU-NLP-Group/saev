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
