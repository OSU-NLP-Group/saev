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
With 128 prototypes, we get a mAP of TODO.

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
