# Datapoint Initialization

Datapoint initialization is an SAE weight initializations strategy independently proposed by [Anthropic](https://transformer-circuits.pub/2025/october-update/index.html#data-point-init) and [Pierre Peigne](https://www.lesswrong.com/posts/YJpMgi7HJuHwXTkjk/taking-features-out-of-superposition-with-sparse) for improving SAE training.

Conceptually, we initialize each decoder column to look like a real datapoint, so every latent starts with a patch of input space where it "wins" and gets some gradient.
Here's the algorithm:

1. Select $n$ random data points from your training data.
2. Compute the mean $\mu$ and zero-center the data: $x_0 = x - \mu$.
3. Linearly blend each zero-centered datapoint with Kaiming initialization: $w = p \cdot (x - \mu) + (1 - p) \cdot r$ where $p$ is your blend probability and $r$ is a randomly sampled Kaiming initalization vector.
4. Initialize $W_\text{enc}$ as a concatenation of $n$ blended vectors.
5. Initialize $W_\text{dec}$ as $W_\text{enc}^T$.

Anthropic suggests $p = 0.8$ for SAEs and 0.4 for "weakly causal crosscoders".
I interpret this that there is no universally appropriate $p$.
