# Normalized MSE

I want to report normalized MSE, or fraction of variance unexplained (FVU) for my validation splits.
[
\frac{\sum_i | x_i - \hat x_i |^2}{\sum_i | x_i - \bar x |^2}
]
where (x_i) are activation vectors, (\hat x_i) are SAE reconstructions, and (\bar x) is the dataset mean (compute on the same split we use for evaluation).
This is exactly "MSE divided by the MSE of always predicting the mean activations" used in Gao et al. ("Scaling and Evaluating SAEs"). 
Many papers describe the same thing as "FVU" or "normalized MSE by input variance."

# Implementation Notes

Update src/saev/framework/inference.py to calculate this metric.

Use the identity
[
\sum_i |x_i-\bar x|^2 ;=; \sum_i |x_i|^2 ;-; N,|\bar x|^2
;=; \underbrace{\sum_i |x_i|^2}*{Q} - \frac{1}{N}\underbrace{|\sum_i x_i|^2}*{|S|^2}.
]
So if we maintain during streaming:

* (E=\sum_i|x_i-\hat x_i|^2) (reconstruction SSE),
* (Q=\sum_i|x_i|^2) (sum of squared norms),
* (S=\sum_i x_i\in\mathbb{R}^D) (vector sum),
* (N=) number of samples (tokens/patches),

then at the end:

* (\bar x = S/N),
* ( \text{SST} = Q - N,|\bar x|^2 = Q - |S|^2/N),
* ( \text{NMSE} = E / \text{SST})

Crucially, we don't need (\bar x) beforehand; we derive it from (S) and (N) after the single pass.

Use float64 accumulators for (E, Q, S) to avoid cancellation in (Q - |S|^2/N).

Don't use a scalar mean over all elements (that would normalize to a constant baseline). We want the vector mean baseline, which is why we keep the vector sum (S) (or a running vector mean).

# Citations

* Gao et al., 2024 (Scaling and Evaluating SAEs): "We report a normalized version of all MSE numbers, where we divide by a baseline reconstruction error of always predicting the mean activations."
* Zaigrajew et al., 2025 (Interpreting CLIP with Hierarchical SAEs): defines FVU = normalized MSE as MSE normalized by the mean-squared value of the mean-centered input; EVR is its complement.
* Lawson et al., 2025 (Residual Stream Analysis with Multi-Layer SAEs): explicitly uses FVU and notes raw MSE is not comparable across layers unless we normalize by variance---exactly why NMSE/FVU is preferred.
* Leask et al., 2025 (ICLR "SAEs do not find canonical units"): reports normalized MSE (NMSE) as a main reconstruction metric across SAE variants, showing it's standard.
