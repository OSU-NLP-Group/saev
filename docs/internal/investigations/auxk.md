# AuxK: Auxiliary Loss for Dead Latents

## Problem

In large sparse autoencoders (SAEs), an increasingly large proportion of latents "die" - they stop activating entirely at some point in training and never recover. This is a fundamental problem with scaling SAEs:

- [Anthropic's Templeton et al. (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/) trained a 34M latent SAE with only 12M alive latents
- Ablations showed up to 90% dead latents when no mitigations are applied
- This makes training computationally wasteful and results in substantially worse MSE

## Background

### Prior Art: Ghost Gradients

Earlier work used "ghost gradients" to revive dead latents. This provides gradient signal to dead neurons to encourage them to become active again. However, ghost grads alone are insufficient at scale.

### OpenAI's AuxK (Gao et al., 2024)

The [OpenAI paper "Scaling and Evaluating Sparse Autoencoders"](https://arxiv.org/abs/2406.04093) introduced two key techniques:

1. Initialize W_enc to transpose of W_dec - We already do this in `modeling.py:69`
2. AuxK auxiliary loss - Use dead latents to model the reconstruction error

## AuxK Formulation

### Core Idea

After computing the main reconstruction `x_hat` using live latents, there is still a residual error `e = x - x_hat`. The AuxK loss encourages dead latents to model this residual. AuxK must run on the raw pre-activations that were zeroed out by TopK/BatchTopK (i.e. the non-activated values), not on the post-activation outputs.

1. Identify which latents are "dead" (haven't activated in N tokens)
2. From those dead latents, take the top-k_aux pre-activations (before the TopK/BatchTopK mask was applied)
3. Decode using only those dead latent activations to get auxiliary reconstruction `e_hat`
4. Add auxiliary loss: `L_aux = ||e - e_hat||^2`

### Full Loss

```
L = L_reconstruction + alpha * L_aux
  = ||x - x_hat||^2 + alpha * ||e - e_hat||^2
```

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| k_aux | 512 | "Power of two close to d_model/2" |
| alpha | 1/32 | Coefficient on auxiliary loss |
| dead threshold (tokens) | 10M | How long before a latent is considered dead |

### Computational Cost

Only ~10% overhead since the encoder forward pass (which dominates) can be shared between main and auxiliary losses. We only need an extra decode pass using dead latent activations.

## Usage Across the Literature

### OpenAI (Gao et al., 2024)
- Original formulation with TopK activation
- Reduced dead latents from 90% to 7% in 16M latent SAE
- k_aux = 512, alpha = 1/32, dead threshold = 10M tokens

### BatchTopK (Bussmann et al., 2024)
- [BatchTopK paper](https://arxiv.org/abs/2412.06410) uses identical AuxK formulation
- Same hyperparameters as OpenAI
- k_aux = 512, alpha = 1/32, dead threshold = 5 batches

### Matryoshka SAE (Bussmann et al., 2025)
- [Matryoshka SAE](https://arxiv.org/abs/2503.17547) uses AuxK within nested dictionary training
- Tracks `num_batches_not_active` per feature
- Feature is dead if `num_batches_not_active >= n_batches_to_dead`

### EleutherAI (sparsify)
- Follows Gao et al. recipe with TopK activation
- Implementation at [github.com/EleutherAI/sparsify](https://github.com/EleutherAI/sparsify)

## Implementation Plan

### 1. Track Dead Latents

Use training state, not the module, to hold a counter tensor that is initialized once before wiring the dataloader:

```python
steps_since_active = torch.zeros(d_sae, device=device, dtype=torch.int64)

# In training loop, after forward pass:
active_mask = (f_x.abs() > 0).any(dim=0)  # shape: (d_sae,)
token_count = n_tokens_in_batch  # e.g., tokens_b

steps_since_active[active_mask] = 0
steps_since_active[~active_mask] += token_count

dead_mask = steps_since_active >= dead_threshold_tokens
```

Track total tokens (OpenAI: 10M tokens) rather than batches to make the threshold invariant to variable batch sizes.

### 2. Compute AuxK Loss

After the main forward pass but within the training step. Gradients from AuxK should only touch parameters for dead latents; detach `x_hat` so live-latent paths are unaffected, while keeping gradients through the dead rows of `W_enc`/`b_enc` and `W_dec`/`b_dec`:

```python
def auxk_loss(
    x: Tensor,           # (batch, d_model) - input
    x_hat: Tensor,       # (batch, d_model) - main reconstruction
    pre_acts: Tensor,    # (batch, d_sae) - pre-activation (before TopK)
    dead_mask: Tensor,   # (d_sae,) - boolean mask of dead latents
    W_dec: Tensor,       # (d_sae, d_model) - decoder weights
    b_dec: Tensor,       # (d_model,) - decoder bias
    k_aux: int = 512,
) -> Tensor:
    # isolate AuxK from live latents by detaching the main reconstruction
    residual = (x - x_hat).detach()  # (batch, d_model)

    # Mask out live latents from pre-activations (raw values, not post-activation)
    masked = pre_acts.masked_fill(~dead_mask, float("-inf"))
    n_dead = dead_mask.sum().item()
    k_use = min(k_aux, n_dead)
    if k_use == 0:
        return residual.new_zeros(())

    # TopK among dead latents only
    _, top_i = masked.topk(k_use, dim=-1)
    aux_acts = torch.zeros_like(pre_acts)
    aux_acts.scatter_(-1, top_i, pre_acts.gather(-1, top_i))

    # Decode using dead latent activations; keep gradients to dead rows of W_enc/W_dec and biases
    aux_recon = aux_acts @ W_dec + b_dec

    return (aux_recon - residual).pow(2).mean()
```

### 3. Access Pre-Activations

Currently, `encode()` returns post-activation `f_x`. For AuxK, we need the pre-activation values (before the TopK mask is applied). Options:

Option A: Return both from encode
```python
def encode(self, x) -> tuple[Tensor, Tensor]:
    h_pre = x @ W_enc + b_enc
    f_x = self.activation(h_pre)
    return f_x, h_pre
```

Option B: Separate pre-activation method
```python
def pre_encode(self, x) -> Tensor:
    return x @ W_enc + b_enc
```

Option C: Recompute in training loop
```python
# Less clean but avoids API changes
h_pre = x @ sae.W_enc + sae.b_enc
```

Option A seems cleanest - the training loop needs both anyway.

### 4. Integration Points

- `src/saev/nn/modeling.py`: Add pre-activation access
- `src/saev/framework/train.py`: Add dead latent tracking, compute AuxK loss
- `src/saev/nn/activations.py`: Maybe add k_aux to config, or keep it as training hyperparameter

### 5. Config

```python
@dataclasses.dataclass
class AuxKConfig:
    enabled: bool = True
    k_aux: int = 512
    alpha: float = 1/32
    dead_threshold_tokens: int = 10_000_000
```

## Open Questions

1. Interaction with Matryoshka? When using Matryoshka prefixes, should AuxK apply to all prefix reconstructions or just the final one? The Matryoshka paper applies AuxK to the combined loss.
2. How to count tokens precisely in packed/bucketted batches (include padding tokens or only real tokens)?

## References

- [Gao et al. 2024 - Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093) - Original AuxK formulation
- [Bussmann et al. 2024 - BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410) - Uses same AuxK
- [Bussmann et al. 2025 - Matryoshka SAEs](https://arxiv.org/abs/2503.17547) - Uses AuxK with nested dictionaries
- [EleutherAI sparsify](https://github.com/EleutherAI/sparsify) - Open source TopK SAE with Gao et al. recipe
- [OpenAI sparse_autoencoder](https://github.com/openai/sparse_autoencoder) - Reference implementation
- [bartbussmann/matryoshka_sae](https://github.com/bartbussmann/matryoshka_sae) - Reference AuxK implementation
