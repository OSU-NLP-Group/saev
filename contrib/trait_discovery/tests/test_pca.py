"""Unit tests for the MiniBatchPCA baseline."""

import itertools
import math

import torch
import torch.nn.functional as F
from tdiscovery.baselines import MiniBatchPCA
from torch import Tensor


def _generate_low_rank_gaussian(
    *,
    n_samples: int = 2048,
    n_features: int = 6,
    n_components: int = 3,
    noise_std: float = 0.01,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    torch.manual_seed(seed)
    base = torch.randn(n_features, n_components)
    q, _ = torch.linalg.qr(base, mode="reduced")
    components = q.mT.contiguous()
    scales = torch.linspace(2.5, 1.0, n_components)
    scores = torch.randn(n_samples, n_components) * scales
    noise = noise_std * torch.randn(n_samples, n_features)
    data = scores @ components + noise
    return data.to(torch.float32), components.to(torch.float32)


def _match_components(similarity: Tensor) -> tuple[int, ...]:
    best_score = float("-inf")
    best_perm: tuple[int, ...] | None = None
    rows, cols = similarity.shape
    assert rows == cols, "Similarity must be square"
    indices = range(cols)
    for perm in itertools.permutations(indices, rows):
        diag = similarity[torch.arange(rows), torch.tensor(perm)]
        score = float(diag.sum().item())
        if score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None
    return best_perm


def _fit_on_low_rank_gaussian(
    *, epochs: int = 12, batch_size: int = 64
) -> tuple[MiniBatchPCA, Tensor, Tensor, int]:
    data, components = _generate_low_rank_gaussian()
    model = MiniBatchPCA(n_components=components.shape[0], device="cpu")
    for _ in range(epochs):
        perm = torch.randperm(data.shape[0])
        for batch in data[perm].split(batch_size):
            model.partial_fit(batch)
    n_batches = math.ceil(data.shape[0] / batch_size)
    expected_steps = epochs * n_batches
    return model, data, components, expected_steps


def test_minibatch_pca_components_match_true_basis():
    model, _, true_components, expected_steps = _fit_on_low_rank_gaussian()
    assert model.components_ is not None
    assert model.mean_ is not None
    learned = F.normalize(model.components_.cpu(), dim=1)
    truth = F.normalize(true_components, dim=1)
    similarities = torch.abs(learned @ truth.T)
    assert similarities.shape == (truth.shape[0], truth.shape[0])
    per_component = similarities.max(dim=1).values
    assert torch.all(per_component > 0.98)
    assert model.components_.shape == truth.shape
    assert model.n_steps_ == expected_steps


def test_minibatch_pca_transform_recovers_latent_scores_on_new_data():
    model, _, true_components, _ = _fit_on_low_rank_gaussian()
    assert model.components_ is not None
    assert model.mean_ is not None
    held_out, _ = _generate_low_rank_gaussian(seed=123)

    learned = F.normalize(model.components_.cpu(), dim=1)
    truth = F.normalize(true_components, dim=1)
    similarities = torch.abs(learned @ truth.T)
    perm = _match_components(similarities)
    permutation = torch.tensor(perm, dtype=torch.long)
    truth_aligned = truth[permutation]

    signs = torch.sign((learned * truth_aligned).sum(dim=1))
    signs[signs == 0] = 1.0

    aligned_scores = model.transform(held_out).cpu() * signs.unsqueeze(0)
    true_centered = held_out - model.mean_.cpu()
    true_scores = true_centered @ truth_aligned.T
    error = torch.mean((aligned_scores - true_scores) ** 2).sqrt()
    assert error < 0.05
