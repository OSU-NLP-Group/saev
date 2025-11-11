"""Unit tests for the MiniBatchKMeans baseline."""

import math

import torch
from tdiscovery.baselines import MiniBatchKMeans
from torch import Tensor


def _generate_separable_blobs(
    *, n_per_cluster: int = 32, noise_std: float = 0.02, seed: int = 0
):
    torch.manual_seed(seed)
    centers = torch.tensor(
        [
            [-5.0, -5.0, 0.0],
            [0.0, 5.0, 5.0],
            [6.0, -1.0, -4.0],
        ],
        dtype=torch.float32,
    )
    points: list[Tensor] = []
    labels: list[Tensor] = []
    for idx, center in enumerate(centers):
        noise = noise_std * torch.randn(n_per_cluster, center.numel())
        points.append(center.unsqueeze(0) + noise)
        labels.append(torch.full((n_per_cluster,), idx, dtype=torch.long))

    return torch.cat(points, dim=0), torch.cat(labels, dim=0), centers


def _fit_on_blobs(
    *, epochs: int = 12, batch_size: int = 16
) -> tuple[MiniBatchKMeans, Tensor, int]:
    data, _, centers = _generate_separable_blobs()
    model = MiniBatchKMeans(k=len(centers), device="cpu")
    for _ in range(epochs):
        perm = torch.randperm(data.shape[0])
        for batch in data[perm].split(batch_size):
            model.partial_fit(batch)

    assert model.cluster_centers_ is not None
    n_batches = math.ceil(data.shape[0] / batch_size)
    expected_steps = epochs * n_batches
    return model, centers, expected_steps


def test_minibatch_kmeans_centers_match_true_means() -> None:
    model, centers, expected_steps = _fit_on_blobs()
    assert model.cluster_centers_ is not None
    learned = model.cluster_centers_.cpu()
    distances = torch.cdist(learned, centers)
    # Every ground-truth mean must be within 0.2 L2 distance of some center.
    assert torch.all(distances.min(dim=0).values < 0.2)
    assert model.n_steps_ == expected_steps
    assert model.n_features_in_ == centers.shape[1]


def test_minibatch_kmeans_assigns_correct_clusters_on_new_points() -> None:
    model, centers, _ = _fit_on_blobs()
    assert model.cluster_centers_ is not None
    held_out, labels, _ = _generate_separable_blobs(seed=123, noise_std=0.015)
    scores = model.transform(held_out)
    assert scores.shape == (held_out.shape[0], centers.shape[0])

    center_to_label = torch.cdist(model.cluster_centers_.cpu(), centers).argmin(dim=1)
    predicted = center_to_label[scores.argmax(dim=1)]
    accuracy = (predicted == labels).float().mean().item()
    assert accuracy > 0.9
