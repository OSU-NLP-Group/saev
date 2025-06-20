import beartype
import torch

import saev.utils.scheduling


@beartype.beartype
def kmeans(
    loader: saev.utils.scheduling.DataLoaderLike,
    ks: list[int],
    d: int,
    n_samples: int = 500_000_000,
    device: str = "cuda",
):
    # TODO: instead of `for x in loader`, do `for x in max_examples(loader, n_samples)` so that we see n_samples samples.
    centroids = [torch.randn(k_i, d, device=device) for k_i in ks]
    accums = [torch.zeros_like(C) for C in centroids]
    counts = [torch.zeros(len(C), dtype=torch.int32, device=device) for C in centroids]

    for x in loader:
        x = x.to(device)
        x2 = (x * x).sum(1, keepdim=True)
        for C, delta, n in zip(centroids, accums, counts):
            c2 = (C * C).sum(1).unsqueeze(0)  # 1×k
            idx = (x2 + c2 - 2 * x @ C.T).argmin(1)  # B
            delta.scatter_add_(0, idx[:, None], x)
            n.scatter_add_(0, idx, 1)

    for C, delta, n in zip(centroids, accums, counts):
        mask = n > 0
        C[mask] = delta[mask] / n[mask][:, None]
        delta.zero_()
        n.zero_()


@beartype.beartype
def main():
    pass


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
