"""
1. Check if activations exist. If they don't, ask user to write them to disk using saev.data then try again.
2. Fit k-means (or whatever method) to dataset. Do a couple hparams in parallel because disk read speeds are slow (multiple values of k, multiple values for the number of principal components, etc).
3. Follow the pseudocode in the experiment description to get some scores.
4. Write the results to disk in a JSON or SQLite format. Tell the reader to explore the results using a marimo notebook of some kind.
"""

import dataclasses
import os.path
import typing

import beartype
import torch
import tyro

import saev.data
import saev.nn
import saev.utils.scheduling


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    data: saev.data.IterableConfig = dataclasses.field(
        default_factory=saev.data.IterableConfig
    )
    """Data configuration"""
    n_patches: int = 500_000_000
    """Number of training examples."""

    device: typing.Literal["cuda", "cpu"] = "cuda"
    """Hardware device."""
    seed: int = 42
    """Random seed."""
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 24.0
    """Slurm job length in hours."""
    log_to: str = os.path.join(".", "logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
def kmeans(loader: saev.utils.scheduling.DataLoaderLike, cfg: Config):
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
def main(cfg: typing.Annotated[Config, tyro.conf.arg(name="")]):
    try:
        dataloader = saev.data.iterable.DataLoader(cfg.data)
    except Exception as err:
        # Log an exception here. Check train.py for my logging conventions. Ask the user to create a datasaet using saev.data; check guide.md for how to do that. AI!
        logger.exception("Could not create dataloader.
    pass


if __name__ == "__main__":
    tyro.cli(main)
