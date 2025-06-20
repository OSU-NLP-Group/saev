"""
1. Check if activations exist. If they don't, ask user to write them to disk using saev.data then try again.
2. Fit k-means (or whatever method) to dataset. Do a couple hparams in parallel because disk read speeds are slow (multiple values of k, multiple values for the number of principal components, etc).
3. Follow the pseudocode in the experiment description to get some scores.
4. Write the results to disk in a JSON or SQLite format. Tell the reader to explore the results using a marimo notebook of some kind.
"""

import dataclasses
import json
import logging
import os.path
import typing

import beartype
import torch
import tyro

import saev.data
import saev.nn
import saev.utils.scheduling
from saev import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


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
def kmeans(loader: saev.data.iterable.DataLoader, cfg: Config):
    """
    Run k-means clustering on the data from the loader.

    Args:
        loader: DataLoader containing the activations
        cfg: Configuration object

    Returns:
        List of centroids for different k values
    """
    logger.info("Starting k-means clustering")

    # Define the range of k values to try
    ks = [50, 100, 200, 500]
    device = torch.device(cfg.device)

    # Get the dimension from the first batch
    for batch in loader:
        d = batch["act"].shape[1]
        break

    # Initialize centroids randomly
    centroids = [torch.randn(k_i, d, device=device) for k_i in ks]
    accums = [torch.zeros_like(C) for C in centroids]
    counts = [torch.zeros(len(C), dtype=torch.int32, device=device) for C in centroids]

    # Limit the number of examples we process
    n_samples = 100_000
    samples_seen = 0

    # Run k-means iterations
    for batch in helpers.progress(loader, desc="k-means clustering"):
        x = batch["act"].to(device)
        samples_seen += x.shape[0]

        # Compute squared norm of x
        x2 = (x * x).sum(1, keepdim=True)

        # Update centroids
        for C, delta, n in zip(centroids, accums, counts):
            # Compute squared norm of centroids
            c2 = (C * C).sum(1).unsqueeze(0)  # 1×k

            # Compute distances and assign points to nearest centroid
            idx = (x2 + c2 - 2 * x @ C.T).argmin(1)  # B

            # Update accumulators
            delta.scatter_add_(0, idx[:, None].expand(-1, d), x)
            n.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))

        # Stop if we've seen enough samples
        if samples_seen >= n_samples:
            break

    # Update centroids based on accumulated values
    for C, delta, n in zip(centroids, accums, counts):
        mask = n > 0
        C[mask] = delta[mask] / n[mask].unsqueeze(1)

    logger.info(f"K-means clustering complete for k values: {ks}")
    return centroids, ks


@beartype.beartype
def main(cfg: typing.Annotated[Config, tyro.conf.arg(name="")]):
    try:
        dataloader = saev.data.iterable.DataLoader(cfg.data)
    except Exception as err:
        logger.exception(
            "Could not create dataloader. Please create a dataset using saev.data first."
        )
        logger.info(
            "To create a dataset, run a command like:\n"
            "uv run python -m saev.data \\\n"
            "  --vit-family <model_family> \\\n"
            "  --vit-ckpt <model_checkpoint> \\\n"
            "  --d-vit <dimension> \\\n"
            "  --vit-layers <layer_numbers> \\\n"
            "  --dump-to <output_directory> \\\n"
            "  data:<dataset_type> \\\n"
            "  --data.root <dataset_path>\n"
            "See src/saev/guide.md for more details on creating datasets."
        )
        # Also log the error, because the error might not just be that we're missing shards; maybe it's some genuine runtime/valueerror/bug in our code. AI!
        return

    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    # Run k-means clustering
    centroids, k_values = kmeans(dataloader, cfg)

    # Save results to disk
    os.makedirs(cfg.log_to, exist_ok=True)
    results = {
        "config": dataclasses.asdict(cfg),
        "k_values": k_values,
        "timestamp": torch.datetime.now().isoformat(),
    }

    # Save centroids for each k value
    for k, centroid in zip(k_values, centroids):
        output_path = os.path.join(cfg.log_to, f"centroids_k{k}.pt")
        torch.save(centroid.cpu(), output_path)
        logger.info(f"Saved centroids for k={k} to {output_path}")

    # Save metadata
    metadata_path = os.path.join(cfg.log_to, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")
    logger.info(
        "To explore results, create a marimo notebook that loads and visualizes the centroids"
    )


if __name__ == "__main__":
    tyro.cli(main)
