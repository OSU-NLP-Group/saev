import dataclasses
import logging
import pathlib
import typing as tp

import beartype
import numpy as np
import scipy.sparse
import torch
import tyro
from tdiscovery.probe1d import Sparse1DProbe
from torch import Tensor

import saev.data
import saev.disk

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("probe1d")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    run: pathlib.Path = pathlib.Path("./runs/abcdefg")
    """Run directory."""
    shards_dir: pathlib.Path = pathlib.Path("./shards/e967c008")
    """Shards directory."""
    device: str = "cuda"
    """Hardware device."""


def sp_csr_to_pt(csr: scipy.sparse.sparray, *, device: str) -> Tensor:
    return torch.sparse_csr_tensor(csr.indptr, csr.indices, csr.data, device=device)


@beartype.beartype
def main(cfg: tp.Annotated[Config, tyro.conf.arg(name="")]) -> int:
    logger.info("Started main().")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA device available, using CPU.")
        cfg = dataclasses.replace(cfg, device="cpu")

    if not (cfg.shards_dir / "labels.bin").exists():
        logger.error("--shards-dir %s doesn't have a labels.bin.", cfg.shards_dir)
        return 1

    run = saev.disk.Run(cfg.run)

    if not (run.inference / cfg.shards_dir.name).exists():
        logger.error(
            "Directory %s doesn't exist. Use inference.py to run inference.",
            run.inference / cfg.shards_dir.name,
        )
        return 1

    # Load metadata
    md = saev.data.Metadata.load(cfg.shards_dir)
    logger.info("Loaded metadata from %s.", cfg.shards_dir)

    # Load SAE activations (sparse matrix)
    token_acts = scipy.sparse.load_npz(
        run.inference / cfg.shards_dir.name / "token_acts.npz"
    )
    logger.info(
        "Loaded activations: shape=%s, nnz=%d.", token_acts.shape, token_acts.nnz
    )
    n_samples, n_latents = token_acts.shape
    token_acts = sp_csr_to_pt(token_acts, device=cfg.device)
    logger.info("Converted activations to Tensor on %s.", cfg.device)

    # Load patch labels from labels.bin
    labels = np.memmap(
        cfg.shards_dir / "labels.bin",
        mode="r",
        dtype=np.uint8,
        shape=(md.n_examples, md.content_tokens_per_example),
    )
    logger.info("Loaded labels: shape=%s.", labels.shape)

    # Flatten labels to (n_samples,) and convert to one-hot
    n_classes = int(labels.max()) + 1
    logger.info("Found %d classes in labels.", n_classes)

    # Convert to one-hot encoding
    y = np.zeros((n_samples, n_classes), dtype=float)
    y[np.arange(n_samples), labels.reshape(n_samples)] = 1.0
    y = torch.from_numpy(y)
    logger.info("Created one-hot labels: shape=%s.", y.shape)

    # Fit probe
    probe = Sparse1DProbe(
        n_latents=n_latents, n_classes=n_classes, device=cfg.device, ridge=1e-8
    )
    logger.info("Fitting probe with %d latents and %d classes.", n_latents, n_classes)
    probe.fit(token_acts, y)
    logger.info("Fit probe.")

    # Compute loss
    loss_matrix = probe.loss_matrix(token_acts, y)
    mean_loss = loss_matrix.mean().item()
    logger.info("Mean loss across all (latent, class) pairs: %.6f", mean_loss)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Config)))
