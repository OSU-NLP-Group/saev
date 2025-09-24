import dataclasses
import logging
import os.path
import typing as tp

import beartype
import scipy.sparse
import torch
import tyro
from torch import Tensor

# from tdiscovery import probe1d

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("visuals")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    root: str = os.path.join(".", "acts", "ade20k", "abcdefg")
    """Path to the saved SAE image activations."""
    device: str = "cuda"
    """Hardware device."""


def sp_csr_to_pt(csr: scipy.sparse.sparray, *, device: str) -> Tensor:
    return torch.sparse_csr_tensor(csr.indptr, csr.indices, csr.data, device=device)


@beartype.beartype
def main(cfg: tp.Annotated[Config, tyro.conf.arg(name="")]):
    logger.info("Started main().")
    acts_ns = scipy.sparse.load_npz(os.path.join(cfg.root, "patch_acts.npz"))
    logger.info("Loaded activations on cpu.")
    acts_ns = sp_csr_to_pt(acts_ns, device=cfg.device)
    logger.info("Convert activations to Tensor on %s.", cfg.device)

    # Now I need the patch labels.
    breakpoint()
