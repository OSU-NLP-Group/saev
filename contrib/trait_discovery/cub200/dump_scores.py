"""
1. Check if activations exist. If they don't, ask user to write them to disk using saev.data then try again.
2. Fit k-means (or whatever method) to dataset. Do a couple hparams in parallel because disk read speeds are slow (multiple values of k, multiple values for the number of principal components, etc).
3. Follow the pseudocode in the experiment description to get some scores.
4. Write the results to disk in a JSON or SQLite format. Tell the reader to explore the results using a marimo notebook of some kind.

Size key:

* B: Batch size
* D: ViT activation dimension (typically 768 or 1024)
* K: Number of prototypes (SAE latent dimension, k for k-means, number of principal components in PCA, etc)
* N: Number of images
* T: Number of traits in CUB-200-2011 (312)
"""

import dataclasses
import gzip
import logging
import os.path
import typing

import beartype
import numpy as np
import torch
import tyro
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

import saev.data
import saev.nn
import saev.utils.scheduling
from saev import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("baselines")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    train_data: saev.data.IterableConfig = dataclasses.field(
        default_factory=saev.data.IterableConfig
    )
    """Train activations."""
    test_data: saev.data.IterableConfig = dataclasses.field(
        default_factory=saev.data.IterableConfig
    )
    """Test activations."""
    cub_root: str = os.path.join(".", "CUB_200_2011_ImageFolder")
    """Root with test/, train/ and metadata/ folders."""
    n_samples: int = 500_000_000
    """Number of training samples (vectors)."""
    n_prototypes: int = 8 * 1024
    """"""
    dump_to: str = os.path.join(".", "data")
    """"""

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
class Scorer:
    def __call__(self, activations: Float[Tensor, "B D"]) -> Float[Tensor, "B K"]:
        raise NotImplementedError()

    @property
    def n_prototypes(self) -> int:
        raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
def get_random_vectors(
    dataloader: saev.data.iterable.DataLoader,
    *,
    n_prototypes: int,
    n_samples: int,
    seed: int,
) -> Float[Tensor, "K D"]:
    """Uniformly sample n_prototypes vectors from a streaming DataLoader using reservoir sampling.

    Args:
        dataloader: DataLoader.
        n: Number of samples to keep.
    """
    reservoir_KD = torch.empty(n_prototypes, load_metadata(dataloader.cfg).d_vit)
    n_seen = 0
    rng = torch.Generator().manual_seed(seed)

    if dataloader.n_samples > n_samples:
        dataloader = saev.utils.scheduling.BatchLimiter(dataloader, n_samples=n_samples)

    for batch in helpers.progress(dataloader):
        x_BD = batch["act"]
        b, d = x_BD.shape

        # 1. Fill reservoir if not full
        need = max(n_prototypes - n_seen, 0)
        if need:
            take = min(need, b)
            reservoir_KD[n_seen : n_seen + take] = x_BD[:take]
            n_seen += take
            x_BD = x_BD[take:]
            b -= take
            assert b >= 0  # take is at most bsz, so bsz - take >= 0
            if b == 0:
                continue

        # 2. Vectorized replacement for remaining items in batch
        idxs_B = torch.arange(n_seen, n_seen + b)  # global indices
        probs = (n_prototypes / (idxs_B + 1)).to(dtype=torch.float32)  # shape (B,)
        keep_mask = torch.rand(b, generator=rng) < probs  # Bernoulli draws
        n_keep = keep_mask.sum().item()
        if n_keep:
            replace_pos = torch.randint(0, n_prototypes, (n_keep,), generator=rng)
            reservoir_KD[replace_pos] = x_BD[keep_mask]
        n_seen += b

    return reservoir_KD


@jaxtyped(typechecker=beartype.beartype)
class RandomVectors(Scorer):
    def __init__(self, dataloader: saev.data.iterable.DataLoader, cfg: Config):
        self.prototypes = get_random_vectors(
            dataloader,
            n_prototypes=cfg.n_prototypes,
            n_samples=cfg.n_samples,
            seed=cfg.seed,
        )

    def __call__(self, activations: Float[Tensor, "B D"]) -> Float[Tensor, "B K"]:
        return activations @ self.prototypes.T

    @property
    def n_prototypes(self) -> int:
        n_prototypes, d = self.prototypes.shape
        return n_prototypes


@jaxtyped(typechecker=beartype.beartype)
def calc_scores(
    dataloader: saev.data.iterable.DataLoader,
    scorer: Scorer,
    *,
    chunk_size: int = 512,
) -> Float[Tensor, "N K"]:
    torch.use_deterministic_algorithms(True)

    metadata = load_metadata(dataloader.cfg)
    shape = (dataloader.n_samples // metadata.n_patches_per_img, scorer.n_prototypes)
    # Initialize score matrix with −inf  so max() works.
    scores_NK = torch.full(shape, -torch.inf)

    for batch in helpers.progress(dataloader, desc="scoring patches"):
        act_BD = batch["act"]
        img_i_B = batch["image_i"]

        patch_scores_BK = scorer(act_BD)
        bsz, k = patch_scores_BK.shape

        # We cannot replicate img_i_B to (B,K) in one go (memory!), so update score_NK in manageable slices.
        for start, end in helpers.batched_idx(k, chunk_size):
            # slice views avoid extra alloc
            dst = scores_NK[:, start:end]
            src = patch_scores_BK[:, start:end]

            # expand image indices once per slice
            idx = img_i_B.unsqueeze(1).expand(bsz, end - start)

            # in-place max-pool across patches → image rows
            dst.scatter_reduce_(0, idx, src, reduce="amax")

    return scores_NK.cpu()


@jaxtyped(typechecker=beartype.beartype)
def pick_best_prototypes(
    scores_NK: Float[Tensor, "N K"], y_NT: Bool[Tensor, "N T"]
) -> Int[Tensor, " T"]:
    """
    Args:
        scores_NK:
        y_NT: Boolean attribute array; y_NT[n, t] is True if image n has trait t.

    Returns:
    """
    raise NotImplementedError()


@beartype.beartype
def load_metadata(cfg: saev.data.iterable.Config) -> saev.data.Metadata:
    metadata_fpath = os.path.join(cfg.shard_root, "metadata.json")
    return saev.data.Metadata.load(metadata_fpath)


@jaxtyped(typechecker=beartype.beartype)
def load_cub_attributes(root: str, *, is_train: bool) -> Bool[Tensor, "N T"]:
    wanted_split = "train" if is_train else "test"
    image_folder_dataset = saev.data.images.ImageFolderDataset(
        os.path.join(root, wanted_split)
    )

    img_path_to_img_id = {}
    with open(os.path.join(root, "metadata", "images.txt")) as fd:
        for line in fd:
            img_id, img_path = line.split()
            img_path_to_img_id[img_path] = int(img_id)

    img_id_in_split = set()
    with open(os.path.join(root, "metadata", "train_test_split.txt")) as fd:
        for line in fd:
            img_id, is_train_str = line.split()
            if is_train_str != is_train:
                continue

            img_id_in_split.add(int(img_id))

    img_id_to_i = {}
    for i, (path, _) in enumerate(image_folder_dataset.samples):
        path, filename = os.path.split(path)
        path, cls = os.path.split(path)
        img_id = img_path_to_img_id[os.path.join(cls, filename)]
        img_id_to_i[img_id] = i

    attr_id_to_attr_name = {}
    with open(os.path.join(root, "metadata", "attributes", "attributes.txt")) as fd:
        for line in fd:
            attr_id, attr_name = line.split()
            attr_id_to_attr_name[int(attr_id)] = attr_name

    attr_id_to_i = {attr_id: i for i, attr_id in sorted(attr_id_to_attr_name.items())}

    y_NT = torch.empty((len(img_id_in_split), len(attr_id_to_attr_name)), dtype=bool)

    # From certainties.txt
    # 1 not visible
    # 2 guessing
    # 3 probably
    # 4 definitely
    certainty_keep = {3, 4}

    fpath = os.path.join(root, "metadata", "attributes", "image_attribute_labels.txt")
    with open(fpath) as fd:
        for line in fd:
            # Explanation of *_: Sometimes there's an extra field (worker_id) but not on every line. So *_ collects all extra fields besides the 5 that are manually unpacked, then we ignore it.
            img_id, attr_id, present, certainty_id, *_, time_s = line.split()
            img_id, attr_id, certainty_id = int(img_id), int(attr_id), int(certainty_id)

            if img_id not in img_id_in_split:
                continue

            i = img_id_to_i[img_id]
            j = attr_id_to_i[attr_id]
            y_NT[i, j] = certainty_id in certainty_keep

    return y_NT


@jaxtyped(typechecker=beartype.beartype)
def eval_prototypes(
    prototypes_i_T: Int[Tensor, " T"],
    scores_NK: Float[Tensor, "N K"],
    y_NT: Float[Tensor, "N T"],
) -> Float[Tensor, "N T"]:
    raise NotImplementedError()


@torch.no_grad()
@beartype.beartype
def main(cfg: typing.Annotated[Config, tyro.conf.arg(name="")]):
    try:
        train_y_NT = load_cub_attributes(cfg.cub_root, is_train=True)
        test_y_NT = load_cub_attributes(cfg.cub_root, is_train=False)
    except Exception:
        logger.exception("Could not load CUB attributes.")
        return
    try:
        train_dataloader = saev.data.iterable.DataLoader(cfg.train_data)
        test_dataloader = saev.data.iterable.DataLoader(cfg.test_data)
    except Exception:
        logger.exception(
            "Could not create dataloader. Please create a dataset using saev.data first. See src/saev/guide.md for more details."
        )
        return

    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    scorer = RandomVectors(train_dataloader, cfg)
    train_scores_NK = calc_scores(train_dataloader, scorer)
    test_scores_NK = calc_scores(test_dataloader, scorer)

    prototypes_i_T = pick_best_prototypes(train_scores_NK, train_y_NT)
    test_scores_NT = eval_prototypes(prototypes_i_T, test_scores_NK, test_y_NT)

    with gzip.open(os.path.join(cfg.dump_to, "randvec-scores_NT.bin.gz"), "wb") as fd:
        np.save(fd, test_scores_NT.numpy())


if __name__ == "__main__":
    tyro.cli(main)
