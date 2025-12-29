"""
We evaluate whether SAE latents can serve as (i) effective inputs for sparse downstream prediction and (ii) a feature library whose task-selected features correspond to coherent, localized visual concepts not present in the label ontology used for feature selection.
We instantiate this "task-driven feature selection + concept audit" protocol on a general-domain pipeline (ImageNet → Places365 → ADE20K) and on a scientific pipeline (FishVista classification → FishVista segmentation + FishBase ecology).

1. Train an SAE on a large dataset.
2. Use SAE features as inputs to a shallow or sparse classifier.
3. For the most important features, measure how well they correlate with part-level labels (segmentation masks).

We can use this task to find new part-level concepts that are useful for classification but unknown to human scientists.

This module trains sparse classifiers and evaluates feature quality using different, non-probing metrics.

TODO:
4. Implement DecisionTree using sklearn's DecisionTreeClassifier (relevant for dichotomous keys in biology)
6. Implement eval metrics (long-term: unify with metrics.py to compare all methods):
   - Localization AP
   - Pointing game: whether max activation patch falls inside the object mask
   - IoU / Dice
   - Yield@K: fraction of task-selected latents whose held-out score (localization, pointing, etc.) exceeds threshold
9. Implement eval_cli for standalone evaluation

NOTES:

FishVista segfolder dataset with multiple image-level labels:
- Dataset: /fs/scratch/PAS2136/samuelstevens/derived-datasets/fish-vista-segfolder2/
- Train shards: e65cf404
- Val shards: b8a9ff56

Example of getting Places365:

```py
ds = torchvision.datasets.Places365('/fs/scratch/PAS2136/samuelstevens/derived-datasets/places365', small=True, download=True)
```
"""

import dataclasses
import enum
import logging
import pathlib
import time
import typing as tp

import beartype
import cloudpickle
import numpy as np
import orjson
import scipy.sparse
import sklearn.linear_model
import sklearn.tree
import tyro
from jaxtyping import Float, Int, jaxtyped

import saev.configs
import saev.data
import saev.data.datasets
import saev.disk
import saev.helpers


class PatchAgg(enum.Enum):
    """How to aggregate patch-level features to image-level."""

    MEAN = "mean"
    MAX = "max"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class LabelGrouping:
    """Defines a classification task by grouping labels.

    If groups is empty, use original labels directly (no grouping).
    If groups is non-empty, map original labels to group names.

    Examples:
        # No grouping - use original habitat labels
        LabelGrouping(name="habitat", source_col="habitat", groups={})

        # Binary grouping
        LabelGrouping(
            name="habitat_speed",
            source_col="habitat",
            groups={
                "low": ["reef-associated", "demersal"],
                "high": ["pelagic", "oceanic"],
            }
        )

        # Ternary grouping
        LabelGrouping(
            name="depth_zone",
            source_col="habitat",
            groups={
                "shallow": ["reef-associated"],
                "mid": ["benthopelagic", "pelagic"],
                "deep": ["bathypelagic"],
            }
        )

        # ImgFolder class grouping (butterflies)
        LabelGrouping(
            name="mimic_pair",
            source_col="class",
            groups={
                "melpomene": ["Heliconius melpomene", "Heliconius melpomene ssp. rosina"],
                "erato": ["Heliconius erato"],
            }
        )
    """

    name: str
    """Task name for output files."""
    source_col: str
    """Label column to read. Use 'class' for ImgFolder class names."""
    groups: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    """Mapping from group name to list of original labels. Empty = no grouping."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class DecisionTree:
    """Decision tree classifier using sklearn's DecisionTreeClassifier."""

    key: tyro.conf.Suppress[tp.Literal["decision-tree"]] = "decision-tree"

    max_depth: int = -1
    """Maximum depth of the tree. Negative numbers mean unlimited."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SparseLinear:
    """Sparse linear classifier using Logistic Regression with L1 penalty."""

    key: tyro.conf.Suppress[tp.Literal["sparse-linear"]] = "sparse-linear"

    C: float = 0.01
    """Inverse regularization strength. Lower C = more sparsity."""

    max_iter: int = 90
    """Maximum iterations for solver convergence."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    run: pathlib.Path = pathlib.Path("./runs/abcdefg")
    """Run directory."""
    train_shards: pathlib.Path = pathlib.Path("./shards/01234567")
    """Training shards directory."""
    test_shards: pathlib.Path = pathlib.Path("./shards/abcdef01")
    """Test shards directory."""
    task: LabelGrouping = dataclasses.field(
        default_factory=lambda: LabelGrouping(name="habitat", source_col="habitat")
    )
    """Classification task definition."""
    patch_agg: tyro.conf.EnumChoicesFromValues[PatchAgg] = PatchAgg.MAX
    """How to aggregate patch features to image features."""
    cls: DecisionTree | SparseLinear = SparseLinear()
    """Classifier architecture."""
    debug: bool = False
    """Debug logging."""
    # Hardware
    mem_gb: int = 80
    """Node memory in GB."""
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 4.0
    """Slurm job length in hours."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
def load_image_labels(
    shards_dpath: pathlib.Path,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Load image-level labels from the dataset stored in shards metadata.

    Supports both ImgSegFolder (multiple label columns from CSV) and ImgFolder (class names from folder structure).

    Args:
        shards_dpath: Path to shards directory containing metadata.json.

    Returns:
        Tuple of (label_columns, labels_dict) where labels_dict maps column name to list of string labels (one per image).
    """
    md = saev.data.Metadata.load(shards_dpath)
    data_cfg = md.make_data_cfg()

    if isinstance(data_cfg, saev.data.datasets.ImgSegFolder):
        ds = saev.data.datasets.ImgSegFolderDataset(
            data_cfg, img_transform=None, mask_transform=None
        )
        if not ds.samples:
            raise ValueError("Dataset has no samples")

        label_cols = list(ds.samples[0].labels.keys())
        labels: dict[str, list[str]] = {col: [] for col in label_cols}
        for sample in ds.samples:
            for col in label_cols:
                labels[col].append(sample.labels[col])

        return label_cols, labels

    elif isinstance(data_cfg, saev.data.datasets.ImgFolder):
        ds = saev.data.datasets.get_dataset(data_cfg)
        # ImgFolder has .classes and .targets
        label_cols = ["class"]
        class_labels = [ds.classes[t] for t in ds.targets]
        return label_cols, {"class": class_labels}

    else:
        raise ValueError(f"Unsupported dataset type: {type(data_cfg)}")


@beartype.beartype
def apply_grouping(
    labels: list[str],
    task: LabelGrouping,
) -> tuple[Int[np.ndarray, " n"], list[str], np.ndarray]:
    """
    Apply label grouping to get integer targets.

    Args:
        labels: List of string labels (one per image).
        task: Label grouping configuration.

    Returns:
        Tuple of (targets, class_names, mask) where:
        - targets: Integer targets for each image (-1 if excluded)
        - class_names: List of class names in order
        - mask: Boolean mask of which images to include
    """
    n = len(labels)

    if not task.groups:
        # No grouping - use original labels
        unique_labels = sorted(set(labels))
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        targets = np.array([label_to_idx[lbl] for lbl in labels], dtype=np.int32)
        mask = np.ones(n, dtype=bool)
        return targets, unique_labels, mask

    # Build reverse mapping: original_label -> group_name
    label_to_group: dict[str, str] = {}
    for group_name, group_labels in task.groups.items():
        for lbl in group_labels:
            assert lbl not in label_to_group, f"Label '{lbl}' in multiple groups"
            label_to_group[lbl] = group_name

    # Map to group names, filter out unlisted labels
    group_names = sorted(task.groups.keys())
    group_to_idx = {g: i for i, g in enumerate(group_names)}

    targets = np.full(n, -1, dtype=np.int32)
    mask = np.zeros(n, dtype=bool)

    for i, lbl in enumerate(labels):
        if lbl in label_to_group:
            group = label_to_group[lbl]
            targets[i] = group_to_idx[group]
            mask[i] = True

    return targets, group_names, mask


@jaxtyped(typechecker=beartype.beartype)
def aggregate_to_images(
    patch_acts: scipy.sparse.csr_matrix | scipy.sparse.csr_array,
    n_images: int,
    n_patches_per_image: int,
    patch_agg: PatchAgg,
) -> Float[np.ndarray, "n_images d_sae"]:
    """
    Aggregate patch-level sparse features to image-level dense features.

    Args:
        patch_acts: Sparse CSR matrix of shape (n_patches, d_sae).
        n_images: Number of images.
        n_patches_per_image: Number of patches per image.
        agg: Aggregation method (mean, max, sum).

    Returns:
        Dense array of shape (n_images, d_sae) with aggregated features.
    """
    n_patches, d_sae = patch_acts.shape
    assert n_patches == n_images * n_patches_per_image

    # Reshape to (n_images, n_patches_per_image, d_sae) by processing each image
    result = np.zeros((n_images, d_sae), dtype=np.float32)

    for i in range(n_images):
        start = i * n_patches_per_image
        end = start + n_patches_per_image
        img_patches = patch_acts[start:end].toarray()  # (n_patches_per_image, d_sae)

        if patch_agg == PatchAgg.MEAN:
            result[i] = img_patches.mean(axis=0)
        elif patch_agg == PatchAgg.MAX:
            result[i] = img_patches.max(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {patch_agg}")

    return result


@beartype.beartype
def train_worker_fn(cfg: TrainConfig) -> int:
    """Main training worker function."""
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, force=True)
    logger = logging.getLogger("cls::train")
    logger.info("Started train_worker_fn().")

    run = saev.disk.Run(cfg.run)

    # Validate paths
    train_shards_dpath = cfg.train_shards
    msg = f"Train shards directory {train_shards_dpath} does not exist."
    assert train_shards_dpath.exists(), msg

    test_shards_dpath = cfg.test_shards
    msg = f"Test shards directory {test_shards_dpath} does not exist."
    assert test_shards_dpath.exists(), msg

    train_inference_dpath = run.inference / train_shards_dpath.name
    msg = f"Train inference directory {train_inference_dpath} doesn't exist. Run inference.py."
    assert train_inference_dpath.exists(), msg

    test_inference_dpath = run.inference / test_shards_dpath.name
    msg = f"Test inference directory {test_inference_dpath} doesn't exist. Run inference.py."
    assert test_inference_dpath.exists(), msg

    train_token_acts_fpath = train_inference_dpath / "token_acts.npz"
    msg = f"Train SAE acts missing: '{train_token_acts_fpath}'. Run inference.py."
    assert train_token_acts_fpath.exists(), msg

    test_token_acts_fpath = test_inference_dpath / "token_acts.npz"
    msg = f"Test SAE acts missing: '{test_token_acts_fpath}'. Run inference.py."
    assert test_token_acts_fpath.exists(), msg

    # Load metadata
    train_md = saev.data.Metadata.load(train_shards_dpath)
    test_md = saev.data.Metadata.load(test_shards_dpath)
    logger.info(
        "Loaded metadata: train=%d images, test=%d images.",
        train_md.n_examples,
        test_md.n_examples,
    )

    # Load image-level labels
    logger.info("Loading image-level labels from train shards...")
    train_label_cols, train_labels = load_image_labels(train_shards_dpath)
    logger.info("Train label columns: %s", train_label_cols)

    logger.info("Loading image-level labels from test shards...")
    test_label_cols, test_labels = load_image_labels(test_shards_dpath)
    logger.info("Test label columns: %s", test_label_cols)

    source_col = cfg.task.source_col
    msg = f"Source column '{source_col}' not in train labels: {train_label_cols}"
    assert source_col in train_labels, msg
    msg = f"Source column '{source_col}' not in test labels: {test_label_cols}"
    assert source_col in test_labels, msg

    # Apply grouping to get targets
    train_targets, class_names, train_mask = apply_grouping(
        train_labels[source_col], cfg.task
    )
    test_targets, _, test_mask = apply_grouping(test_labels[source_col], cfg.task)

    n_classes = len(class_names)
    logger.info("Task '%s': %d classes %s", cfg.task.name, n_classes, class_names)
    logger.info(
        "Train: %d/%d images after filtering", train_mask.sum(), len(train_mask)
    )
    logger.info("Test: %d/%d images after filtering", test_mask.sum(), len(test_mask))

    # Load SAE activations (sparse CSR)
    logger.info("Loading train SAE activations from %s...", train_token_acts_fpath)
    train_acts_csr = scipy.sparse.load_npz(train_token_acts_fpath)
    logger.info(
        "Train acts: shape=%s, nnz=%d", train_acts_csr.shape, train_acts_csr.nnz
    )

    logger.info("Loading test SAE activations from %s...", test_token_acts_fpath)
    test_acts_csr = scipy.sparse.load_npz(test_token_acts_fpath)
    logger.info("Test acts: shape=%s, nnz=%d", test_acts_csr.shape, test_acts_csr.nnz)

    # Validate dimensions
    train_n_patches_per_image = train_md.content_tokens_per_example
    train_n_patches_expected = train_md.n_examples * train_n_patches_per_image
    msg = f"Train acts have {train_acts_csr.shape[0]} rows, expected {train_n_patches_expected}"
    assert train_acts_csr.shape[0] == train_n_patches_expected, msg

    test_n_patches_per_image = test_md.content_tokens_per_example
    test_n_patches_expected = test_md.n_examples * test_n_patches_per_image
    msg = f"Test acts have {test_acts_csr.shape[0]} rows, expected {test_n_patches_expected}"
    assert test_acts_csr.shape[0] == test_n_patches_expected, msg

    # Aggregate patch features to image features
    logger.info("Aggregating train patches to images using %s...", cfg.patch_agg.value)
    train_X_all = aggregate_to_images(
        train_acts_csr, train_md.n_examples, train_n_patches_per_image, cfg.patch_agg
    )
    logger.info("Train features (all): shape=%s", train_X_all.shape)

    logger.info("Aggregating test patches to images using %s...", cfg.patch_agg.value)
    test_X_all = aggregate_to_images(
        test_acts_csr, test_md.n_examples, test_n_patches_per_image, cfg.patch_agg
    )
    logger.info("Test features (all): shape=%s", test_X_all.shape)

    # Apply mask to filter images
    train_X = train_X_all[train_mask]
    train_y = train_targets[train_mask]
    test_X = test_X_all[test_mask]
    test_y = test_targets[test_mask]

    logger.info("Train features (filtered): shape=%s", train_X.shape)
    logger.info("Test features (filtered): shape=%s", test_X.shape)

    # Train classifier
    if isinstance(cfg.cls, SparseLinear):
        logger.info(
            "Training SparseLinear (LogisticRegression L1) with C=%g...", cfg.cls.C
        )
        classifier = sklearn.linear_model.LogisticRegression(
            penalty="l1",
            C=cfg.cls.C,
            solver="saga",
            max_iter=cfg.cls.max_iter,
            tol=3e-4,
            verbose=1,
        )
        classifier.fit(train_X, train_y)
    elif isinstance(cfg.cls, DecisionTree):
        max_depth = cfg.cls.max_depth if cfg.cls.max_depth > 0 else None
        logger.info("Training DecisionTree with max_depth=%s...", max_depth)
        classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth)
        classifier.fit(train_X, train_y)
    else:
        raise ValueError(f"Unknown classifier type: {type(cfg.cls)}")

    logger.info("Training complete.")

    # Evaluate
    test_pred = classifier.predict(test_X)
    test_acc = (test_pred == test_y).mean()
    logger.info("Test accuracy: %.4f", test_acc)

    # Log feature sparsity
    if isinstance(cfg.cls, SparseLinear):
        nonzero_mask = np.any(classifier.coef_ != 0, axis=0)
        nonzero_i = np.where(nonzero_mask)[0]
        logger.info("L1 selected %d features: %s", len(nonzero_i), nonzero_i.tolist())
    elif isinstance(cfg.cls, DecisionTree):
        importances = classifier.feature_importances_
        top_k = min(10, (importances > 0).sum())
        top_i = np.argsort(importances)[-top_k:][::-1]
        logger.info(
            "Tree uses %d features, top %d: %s",
            (importances > 0).sum(),
            top_k,
            top_i.tolist(),
        )

    # Save checkpoint: JSON header on first line, then pickle
    if isinstance(cfg.cls, SparseLinear):
        cls_str = f"C{cfg.cls.C}"
    else:
        cls_str = f"depth{cfg.cls.max_depth}"
    out_fpath = (
        test_inference_dpath
        / f"cls_{cfg.task.name}_{cfg.patch_agg.value}_{cls_str}.pkl"
    )
    header = {
        "cfg": dataclasses.asdict(cfg),
        "test_acc": float(test_acc),
        "n_classes": int(n_classes),
        "class_names": class_names,
    }
    with open(out_fpath, "wb") as fd:
        saev.helpers.jdump(header, fd, option=orjson.OPT_APPEND_NEWLINE)
        cloudpickle.dump(
            {"classifier": classifier, "test_pred": test_pred, "test_y": test_y}, fd
        )
    logger.info("Saved checkpoint to %s", out_fpath)

    return 0


@beartype.beartype
def train_cli(
    cfg: tp.Annotated[TrainConfig, tyro.conf.arg(name="")],
    sweep: pathlib.Path | None = None,
) -> int:
    """
    Train a sparse classifier on SAE activations for image-level prediction.

    Args:
        cfg: Training configuration.
        sweep: Optional sweep file that expands into many configs.
    """
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, force=True)
    logger = logging.getLogger("cls::main")
    logger.info("Started train_cli().")

    if sweep is None:
        cfgs = [cfg]
    else:
        sweep_dcts = saev.configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            return 1

        cfgs, errs = saev.configs.load_cfgs(
            cfg, default=TrainConfig(), sweep_dcts=sweep_dcts
        )
        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return 1

    if not cfgs:
        logger.error("No configs resolved for classification sweep.")
        return 1

    logger.info("Prepared %d config(s).", len(cfgs))

    base_cfg = cfgs[0]
    if not base_cfg.slurm_acct:
        for idx, c in enumerate(cfgs, start=1):
            logger.info("Running config %d/%d locally.", idx, len(cfgs))
            train_worker_fn(c)
        logger.info("Jobs done.")
        return 0

    import submitit

    executor = submitit.SlurmExecutor(folder=base_cfg.log_to)
    n_cpus = max(8, base_cfg.mem_gb // 10)
    executor.update_parameters(
        time=int(base_cfg.n_hours * 60),
        partition=base_cfg.slurm_partition,
        gpus_per_node=0,  # CPU only for OMP
        ntasks_per_node=1,
        cpus_per_task=n_cpus,
        stderr_to_stdout=True,
        account=base_cfg.slurm_acct,
    )

    try:
        cloudpickle.dumps(train_worker_fn)
        for c in cfgs:
            cloudpickle.dumps(c)
    except TypeError as err:
        logger.error("Failed to pickle job payload: %s", err)
        return 1

    for idx, _ in saev.helpers.submit_job_array(
        executor, train_worker_fn, cfgs, logger=logger
    ):
        logger.info("Job %d/%d finished.", idx + 1, len(cfgs))

    logger.info("Jobs done.")
    return 0


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class EvalConfig:
    pass


@beartype.beartype
def eval_cli(
    cfg: tp.Annotated[EvalConfig, tyro.conf.arg(name="")],
    sweep: pathlib.Path | None = None,
): ...
