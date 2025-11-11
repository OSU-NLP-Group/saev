import dataclasses
import logging
import pathlib
import time
import typing as tp

import beartype
import cloudpickle
import orjson
import scipy.sparse
import submitit
import torch
import tyro
import wandb
from jaxtyping import Float, jaxtyped
from submitit.core.utils import UncompletedJobError
from torch import Tensor

import saev.configs
import saev.data
import saev.helpers
import saev.utils.scheduling

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

BaselineMethod = tp.Literal["kmeans", "pca"]


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class BaselineRun:
    run_dir: pathlib.Path

    @classmethod
    def new(
        cls,
        run_id: str,
        *,
        train_shards_dir: pathlib.Path,
        val_shards_dir: pathlib.Path,
        runs_root: pathlib.Path,
    ) -> "BaselineRun":
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoint").mkdir()
        (run_dir / "links").mkdir()
        (run_dir / "inference").mkdir()

        cls._link(run_dir / "links" / "train-shards", train_shards_dir)
        cls._link(run_dir / "links" / "val-shards", val_shards_dir)
        return cls(run_dir)

    @classmethod
    def open(cls, run_dir: pathlib.Path) -> "BaselineRun":
        return cls(run_dir)

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return self.run_dir / "checkpoint"

    @property
    def inference_dir(self) -> pathlib.Path:
        return self.run_dir / "inference"

    @property
    def train_shards(self) -> pathlib.Path:
        return (self.run_dir / "links" / "train-shards").resolve()

    @property
    def val_shards(self) -> pathlib.Path:
        return (self.run_dir / "links" / "val-shards").resolve()

    @staticmethod
    def _link(dst: pathlib.Path, src: pathlib.Path) -> None:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def _sanitize_for_wandb(obj: tp.Any) -> tp.Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_wandb(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_wandb(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_wandb(v) for v in obj]
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return obj


def _deserialize_shuffled_config(dct: dict[str, object]) -> saev.data.ShuffledConfig:
    kwargs = dict(dct)
    shards = kwargs.get("shards")
    if isinstance(shards, str):
        kwargs["shards"] = pathlib.Path(shards)
    return saev.data.ShuffledConfig(**kwargs)


@jaxtyped(typechecker=beartype.beartype)
class MiniBatchKMeans(torch.nn.Module):
    """GPU mini-batch k-means estimator following the sklearn-style API."""

    def __init__(self, k: int, device: str = "cuda", collapse_tol: float = 0.5):
        super().__init__()
        self.k = k
        self.device = torch.device(device)
        self.cluster_centers_: Tensor | None = None
        self.cluster_counts_: Tensor | None = None
        self.n_steps_: int = 0
        self.n_features_in_: int | None = None
        self.last_batch_inertia_: float | None = None
        self.collapse_tol = collapse_tol

    def partial_fit(self, batch: Float[Tensor, "batch d_model"]) -> tp.Self:
        assert batch.ndim == 2, f"batch must be 2D, got {batch.shape}"
        batch_device = batch.to(self.device)

        if self.n_features_in_ is None:
            self.n_features_in_ = int(batch_device.shape[1])
        else:
            msg = (
                f"Expected {self.n_features_in_} features, got {batch_device.shape[1]}"
            )
            assert batch_device.shape[1] == self.n_features_in_, msg

        if self.cluster_centers_ is None:
            self._initialize_centers(batch_device)

        assert self.cluster_centers_ is not None
        distances = torch.cdist(batch_device, self.cluster_centers_)
        assignments = distances.argmin(dim=1)

        counts_batch = torch.bincount(assignments, minlength=self.k).to(
            device=batch_device.device, dtype=batch_device.dtype
        )
        sums_batch = torch.zeros_like(self.cluster_centers_)
        sums_batch.index_add_(0, assignments, batch_device)

        assert self.cluster_counts_ is not None
        prev_counts = self.cluster_counts_.clone()

        empty = (prev_counts == 0) & (counts_batch == 0)
        if empty.any():
            replacement_idx = torch.randint(
                0,
                batch_device.shape[0],
                (empty.sum().item(),),
                device=batch_device.device,
            )
            replacement = batch_device[replacement_idx]
            counts_batch = counts_batch.clone()
            counts_batch[empty] = 1.0
            sums_batch[empty] = replacement

        self.cluster_counts_ += counts_batch

        mask = counts_batch > 0
        if mask.any():
            self.cluster_centers_[mask] = (
                self.cluster_centers_[mask] * prev_counts[mask].unsqueeze(1)
                + sums_batch[mask]
            ) / self.cluster_counts_[mask].unsqueeze(1)

        arange = torch.arange(assignments.shape[0], device=batch_device.device)
        min_dist_sq = distances[arange, assignments].pow(2)
        self.last_batch_inertia_ = float(min_dist_sq.mean().item())

        self._split_collapsed_centers(batch_device)
        self.n_steps_ += 1
        return self

    def _initialize_centers(self, batch: Tensor) -> None:
        n_samples = batch.shape[0]
        if n_samples >= self.k:
            indices = torch.randperm(n_samples, device=batch.device)[: self.k]
            initial = batch[indices]
        else:
            repeats = -(-self.k // n_samples)
            initial = batch.repeat((repeats, 1))[: self.k]
        self.cluster_centers_ = initial.clone()
        self.cluster_counts_ = torch.zeros(
            self.k, device=batch.device, dtype=batch.dtype
        )

    def _split_collapsed_centers(self, batch: Tensor) -> None:
        if self.cluster_centers_ is None or self.cluster_counts_ is None or self.k < 2:
            return
        pairwise = torch.cdist(self.cluster_centers_, self.cluster_centers_)
        close_pairs = torch.triu(pairwise < self.collapse_tol, diagonal=1)
        if not close_pairs.any():
            return
        pairs = close_pairs.nonzero(as_tuple=False)
        cnt_i = self.cluster_counts_[pairs[:, 0]]
        cnt_j = self.cluster_counts_[pairs[:, 1]]
        losers = torch.where(cnt_i <= cnt_j, pairs[:, 0], pairs[:, 1])
        loser_mask = torch.zeros(self.k, dtype=torch.bool, device=self.device)
        loser_mask[losers] = True
        if not loser_mask.any():
            return
        n_needed = int(loser_mask.sum().item())
        candidate_points = batch
        if batch.shape[0] < n_needed:
            repeats = -(-n_needed // batch.shape[0])
            candidate_points = batch.repeat((repeats, 1))
        candidate_dist = torch.cdist(candidate_points, self.cluster_centers_)
        farthest_scores = candidate_dist.max(dim=1).values
        indices = torch.argsort(farthest_scores, descending=True)[:n_needed]
        replacement = candidate_points[indices]
        self.cluster_centers_[loser_mask] = replacement
        self.cluster_counts_[loser_mask] = torch.zeros_like(
            self.cluster_counts_[loser_mask]
        )

    def transform(
        self, batch: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch k"]:
        assert self.cluster_centers_ is not None, "MiniBatchKMeans has not been fitted"
        distances = torch.cdist(batch.to(self.device), self.cluster_centers_)
        return -distances


@jaxtyped(typechecker=beartype.beartype)
class MiniBatchPCA(torch.nn.Module):
    """Placeholder PCA estimator (implementation pending)."""

    def __init__(self, n_components: int, device: str = "cuda"):
        super().__init__()
        self.n_components = n_components
        self.device = torch.device(device)
        self.components_: Tensor | None = None
        self.n_steps_: int = 0

    def partial_fit(self, batch: Float[Tensor, "batch d_model"]) -> tp.Self:
        raise NotImplementedError("MiniBatchPCA.partial_fit is not implemented yet")

    def transform(
        self, batch: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch c"]:
        assert self.components_ is not None, "Call partial_fit first"
        return batch.to(self.device) @ self.components_.T


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    method: BaselineMethod = "kmeans"
    train_data: saev.data.ShuffledConfig = saev.data.ShuffledConfig()
    val_data: saev.data.ShuffledConfig = saev.data.ShuffledConfig()
    n_train: int = 100_000_000
    n_val: int = 10_000_000
    k: int = 1024 * 16
    collapse_tol: float = 0.5
    device: tp.Literal["cuda", "cpu"] = "cuda"
    seed: int = 42
    runs_root: pathlib.Path = pathlib.Path("./tdiscovery/runs")
    slurm_acct: str = ""
    slurm_partition: str = ""
    n_hours: float = 24.0
    mem_gb: int = 128
    log_to: pathlib.Path = pathlib.Path("./logs")
    debug: bool = False
    track: bool = True
    wandb_project: str = "tdiscovery"
    tag: str = ""
    log_every: int = 50


@dataclasses.dataclass(frozen=True)
class InferenceConfig:
    run: pathlib.Path = pathlib.Path("./tdiscovery/runs/example")
    split: tp.Literal["train", "val"] = "train"
    batch_size: int | None = None
    device: tp.Literal["cuda", "cpu"] = "cuda"
    force: bool = False
    slurm_acct: str = ""
    slurm_partition: str = ""
    n_hours: float = 4.0
    mem_gb: int = 80
    log_to: pathlib.Path = pathlib.Path("./logs")


@beartype.beartype
def get_training_metrics(model: torch.nn.Module, n_samples: int) -> dict[str, float]:
    if isinstance(model, MiniBatchKMeans):
        if model.last_batch_inertia_ is None:
            return {}
        return {
            "train/inertia": model.last_batch_inertia_,
            "train/l0": 1.0,
            "train/n_samples": float(n_samples),
        }
    else:
        tp.assert_never(model)


@beartype.beartype
def eval_kmeans(cfg: TrainConfig, model: MiniBatchKMeans) -> dict[str, float]:
    if cfg.n_val <= 0:
        return {}

    dataloader = saev.data.ShuffledDataLoader(cfg.val_data)
    loader_like = tp.cast(saev.utils.scheduling.DataLoaderLike, dataloader)
    limit = min(cfg.n_val, dataloader.n_samples)
    limiter = saev.utils.scheduling.BatchLimiter(loader_like, limit)

    assert model.cluster_centers_ is not None
    centers = model.cluster_centers_

    hits = torch.zeros(centers.shape[0], device="cpu")
    total_samples = 0
    sum_sq_distance = 0.0

    for batch in saev.helpers.progress(limiter):
        acts = batch["act"].to(model.device, non_blocking=True)
        distances = torch.cdist(acts, centers)
        min_dist, assignments = distances.min(dim=1)
        sum_sq_distance += float((min_dist**2).sum().item())
        hits.index_add_(
            0,
            assignments.cpu(),
            torch.ones_like(assignments, dtype=torch.float32).cpu(),
        )
        total_samples += acts.shape[0]

    if total_samples == 0:
        return {}

    utilization = (hits > 0).float().mean().item()
    max_pop = hits.max().item()
    mean_pop = hits.mean().item()
    inertia = sum_sq_distance / total_samples

    return {
        "eval/inertia": inertia,
        "eval/utilization": utilization,
        "eval/mean_pop": mean_pop,
        "eval/max_pop": max_pop,
    }


@beartype.beartype
def train_worker_fn(cfg: TrainConfig) -> str:
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, force=True)
    logger = logging.getLogger("train_worker_fn")
    logger.info("Started baseline job (method=%s)", cfg.method)

    torch.manual_seed(cfg.seed)
    cfg.log_to.mkdir(parents=True, exist_ok=True)
    dl = saev.data.ShuffledDataLoader(cfg.train_data)
    dl = saev.utils.scheduling.BatchLimiter(dl, cfg.n_train)

    if cfg.method == "kmeans":
        model = MiniBatchKMeans(k=cfg.k, device=cfg.device)
    elif cfg.method == "pca":
        model = MiniBatchPCA(n_components=cfg.k, device=cfg.device)
    else:
        tp.assert_never(cfg.method)

    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict["train_data"]["metadata"] = dataclasses.asdict(dl.metadata)
    tags = [cfg.tag] if cfg.tag else []
    mode = "online" if cfg.track else "disabled"
    run = wandb.init(project=cfg.wandb_project, config=cfg_dict, mode=mode, tags=tags)

    n_samples = 0
    t_start = time.perf_counter()
    for batch in dl:
        acts = batch["act"]
        msg = f"Expected 2D activations, got shape {tuple(acts.shape)}"
        assert acts.ndim == 2, msg
        acts = acts.to(model.device) if hasattr(model, "device") else acts
        model.partial_fit(acts)
        n_samples += acts.shape[0]

        if model.n_steps_ % cfg.log_every == 0:
            metrics = get_training_metrics(model, n_samples)
            run.log(metrics, step=model.n_steps_)
            logger.info(", ".join(f"{key}={value}" for key, value in metrics.items()))

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Training complete: method=%s steps=%d samples=%d elapsed=%.2fs",
        cfg.method,
        getattr(model, "n_steps_", 0),
        n_samples,
        elapsed,
    )

    if cfg.method == "kmeans":
        assert isinstance(model, MiniBatchKMeans)
        assert model.cluster_centers_ is not None
        assert model.cluster_counts_ is not None
        assert model.n_features_in_ is not None
        eval_metrics = eval_kmeans(cfg, model)
    else:
        tp.assert_never(cfg.method)

    for key, value in eval_metrics.items():
        logger.info("%s=%.6f", key, value)
    if run and eval_metrics:
        run.log(eval_metrics, step=getattr(model, "n_steps_", 0))

    baseline_run = BaselineRun.new(
        run.id,
        train_shards_dir=cfg.train_data.shards.resolve(),
        val_shards_dir=cfg.val_data.shards.resolve(),
        runs_root=cfg.runs_root,
    )

    config_json = _sanitize_for_wandb(dataclasses.asdict(cfg))
    with open(baseline_run.checkpoint_dir / "config.json", "wb") as fd:
        saev.helpers.jdump(config_json, fd, option=orjson.OPT_INDENT_2)

    ckpt: dict[str, tp.Any] = {
        "method": cfg.method,
        "config": config_json,
        "metrics": eval_metrics,
    }
    if cfg.method == "kmeans":
        assert isinstance(model, MiniBatchKMeans)
        ckpt.update({
            "cluster_centers": model.cluster_centers_.cpu(),
            "cluster_counts": model.cluster_counts_.cpu(),
            "n_steps": model.n_steps_,
            "n_features_in": model.n_features_in_,
        })
    elif cfg.method == "pca":
        assert isinstance(model, MiniBatchPCA)
        ckpt.update({
            "components": None
            if model.components_ is None
            else model.components_.cpu(),
            "n_steps": model.n_steps_,
        })

    ckpt_path = baseline_run.checkpoint_dir / f"{cfg.method}.pt"
    torch.save(ckpt, ckpt_path)
    logger.info("Saved checkpoint to %s", ckpt_path)
    run.finish()
    return run.id


@beartype.beartype
def inference_worker_fn(cfg: InferenceConfig):
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("tdiscovery.baselines.inference")

    run = BaselineRun.open(cfg.run)
    config_path = run.checkpoint_dir / "config.json"
    with open(config_path, "rb") as fd:
        train_cfg_dict = orjson.loads(fd.read())

    method = train_cfg_dict["method"]
    if method != "kmeans":
        raise NotImplementedError("Only kmeans inference is implemented")

    ckpt_path = run.checkpoint_dir / f"{method}.pt"
    ckpt = torch.load(ckpt_path, map_location=cfg.device)

    model = MiniBatchKMeans(
        k=ckpt["cluster_centers"].shape[0],
        device=cfg.device,
        collapse_tol=float(train_cfg_dict.get("collapse_tol", 0.5)),
    )
    model.cluster_centers_ = ckpt["cluster_centers"].to(cfg.device)
    model.cluster_counts_ = ckpt["cluster_counts"].to(cfg.device)
    model.n_features_in_ = int(
        ckpt.get("n_features_in", model.cluster_centers_.shape[1])
    )

    split_key = "train_data" if cfg.split == "train" else "val_data"
    split_cfg = _deserialize_shuffled_config(train_cfg_dict[split_key])
    shards_dir = run.train_shards if cfg.split == "train" else run.val_shards
    ordered_cfg = saev.data.make_ordered_config(
        split_cfg,
        batch_size=cfg.batch_size or split_cfg.batch_size,
        shards=shards_dir,
    )

    dataloader = saev.data.OrderedDataLoader(ordered_cfg)
    total_samples = dataloader.n_samples
    logger.info(
        "Running inference for %s split with %d samples", cfg.split, total_samples
    )

    md = saev.data.Metadata.load(ordered_cfg.shards)
    root = run.inference_dir / md.hash
    root.mkdir(parents=True, exist_ok=True)
    token_acts_path = root / "token_acts.npz"
    if token_acts_path.exists() and not cfg.force:
        logger.info("token_acts.npz already exists at %s; skipping", token_acts_path)
        return 0

    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    offset = 0

    for batch in saev.helpers.progress(dataloader, every=20, desc="kmeans-inference"):
        acts = batch["act"].to(model.device)
        distances = torch.cdist(acts, model.cluster_centers_)
        min_dist, assignments = distances.min(dim=1)
        scores = 1.0 / (1.0 + min_dist)

        batch_rows = torch.arange(offset, offset + acts.shape[0])
        rows.append(batch_rows)
        cols.append(assignments.cpu())
        vals.append(scores.cpu())
        offset += acts.shape[0]

    if offset != total_samples:
        logger.warning(
            "Expected %d samples but processed %d; recomputing shape",
            total_samples,
            offset,
        )
        total_samples = offset

    row_idx = torch.cat(rows).numpy()
    col_idx = torch.cat(cols).numpy()
    data = torch.cat(vals).numpy()
    token_acts = scipy.sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(total_samples, model.k)
    )
    scipy.sparse.save_npz(token_acts_path, token_acts)
    with open(root / "config.json", "wb") as fd:
        saev.helpers.jdump(
            {"split": cfg.split, "ordered_config": dataclasses.asdict(ordered_cfg)},
            fd,
            option=orjson.OPT_INDENT_2,
        )
    logger.info("Wrote %s", token_acts_path)
    return 0


@beartype.beartype
def train_cli(
    cfg: tp.Annotated[TrainConfig, tyro.conf.arg(name="")],
    sweep: pathlib.Path | None = None,
):
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("tdiscovery.baselines.train")

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
        logger.error("No configs resolved for train sweep.")
        return 1

    base = cfgs[0]
    if any(c.slurm_acct != base.slurm_acct for c in cfgs):
        logger.error("All configs must share the same slurm_acct.")
        return 1
    if any(c.slurm_partition != base.slurm_partition for c in cfgs):
        logger.error("All configs must share the same slurm_partition.")
        return 1
    if any(c.log_to != base.log_to for c in cfgs):
        logger.error("All configs must share the same log directory.")
        return 1

    base.log_to.mkdir(parents=True, exist_ok=True)
    logger.info("Prepared %d config(s).", len(cfgs))

    if not base.slurm_acct:
        for idx, cfg_i in enumerate(cfgs, start=1):
            logger.info("Running config %d/%d locally.", idx, len(cfgs))
            train_worker_fn(cfg_i)
        logger.info("Jobs done.")
        return 0

    executor = submitit.SlurmExecutor(folder=str(base.log_to))

    n_cpus = max(cfg.train_data.n_threads, cfg.val_data.n_threads) + 4
    if cfg.mem_gb // 10 > n_cpus:
        logger.info(
            "Using %d CPUs instead of %d to get more RAM.", cfg.mem_gb // 10, n_cpus
        )
        n_cpus = cfg.mem_gb // 10

    executor.update_parameters(
        time=int(base.n_hours * 60),
        partition=base.slurm_partition,
        gpus_per_node=1,
        ntasks_per_node=1,
        cpus_per_task=n_cpus,
        stderr_to_stdout=True,
        account=base.slurm_acct,
    )

    try:
        cloudpickle.dumps(train_worker_fn)
        for c in cfgs:
            cloudpickle.dumps(c)
    except TypeError as err:
        logger.error("Failed to pickle job payload: %s", err)
        return 1

    with executor.batch():
        jobs = [executor.submit(train_worker_fn, c) for c in cfgs]

    time.sleep(5.0)
    for idx, job in enumerate(jobs, start=1):
        logger.info("Job %d/%d: %s %s", idx, len(jobs), job.job_id, job.state)

    for idx, job in enumerate(jobs, start=1):
        try:
            job.result()
            logger.info("Job %d/%d finished.", idx, len(jobs))
        except UncompletedJobError:
            logger.warning("Job %s (%d) did not finish.", job.job_id, idx)

    logger.info("Jobs done.")
    return 0


@beartype.beartype
def inference_cli(
    cfg: tp.Annotated[InferenceConfig, tyro.conf.arg(name="")],
):
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("tdiscovery.baselines.inference.cli")

    cfg.log_to.mkdir(parents=True, exist_ok=True)
    if not cfg.slurm_acct:
        return inference_worker_fn(cfg)

    executor = submitit.SlurmExecutor(folder=str(cfg.log_to))
    n_cpus = max(1, cfg.mem_gb // 10)
    gpus = 1 if cfg.device == "cuda" else 0
    executor.update_parameters(
        time=int(cfg.n_hours * 60),
        partition=cfg.slurm_partition,
        gpus_per_node=gpus,
        ntasks_per_node=1,
        cpus_per_task=n_cpus,
        mem_gb=cfg.mem_gb,
        stderr_to_stdout=True,
        account=cfg.slurm_acct,
    )

    try:
        cloudpickle.dumps(inference_worker_fn)
        cloudpickle.dumps(cfg)
    except TypeError as err:
        logger.error("Failed to pickle job payload: %s", err)
        return 1

    job = executor.submit(inference_worker_fn, cfg)
    logger.info("Submitted job %s", job.job_id)
    try:
        return job.result()
    except UncompletedJobError:
        logger.warning("Inference job %s did not finish", job.job_id)
        return 1
