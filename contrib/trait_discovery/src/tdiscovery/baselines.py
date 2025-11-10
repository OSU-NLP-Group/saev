import dataclasses
import logging
import pathlib
import time
import typing as tp

import beartype
import cloudpickle
import torch
import tyro
import wandb
from jaxtyping import Float, jaxtyped
from torch import Tensor

import saev.configs
import saev.data
import saev.utils.scheduling

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

BaselineMethod = tp.Literal["kmeans", "pca"]


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
class Config:
    method: BaselineMethod = "kmeans"
    train_data: saev.data.ShuffledConfig = saev.data.ShuffledConfig()
    val_data: saev.data.ShuffledConfig = saev.data.ShuffledConfig()
    n_train: int = 100_000_000
    n_val: int = 10_000_000
    n_clusters: int = 1024 * 16
    n_components: int = 1024 * 16
    collapse_tol: float = 0.5
    device: tp.Literal["cuda", "cpu"] = "cuda"
    seed: int = 42
    slurm_acct: str = ""
    slurm_partition: str = ""
    n_hours: float = 24.0
    mem_gb: int = 128
    log_to: pathlib.Path = pathlib.Path("./logs")
    dump_to: pathlib.Path = pathlib.Path("./results")
    debug: bool = False
    track: bool = True
    wandb_project: str = "tdiscovery"
    tag: str = ""
    log_every: int = 50


@beartype.beartype
def _training_metrics(
    method: BaselineMethod, model: torch.nn.Module, n_samples: int
) -> dict[str, float]:
    if method == "kmeans":
        assert isinstance(model, MiniBatchKMeans)
        if model.last_batch_inertia_ is None:
            return {}
        return {
            "train/inertia": model.last_batch_inertia_,
            "train/l0": 1.0,
            "train/n_samples": float(n_samples),
        }
    else:
        tp.assert_never(method)


@beartype.beartype
def _evaluate_kmeans(cfg: Config, model: MiniBatchKMeans) -> dict[str, float]:
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

    for batch in limiter:
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
def worker_fn(cfg: Config):
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, force=True)
    job_logger = logging.getLogger("tdiscovery.baselines.dense")
    job_logger.info("Started dense baseline job (method=%s)", cfg.method)

    torch.manual_seed(cfg.seed)
    dataloader = saev.data.ShuffledDataLoader(cfg.train_data)
    loader_like = tp.cast(saev.utils.scheduling.DataLoaderLike, dataloader)
    limited_loader = saev.utils.scheduling.BatchLimiter(loader_like, cfg.n_train)

    if cfg.method == "kmeans":
        model: torch.nn.Module = MiniBatchKMeans(
            k=cfg.n_clusters, device=cfg.device, collapse_tol=cfg.collapse_tol
        )
    elif cfg.method == "pca":
        model = MiniBatchPCA(n_components=cfg.n_components, device=cfg.device)
    else:  # pragma: no cover - config guaranteed by Literal
        raise ValueError(f"Unknown method: {cfg.method}")

    mode = "online" if cfg.track else "disabled"
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict["train_data"]["metadata"] = dataclasses.asdict(dataloader.metadata)
    tags = [cfg.tag] if cfg.tag else []
    run = wandb.init(project=cfg.wandb_project, config=cfg_dict, mode=mode, tags=tags)

    n_samples = 0
    t_start = time.perf_counter()
    for batch in limited_loader:
        acts = batch["act"]
        msg = f"Expected 2D activations, got shape {tuple(acts.shape)}"
        assert acts.ndim == 2, msg
        acts = acts.to(model.device) if hasattr(model, "device") else acts
        model.partial_fit(acts)
        n_samples += acts.shape[0]

        if run and model.n_steps_ % cfg.log_every == 0:
            metrics = _training_metrics(cfg.method, model, n_samples)
            if metrics:
                run.log(metrics, step=model.n_steps_)

    elapsed = time.perf_counter() - t_start
    job_logger.info(
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
        eval_metrics = _evaluate_kmeans(cfg, model)
    else:
        eval_metrics = {}

    for key, value in eval_metrics.items():
        job_logger.info("%s=%.6f", key, value)
    if run and eval_metrics:
        run.log(eval_metrics, step=getattr(model, "n_steps_", 0))

    cfg.dump_to.mkdir(parents=True, exist_ok=True)
    ckpt: dict[str, tp.Any] = {
        "method": cfg.method,
        "config": dataclasses.asdict(cfg),
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

    ckpt_fname = f"{cfg.method}_train{cfg.n_train}_seed{cfg.seed}.pt"
    ckpt_path = cfg.dump_to / ckpt_fname
    torch.save(ckpt, ckpt_path)
    job_logger.info("Saved checkpoint to %s", ckpt_path)
    if run:
        run.finish()
    return 0


@beartype.beartype
def cli(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")], sweep: pathlib.Path | None = None
) -> int:
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("tdiscovery.baselines.cli")

    if sweep is None:
        cfgs = [cfg]
    else:
        sweep_dcts = saev.configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            return 1
        cfgs, errs = saev.configs.load_cfgs(
            cfg, default=Config(), sweep_dcts=sweep_dcts
        )
        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return 1

    if not cfgs:
        logger.error("No configs resolved for dense baseline sweep.")
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
        n_errs = 0
        for idx, cfg_i in enumerate(cfgs, start=1):
            try:
                logger.info("Running config %d/%d locally.", idx, len(cfgs))
                worker_fn(cfg_i)
            except Exception as err:  # pragma: no cover - defensive logging
                logger.exception("Job %d/%d failed: %s", idx, len(cfgs), err)
                n_errs += 1
        logger.info("Jobs done. %d error(s)", n_errs)
        return 0

    import submitit
    from submitit.core.utils import UncompletedJobError

    executor = submitit.SlurmExecutor(folder=str(base.log_to))
    n_cpus = max(1, base.mem_gb // 10)
    gpus = 1 if base.device == "cuda" else 0
    executor.update_parameters(
        time=int(base.n_hours * 60),
        partition=base.slurm_partition,
        gpus_per_node=gpus,
        ntasks_per_node=1,
        cpus_per_task=n_cpus,
        mem_gb=base.mem_gb,
        stderr_to_stdout=True,
        account=base.slurm_acct,
    )

    try:
        cloudpickle.dumps(worker_fn)
        for c in cfgs:
            cloudpickle.dumps(c)
    except TypeError as err:  # pragma: no cover - defensive
        logger.error("Failed to pickle job payload: %s", err)
        return 1

    with executor.batch():
        jobs = [executor.submit(worker_fn, c) for c in cfgs]

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
