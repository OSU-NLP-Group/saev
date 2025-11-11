import collections.abc
import dataclasses
import io
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

import saev
import saev.configs
import saev.data
import saev.helpers
import saev.utils.scheduling
from saev.framework.inference import Filepaths

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

BaselineMethod = tp.Literal["kmeans", "pca"]

BASELINE_SCHEMA_VERSION = 1


@jaxtyped(typechecker=beartype.beartype)
class MiniBatchKMeans(torch.nn.Module):
    """GPU mini-batch k-means estimator following the sklearn-style API."""

    method = "kmeans"

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
    """Streaming PCA estimator that maintains the covariance statistics online."""

    method = "pca"

    def __init__(self, n_components: int, device: str = "cuda"):
        super().__init__()
        assert n_components > 0, "n_components must be positive"
        self.n_components = n_components
        self.device = torch.device(device)
        self.components_: Tensor | None = None
        self.explained_variance_: Tensor | None = None
        self.n_steps_: int = 0
        self.n_features_in_: int | None = None
        self.n_samples_seen_: int = 0
        self.mean_: Tensor | None = None
        self.scatter_: Tensor | None = None

    def partial_fit(self, batch: Float[Tensor, "batch d_model"]) -> tp.Self:
        assert batch.ndim == 2, f"batch must be 2D, got {batch.shape}"
        if batch.shape[0] == 0:
            return self

        batch_device = batch.to(self.device)
        n_batch, n_features = batch_device.shape
        if self.n_features_in_ is None:
            self.n_features_in_ = n_features
        else:
            msg = f"Expected {self.n_features_in_} features, got {n_features}"
            assert n_features == self.n_features_in_, msg

        assert self.n_components <= n_features, (
            f"n_components={self.n_components} > n_features={n_features}"
        )

        batch_mean = batch_device.mean(dim=0)
        centered = batch_device - batch_mean
        scatter_update = centered.T @ centered

        n_prev = self.n_samples_seen_
        prev_mean = self.mean_
        prev_scatter = self.scatter_
        if n_prev == 0 or prev_mean is None or prev_scatter is None:
            self.mean_ = batch_mean
            self.scatter_ = scatter_update
            self.n_samples_seen_ = n_batch
        else:
            n_total = n_prev + n_batch
            delta = batch_mean - prev_mean
            correction = torch.outer(delta, delta) * (n_prev * n_batch / n_total)
            self.scatter_ = prev_scatter + scatter_update + correction
            self.mean_ = prev_mean + delta * (n_batch / n_total)
            self.n_samples_seen_ = n_total

        scatter = self.scatter_
        mean = self.mean_
        assert scatter is not None
        assert mean is not None

        denom = max(self.n_samples_seen_ - 1, 1)
        covariance = scatter / denom
        covariance = 0.5 * (covariance + covariance.mT)

        eigvals, eigvecs = torch.linalg.eigh(covariance)
        order = torch.argsort(eigvals, descending=True)
        top = order[: self.n_components]
        self.explained_variance_ = eigvals[top]
        self.components_ = eigvecs[:, top].mT.contiguous()

        self.n_steps_ += 1
        return self

    def transform(
        self, batch: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "batch c"]:
        assert self.components_ is not None, "Call partial_fit first"
        assert self.mean_ is not None, "Call partial_fit first"
        centered = batch.to(self.device) - self.mean_
        return centered @ self.components_.T


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
    data: saev.data.OrderedConfig = saev.data.OrderedConfig()
    """Data configuration"""
    device: tp.Literal["cuda", "cpu"] = "cuda"
    n_dists: int = 25
    """Number of latent dimensions to snapshot for distributions."""
    force: bool = False
    slurm_acct: str = ""
    slurm_partition: str = ""
    n_hours: float = 4.0
    mem_gb: int = 80
    log_to: pathlib.Path = pathlib.Path("./logs")


@beartype.beartype
def _materialize_model(
    method: BaselineMethod, state: dict[str, tp.Any], *, device: str
) -> torch.nn.Module:
    if method == "kmeans":
        centers = state["cluster_centers"]
        counts = state["cluster_counts"]
        collapse_tol = state["collapse_tol"]

        k, n_features_in = centers.shape

        model = MiniBatchKMeans(k=k, device=device, collapse_tol=collapse_tol)
        model.cluster_centers_ = centers.to(device)
        model.cluster_counts_ = counts.to(device)
        model.n_steps_ = state["n_steps"]
        model.n_features_in_ = n_features_in
        return model

    if method == "pca":
        raise NotImplementedError("MiniBatchPCA deserialization not implemented")
    tp.assert_never(method)


@beartype.beartype
def load(run: saev.disk.Run, *, device: str = "cpu") -> torch.nn.Module:
    ckpt_path = run.ckpt_dir / "baseline.pt"
    with open(ckpt_path, "rb") as fd:
        header = orjson.loads(fd.readline())
        buffer = io.BytesIO(fd.read())

    state = torch.load(buffer, map_location=device, weights_only=False)
    assert isinstance(state, dict), f"Unexpected checkpoint payload in {ckpt_path}"

    model = _materialize_model(header["method"], state, device=device)

    return model


@beartype.beartype
def _serialize_model_state(model: torch.nn.Module) -> dict[str, tp.Any]:
    if isinstance(model, MiniBatchKMeans):
        assert model.cluster_centers_ is not None, "cluster_centers_ missing"
        assert model.cluster_counts_ is not None, "cluster_counts_ missing"
        assert model.n_features_in_ is not None, "n_features_in_ missing"
        return {
            "cluster_centers": model.cluster_centers_.cpu(),
            "cluster_counts": model.cluster_counts_.cpu(),
            "n_steps": model.n_steps_,
            "n_features_in": model.n_features_in_,
            "collapse_tol": float(model.collapse_tol),
        }
    if isinstance(model, MiniBatchPCA):
        raise NotImplementedError("MiniBatchPCA serialization not implemented")
    raise TypeError(f"Unsupported baseline model type: {type(model).__name__}")


@beartype.beartype
def dump(dpath: pathlib.Path, cfg: TrainConfig, model: torch.nn.Module) -> pathlib.Path:
    dpath.mkdir(parents=True, exist_ok=True)
    header: dict[str, tp.Any] = {
        "method": model.method,
        "schema": BASELINE_SCHEMA_VERSION,
        "commit": saev.helpers.current_git_commit() or "unknown",
        "lib": saev.__version__,
    }

    cfg_dict = dataclasses.asdict(cfg)
    config_path = dpath / "config.json"
    with open(config_path, "wb") as fd:
        saev.helpers.jdump(cfg_dict, fd, option=orjson.OPT_INDENT_2)

    fpath = dpath / "baseline.pt"
    with open(fpath, "wb") as fd:
        saev.helpers.jdump(header, fd, option=orjson.OPT_APPEND_NEWLINE)
        torch.save(_serialize_model_state(model), fd)
    return fpath


@beartype.beartype
def get_training_metrics(model: torch.nn.Module, n_samples: int) -> dict[str, float]:
    if isinstance(model, MiniBatchKMeans):
        return {
            "train/inertia": model.last_batch_inertia_ or 0.0,
            "train/l0": 1.0,
            "train/n_samples": n_samples,
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

    for batch in saev.helpers.progress(limiter, desc="eval"):
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
def train_worker_fn(cfg: TrainConfig):
    level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, force=True)
    logger = logging.getLogger("train_worker_fn")
    logger.info("Started baseline job (method=%s)", cfg.method)

    torch.manual_seed(cfg.seed)
    cfg.log_to.mkdir(parents=True, exist_ok=True)
    dl = saev.data.ShuffledDataLoader(cfg.train_data)
    dl = saev.utils.scheduling.BatchLimiter(dl, cfg.n_train)
    train_metadata = getattr(dl, "metadata", None)

    if cfg.method == "kmeans":
        model = MiniBatchKMeans(k=cfg.k, device=cfg.device)
    elif cfg.method == "pca":
        model = MiniBatchPCA(n_components=cfg.k, device=cfg.device)
    else:
        tp.assert_never(cfg.method)

    cfg_dict = dataclasses.asdict(cfg)
    if train_metadata is not None:
        cfg_dict["train_data"]["metadata"] = dataclasses.asdict(train_metadata)
    tags = [cfg.tag] if cfg.tag else []
    mode = "online" if cfg.track else "disabled"
    run = wandb.init(project=cfg.wandb_project, config=cfg_dict, mode=mode, tags=tags)

    n_samples = 0
    t_start = time.perf_counter()
    for batch in saev.helpers.progress(dl, every=cfg.log_every, desc="train"):
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

    run.finish()
    run_id = getattr(run, "id", "offline")
    # Baseline Run
    run = saev.disk.Run.new(
        run_id,
        train_shards_dir=cfg.train_data.shards.resolve(),
        val_shards_dir=cfg.val_data.shards.resolve(),
        runs_root=cfg.runs_root,
    )
    ckpt_path = dump(run.ckpt_dir, cfg, model)
    logger.info("Saved checkpoint to %s", ckpt_path)


@beartype.beartype
@torch.inference_mode()
def inference_worker_fn(cfg: InferenceConfig):
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("inference")

    run = saev.disk.Run(cfg.run)
    model = load(run, device=cfg.device)
    assert isinstance(model, MiniBatchKMeans), "Only kmeans inference supported."
    assert model.cluster_centers_ is not None
    k = model.k
    d_model = int(model.cluster_centers_.shape[1])

    md = saev.data.Metadata.load(cfg.data.shards)
    batch_size = (
        cfg.data.batch_size
        // md.content_tokens_per_example
        * md.content_tokens_per_example
    )
    assert batch_size > 0, "Batch size must be at least one example worth of tokens."
    data_cfg = dataclasses.replace(cfg.data, batch_size=batch_size)
    dataloader = saev.data.OrderedDataLoader(data_cfg)
    logger.info("Running inference on %d samples", dataloader.n_samples)

    fpaths = Filepaths.from_run(run, md)
    missing = [fpath for fpath in fpaths if not fpath.exists()]
    if not cfg.force and not missing:
        logger.info("Found all baseline inference artifacts; skipping.")
        return

    if cfg.force:
        logger.info("Force flag set; recomputing baseline inference.")
    else:
        missing_msg = ", ".join(str(fpath) for fpath in missing)
        logger.info("Missing files %s; recomputing baseline inference.", missing_msg)

    root = fpaths.mean_values.parent
    with open(root / "config.json", "wb") as fd:
        saev.helpers.jdump(dataclasses.asdict(cfg), fd, option=orjson.OPT_INDENT_2)

    device = model.cluster_centers_.device
    mean_values_c = torch.zeros((k,), device=device)
    sparsity_c = torch.zeros((k,), device=device)
    distributions_nm = torch.zeros(
        (dataloader.n_samples, cfg.n_dists), dtype=torch.float32
    )

    rows: list[Tensor] = []
    cols: list[Tensor] = []
    vals: list[Tensor] = []
    prev_i = -1

    reconstruction_sse = torch.zeros((), dtype=torch.float64, device=device)
    sum_sq = torch.zeros((), dtype=torch.float64, device=device)
    sum_vec_d = torch.zeros((d_model,), dtype=torch.float64, device=device)
    n_tokens = 0

    dataloader_iter = tp.cast(
        collections.abc.Iterable[dict[str, torch.Tensor]], dataloader
    )

    for batch in saev.helpers.progress(dataloader_iter, desc="inference", every=1):
        acts = batch["act"].to(device)
        distances = torch.cdist(acts, model.cluster_centers_)
        min_dist, assignments = distances.min(dim=1)
        scores = 1.0 / (1.0 + min_dist)

        assignments_cpu = assignments.to("cpu")
        scores_cpu = scores.to("cpu")

        counts = torch.bincount(assignments_cpu, minlength=k).to(
            device=sparsity_c.device, dtype=sparsity_c.dtype
        )
        sparsity_c += counts
        mean_values_c.index_add_(0, assignments, scores)

        batch_idx = (
            batch["example_idx"] * md.content_tokens_per_example + batch["token_idx"]
        )
        assert batch_idx[0].item() == prev_i + 1
        assert (torch.sort(batch_idx).values == batch_idx).all()
        assert (torch.arange(batch_idx[0], batch_idx[-1] + 1) == batch_idx).all()
        prev_i = batch_idx[-1].item()

        rows.append(batch_idx)
        cols.append(assignments_cpu)
        vals.append(scores_cpu)

        if cfg.n_dists > 0:
            limited = assignments_cpu < cfg.n_dists
            if limited.any():
                row_sel = batch["example_idx"][limited]
                col_sel = assignments_cpu[limited]
                distributions_nm[row_sel, col_sel] = scores_cpu[limited]

        min_dist_sq = min_dist.pow(2).to(torch.float64)
        reconstruction_sse += min_dist_sq.sum()
        acts64 = acts.to(torch.float64)
        sum_sq += (acts64 * acts64).sum()
        sum_vec_d += acts64.sum(dim=0)
        n_tokens += batch_idx.shape[0]

    assert n_tokens == dataloader.n_samples
    assert n_tokens > 0

    mean_values_c /= sparsity_c
    sparsity_c /= dataloader.n_samples

    row_idx = torch.cat(rows).numpy()
    col_idx = torch.cat(cols).numpy()
    data = torch.cat(vals).numpy()
    token_acts = scipy.sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(dataloader.n_samples, k)
    )
    scipy.sparse.save_npz(fpaths.token_acts, token_acts)

    torch.save(mean_values_c.cpu(), fpaths.mean_values)
    torch.save(sparsity_c.cpu(), fpaths.sparsity)
    torch.save(distributions_nm.cpu(), fpaths.distributions)

    sse_sae = reconstruction_sse.item()
    sum_sq_item = sum_sq.item()
    sum_vec_sq = torch.dot(sum_vec_d, sum_vec_d).item()
    sse_baseline = sum_sq_item - sum_vec_sq / n_tokens
    msg = (
        f"Baseline variance is non-positive (sse_baseline={sse_baseline:.6e});"
        " cannot compute normalized MSE."
    )
    assert sse_baseline > 0.0, msg
    metrics = {
        "normalized_mse": sse_sae / sse_baseline,
        "sse_sae": sse_sae,
        "sse_baseline": sse_baseline,
    }

    with open(fpaths.metrics, "wb") as fd:
        saev.helpers.jdump(metrics, fd, option=orjson.OPT_INDENT_2)

    logger.info(
        "Wrote baseline inference artifacts: %s, %s, %s, %s, %s",
        fpaths.token_acts,
        fpaths.mean_values,
        fpaths.sparsity,
        fpaths.distributions,
        fpaths.metrics,
    )


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
    sweep: pathlib.Path | None = None,
):
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("inference_cli")

    if sweep is None:
        cfgs = [cfg]
    else:
        sweep_dcts = saev.configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            return 1
        cfgs, errs = saev.configs.load_cfgs(
            cfg, default=InferenceConfig(), sweep_dcts=sweep_dcts
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
            inference_worker_fn(cfg_i)
        logger.info("Jobs done.")
        return 0

    executor = submitit.SlurmExecutor(folder=str(cfg.log_to))
    n_cpus = 4
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
        cloudpickle.dumps(inference_worker_fn)
        for c in cfgs:
            cloudpickle.dumps(c)
    except TypeError as err:
        logger.error("Failed to pickle job payload: %s", err)
        return 1

    with executor.batch():
        jobs = [executor.submit(inference_worker_fn, c) for c in cfgs]

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
