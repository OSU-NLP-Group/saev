# train.py
"""
Trains many SAEs in parallel to amortize the cost of loading a single batch of data over many SAE training runs.

Checklist for making sure your training doesn't suck:

* [ ] Data scaling: scale vectors so their average L2 norm is sqrt(n).
* [ ] Initialize W_e to 1/exp_factor * W_d.T
* [ ] Initialize b_e such that each feature activates 10K/(n * exp_factor) of the time, which means that on average, each example activates 10K features.
* [ ] Initialize W_d to ~uniform(-1/sqrt(n), 1/sqrt(n))
* [x] Initialize b_d to 0.
* [x] Sweep learning rate and sparsity coefficients.
* [ ] Decay learning rate to 0 over the last 20% of training.
* [ ] Warmup sparsity over all of training.
* [ ] Gradient clipping (clip at 1 with clip_grad_norm)
"""

import dataclasses
import logging
import os.path
import pathlib
import sys
import time
import typing as tp

import beartype
import einops
import orjson
import torch
import tyro
import wandb
from jaxtyping import Float
from torch import Tensor

import saev.data
import saev.utils.scheduling
import saev.utils.wandb
from saev import configs, disk, helpers, nn
from saev.utils import statistics
from saev.utils.monitoring import DataloaderMonitor

logger = logging.getLogger("train.py")

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("train")


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    train_data: saev.data.ShuffledConfig = saev.data.ShuffledConfig()
    """Training data."""
    val_data: saev.data.ShuffledConfig = saev.data.ShuffledConfig()
    """Validation data."""
    n_train: int = 100_000_000
    """Number of SAE training samples."""
    n_val: int = 10_000_000
    """Number of SAE evaluation samples."""
    sae: nn.SparseAutoencoderConfig = nn.SparseAutoencoderConfig()
    """SAE configuration."""
    objective: nn.ObjectiveConfig = dataclasses.field(
        default_factory=nn.objectives.Vanilla
    )
    """SAE loss configuration."""
    n_sparsity_warmup: int = 0
    """Number of sparsity coefficient warmup steps."""
    lr: float = 0.0004
    """Learning rate."""
    n_lr_warmup: int = 500
    """Number of learning rate warmup steps."""
    grad_clip: float = 1.0
    """Maximum gradient norm across all SAE parameters."""

    # Logging
    track: bool = True
    """Whether to track with WandB."""
    wandb_project: str = "saev"
    """WandB project name."""
    tag: str = ""
    """Tag to add to WandB run."""
    log_every: int = 25
    """How often to log to WandB."""
    runs_root: pathlib.Path = pathlib.Path("$SAEV_NFS/saev/runs")
    """Root directory for runs."""

    device: tp.Literal["cuda", "cpu"] = "cuda"
    """Hardware device."""
    seed: int = 42
    """Random seed."""
    slurm_acct: str = ""
    """Slurm account string. Empty means to not use Slurm."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 24.0
    """Slurm job length in hours."""
    mem_gb: int = 128
    """Node memory in GB."""
    log_to: str = os.path.join(".", "logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
def make_saes(
    cfgs: list[tuple[nn.SparseAutoencoderConfig, nn.ObjectiveConfig]],
) -> tuple[torch.nn.ModuleList, torch.nn.ModuleList, list[dict[str, object]]]:
    saes, objectives, param_groups = [], [], []
    for sae_cfg, obj_cfg in cfgs:
        sae = nn.SparseAutoencoder(sae_cfg)
        saes.append(sae)
        # Use an empty LR because our first step is warmup.
        param_groups.append({"params": sae.parameters(), "lr": 0.0})
        objectives.append(nn.get_objective(obj_cfg))

    return torch.nn.ModuleList(saes), torch.nn.ModuleList(objectives), param_groups


@beartype.beartype
def worker_fn(cfgs: list[Config]) -> list[str]:
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    saes, objectives, run, steps = train(cfgs)
    # Cheap(-ish) evaluation
    eval_metrics = evaluate(cfgs, saes, objectives)
    metrics = [metric.for_wandb() for metric in eval_metrics]
    run.log(metrics, step=steps)
    ids = run.finish()

    for cfg, id, metric, sae in zip(cfgs, ids, eval_metrics, saes):
        logger.info(
            "Checkpoint %s has %d dense features (%.1f)",
            id,
            metric.n_dense,
            metric.n_dense / sae.cfg.d_sae * 100,
        )
        logger.info(
            "Checkpoint %s has %d dead features (%.1f%%)",
            id,
            metric.n_dead,
            metric.n_dead / sae.cfg.d_sae * 100,
        )
        logger.info(
            "Checkpoint %s has %d *almost* dead (<1e-7) features (%.1f)",
            id,
            metric.n_almost_dead,
            metric.n_almost_dead / sae.cfg.d_sae * 100,
        )

        run = disk.Run.new(
            id,
            train_shards_dir=cfg.train_data.shards,
            val_shards_dir=cfg.val_data.shards,
            runs_root=cfg.runs_root,
        )
        nn.dump(run.ckpt, sae)
        logger.info("Dumped checkpoint to '%s'.", run.ckpt)
        with open(run.run_dir / "checkpoint" / "config.json", "wb") as fd:
            helpers.dump(cfg, fd, option=orjson.OPT_INDENT_2)

    return ids


@beartype.beartype
def train(
    cfgs: list[Config],
) -> tuple[
    torch.nn.ModuleList, torch.nn.ModuleList, saev.utils.wandb.ParallelWandbRun, int
]:
    """
    Explicitly declare the optimizer, schedulers, dataloader, etc outside of `main` so that all the variables are dropped from scope and can be garbage collected.
    """
    if len(split_cfgs(cfgs)) != 1:
        raise ValueError("Configs are not parallelizeable: {cfgs}.")

    logger.info("Parallelizing %d runs.", len(cfgs))

    cfg = cfgs[0]
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True

    dataloader = saev.data.ShuffledDataLoader(cfg.train_data)
    dataloader = saev.utils.scheduling.BatchLimiter(dataloader, cfg.n_train)

    saes, objectives, param_groups = make_saes([(c.sae, c.objective) for c in cfgs])

    mode = "online" if cfg.track else "disabled"
    tags = [cfg.tag] if cfg.tag else []

    # Add metadata to configs for WandB logging
    metadata_dict = dataclasses.asdict(dataloader.metadata)
    wandb_configs = []
    for c in cfgs:
        cfg_dict = dataclasses.asdict(c)
        cfg_dict["train_data"]["metadata"] = metadata_dict
        wandb_configs.append(cfg_dict)

    run = saev.utils.wandb.ParallelWandbRun(
        cfg.wandb_project, wandb_configs, mode, tags
    )

    optimizer = torch.optim.Adam(param_groups, fused=True)
    lr_schedulers = [
        saev.utils.scheduling.WarmupCosine(
            0.0, c.n_lr_warmup, c.lr, len(dataloader), 0.0
        )
        for c in cfgs
    ]
    sparsity_schedulers = [
        saev.utils.scheduling.Warmup(
            0.0, c.objective.sparsity_coeff, c.n_sparsity_warmup
        )
        for c in cfgs
    ]

    saes.train()
    saes = saes.to(cfg.device)
    objectives.train()
    objectives = objectives.to(cfg.device)

    global_step, n_patches_seen = 0, 0
    dl_monitor = DataloaderMonitor(dataloader)

    for batch in helpers.progress(dataloader, every=cfg.log_every):
        acts_BD = batch["act"].to(cfg.device, non_blocking=True)
        for sae in saes:
            sae.normalize_w_dec()
        # Forward passes and loss calculations.
        losses = []
        for sae, objective in zip(saes, objectives):
            # Objective now handles the forward pass internally
            loss = objective(sae, acts_BD)
            losses.append(loss)

        n_patches_seen += len(acts_BD)

        for loss in losses:
            loss.loss.backward()

        # remove parallel gradients or normalize columns?
        for sae in saes:
            sae.remove_parallel_grads()

        # Calculate gradient norms before optimizer step
        grad_norms = []
        for sae, cfg in zip(saes, cfgs):
            # Clip gradients and get the gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                sae.parameters(), max_norm=cfg.grad_clip
            )

            grad_norms.append(grad_norm)

        # Log metrics after gradient computation
        if (global_step + 1) % cfg.log_every == 0:
            with torch.no_grad():
                now = time.time()
                dl_metrics = dl_monitor.compute(now=now)

                metadata = dataloader.metadata
                entropy_metrics = statistics.calc_batch_entropy(
                    batch["example_idx"].to("cpu"),
                    batch["token_idx"].to("cpu"),
                    metadata.n_examples,
                    metadata.content_tokens_per_example,
                )
                dl_metrics.update(entropy_metrics)

                metrics = []
                for i, (loss, sae, objective, group) in enumerate(
                    zip(losses, saes, objectives, optimizer.param_groups)
                ):
                    # Recompute f_x and x_hat for metrics (since we don't have them anymore)
                    with torch.no_grad():
                        f_x = sae.encode(acts_BD)
                        x_hat = sae.decode(f_x)

                    # Explained variance: 1 - Var(x - x_hat) / Var(x)
                    residual = acts_BD - x_hat[:, -1, :]
                    explained_var = 1 - residual.var() / acts_BD.var()

                    # Dead unit percentage: fraction of units that never activate
                    dead_pct = ((f_x.abs() > 1e-8).sum(0) == 0).float().mean()

                    # Dictionary coherence: max |<w_i, w_j>| for i != j
                    W = sae.W_dec  # (d_sae, d_model)
                    # Normalize each row (each SAE feature)
                    W_norm = W / W.norm(dim=1, keepdim=True)
                    coherence = (W_norm @ W_norm.T).abs().triu(1).max()

                    # Average decoder row L2 norm (since W_dec is d_sae x d_model)
                    avg_w_row_norm = sae.W_dec.norm(dim=1).mean()

                    metric = {
                        **loss.metrics(),
                        "progress/n_patches_seen": n_patches_seen,
                        "progress/learning_rate": group["lr"],
                        "progress/sparsity_coeff": objective.sparsity_coeff,
                        "metrics/explained_variance": explained_var.item(),
                        "metrics/dead_unit_pct": dead_pct.item(),
                        "metrics/dictionary_coherence": coherence.item(),
                        "metrics/avg_decoder_row_norm": avg_w_row_norm.item(),
                        "metrics/grad_norm": grad_norms[i].item(),
                        **dl_metrics,
                    }

                    metrics.append(metric)
                run.log(metrics, step=global_step)

                logger.info(
                    ", ".join(
                        f"{key}: {value:.5f}"
                        for key, value in losses[0].metrics().items()
                    )
                )

        optimizer.step()

        # Update LR and sparsity coefficients.
        for param_group, scheduler in zip(optimizer.param_groups, lr_schedulers):
            param_group["lr"] = scheduler.step()

        for objective, scheduler in zip(objectives, sparsity_schedulers):
            objective.sparsity_coeff = scheduler.step()

        # Don't need these anymore.
        optimizer.zero_grad()

        global_step += 1

    return saes, objectives, run, global_step


# TODO: I think this needs to be jaxtyped, but jaxtyped in a submitit context can cause real issues.
@beartype.beartype
@dataclasses.dataclass(frozen=True)
class EvalMetrics:
    """Results of evaluating a trained SAE on a datset."""

    l0: float
    """Mean L0 across all examples."""
    l1: float
    """Mean L1 across all examples."""
    mse: float
    """Mean MSE across all examples."""
    n_dead: int
    """Number of neurons that never fired on any example."""
    n_almost_dead: int
    """Number of neurons that fired on fewer than `almost_dead_threshold` of examples."""
    n_dense: int
    """Number of neurons that fired on more than `dense_threshold` of examples."""

    freqs: Float[Tensor, " d_sae"]
    """How often each feature fired."""
    mean_values: Float[Tensor, " d_sae"]
    """The mean value for each feature when it did fire."""

    almost_dead_threshold: float
    """Threshold for an "almost dead" neuron."""
    dense_threshold: float
    """Threshold for a dense neuron."""

    def for_wandb(self) -> dict[str, int | float]:
        dct = dataclasses.asdict(self)
        # Store arrays as tables.
        dct["freqs"] = wandb.Table(columns=["freq"], data=dct["freqs"][:, None].numpy())
        dct["mean_values"] = wandb.Table(
            columns=["mean_value"], data=dct["mean_values"][:, None].numpy()
        )
        return {f"eval/{key}": value for key, value in dct.items()}


@beartype.beartype
@torch.no_grad()
def evaluate(
    cfgs: list[Config], saes: torch.nn.ModuleList, objectives: torch.nn.ModuleList
) -> list[EvalMetrics]:
    """
    Evaluates SAE quality by counting dead and dense features and recording loss metrics. Also makes histogram plots to help human qualitative comparison.

    The metrics computed are mean ``L0``/``L1``/``MSE`` losses, the number of dead, almost dead, and dense neurons, plus per-feature firing frequencies and mean values.  A list of `EvalMetrics` is returned, one for each SAE.
    """

    torch.cuda.empty_cache()

    if len(split_cfgs(cfgs)) != 1:
        raise ValueError("Configs are not parallelizeable: {cfgs}.")

    saes.eval()

    cfg = cfgs[0]

    almost_dead_lim = 1e-7
    dense_lim = 1e-2

    dataloader = saev.data.ShuffledDataLoader(cfg.val_data)
    dataloader = saev.utils.scheduling.BatchLimiter(dataloader, cfg.n_val)

    n_fired = torch.zeros((len(cfgs), saes[0].cfg.d_sae))
    values = torch.zeros((len(cfgs), saes[0].cfg.d_sae))
    total_l0 = torch.zeros(len(cfgs))
    total_l1 = torch.zeros(len(cfgs))
    total_mse = torch.zeros(len(cfgs))

    for batch in helpers.progress(dataloader, desc="eval", every=cfg.log_every):
        acts_BD = batch["act"].to(cfg.device, non_blocking=True)
        for i, (sae, objective) in enumerate(zip(saes, objectives)):
            # Objective now handles the forward pass internally
            loss = objective(sae, acts_BD)
            # Get f_x for metrics
            f_x_BS = sae.encode(acts_BD)
            n_fired[i] += einops.reduce(f_x_BS > 0, "batch d_sae -> d_sae", "sum").cpu()
            values[i] += einops.reduce(f_x_BS, "batch d_sae -> d_sae", "sum").cpu()
            total_l0[i] += loss.l0.cpu()
            total_l1[i] += loss.l1.cpu()
            total_mse[i] += loss.mse.cpu()

    mean_values = values / n_fired
    freqs = n_fired / cfg.n_val

    l0 = (total_l0 / len(dataloader)).tolist()
    l1 = (total_l1 / len(dataloader)).tolist()
    mse = (total_mse / len(dataloader)).tolist()

    n_dead = einops.reduce(freqs == 0, "n_saes d_sae -> n_saes", "sum").tolist()
    n_almost_dead = einops.reduce(
        freqs < almost_dead_lim, "n_saes d_sae -> n_saes", "sum"
    ).tolist()
    n_dense = einops.reduce(freqs > dense_lim, "n_saes d_sae -> n_saes", "sum").tolist()

    metrics = []
    for row in zip(l0, l1, mse, n_dead, n_almost_dead, n_dense, freqs, mean_values):
        metrics.append(EvalMetrics(*row, almost_dead_lim, dense_lim))

    return metrics


#####################
# Parallel Training #
#####################


CANNOT_PARALLELIZE = set([
    "train_data",
    "val_data",
    "n_train",
    "n_val",
    "track",
    "wandb_project",
    "tag",
    "log_every",
    "runs_root",
    "device",
    "slurm_acct",
    "slurm_partition",
    "n_hours",
    "log_to",
    "sae.exp_factor",
    "sae.d_model",
])


@beartype.beartype
def split_cfgs(cfgs: list[Config]) -> list[list[Config]]:
    """
    Splits configs into groups that can be parallelized.

    Arguments:
        A list of configs from a sweep file.

    Returns:
        A list of lists, where the configs in each sublist do not differ in any keys that are in `CANNOT_PARALLELIZE`. This means that each sublist is a valid "parallel" set of configs for `train`.
    """
    # Group configs by their values for CANNOT_PARALLELIZE keys
    groups = {}
    for cfg in cfgs:
        dct = dataclasses.asdict(cfg)

        # Create a key tuple from the values of CANNOT_PARALLELIZE keys
        key_values = []
        for key in sorted(CANNOT_PARALLELIZE):
            key_values.append((key, helpers.make_hashable(helpers.get(dct, key))))
        group_key = tuple(key_values)

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(cfg)

    # Convert groups dict to list of lists
    return list(groups.values())


@beartype.beartype
def main(
    cfg: tp.Annotated[Config, tyro.conf.arg(name="")],
    sweep: pathlib.Path | None = None,
):
    """
    Train an SAE over activations, optionally running a parallel grid search over a set of hyperparameters.

    Args:
        cfg: Baseline config for training an SAE.
        sweep: Path to .py file defining the sweep parameters.
    """
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    import submitit

    if sweep is not None:
        sweep_dcts = configs.load_sweep(sweep)
        if not sweep_dcts:
            logger.error("No valid sweeps found in '%s'.", sweep)
            sys.exit(1)

        cfgs, errs = configs.load_cfgs(cfg, default=Config(), sweep_dcts=sweep_dcts)

        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return

    else:
        cfgs = [cfg]

    cfgs = split_cfgs(cfgs)

    logger.info("Running %d training jobs.", len(cfgs))

    if cfg.slurm_acct:
        executor = submitit.SlurmExecutor(folder=cfg.log_to)

        n_cpus = max(cfg.train_data.n_threads, cfg.val_data.n_threads) + 4
        if cfg.mem_gb // 10 > n_cpus:
            logger.info(
                "Using %d CPUs instead of %d to get more RAM.", cfg.mem_gb // 10, n_cpus
            )
            n_cpus = cfg.mem_gb // 10

        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=n_cpus,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    with executor.batch():
        jobs = [executor.submit(worker_fn, group) for group in cfgs]

    # Give the executor five seconds to fire the jobs off.
    time.sleep(5.0)

    # Log initial status.
    for j, job in enumerate(jobs):
        logger.info("Job %d/%d: %s %s", j + 1, len(jobs), job.job_id, job.state)

    for j, job in enumerate(jobs):
        try:
            job.result()
            logger.info("Job %d/%d finished.", j + 1, len(jobs))
        except submitit.core.utils.UncompletedJobError:
            logger.warning("Job %s (%d) did not finish.", job.job_id, j)

    logger.info("Jobs done.")


if __name__ == "__main__":
    tyro.cli(main)
