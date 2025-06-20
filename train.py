# train.py
"""
Trains many SAEs in parallel to amortize the cost of loading a single batch of data over many SAE training runs.
"""

import collections.abc
import dataclasses
import itertools
import json
import logging
import os.path
import tomllib
import typing

import beartype
import einops
import numpy as np
import torch
import tyro
import wandb
from jaxtyping import Float
from torch import Tensor

import saev.data
import saev.data.iterable
import saev.utils.scheduling
import saev.utils.wandb
from saev import helpers, nn

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("train")


@beartype.beartype
@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    data: saev.data.IterableConfig = dataclasses.field(
        default_factory=saev.data.IterableConfig
    )
    """Data configuration"""
    n_workers: int = 32
    """Number of dataloader workers."""
    n_patches: int = 100_000_000
    """Number of SAE training examples."""
    sae: nn.SparseAutoencoderConfig = dataclasses.field(
        default_factory=nn.modeling.Relu
    )
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
    sae_batch_size: int = 1024 * 16
    """Batch size for SAE training."""

    # Logging
    track: bool = True
    """Whether to track with WandB."""
    wandb_project: str = "saev"
    """WandB project name."""
    tag: str = ""
    """Tag to add to WandB run."""
    log_every: int = 25
    """How often to log to WandB."""
    ckpt_path: str = os.path.join(".", "checkpoints")
    """Where to save checkpoints."""

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


# TODO: can we remove this?
@torch.no_grad()
def init_b_dec_batched(
    saes: torch.nn.ModuleList, dataset: saev.data.iterable.DataLoader
):
    n_samples = max(sae.cfg.n_reinit_samples for sae in saes)
    if not n_samples:
        return
    # Pick random samples using first SAE's seed.
    perm = np.random.default_rng(seed=saes[0].cfg.seed).permutation(len(dataset))
    perm = perm[:n_samples]
    examples, _, _ = zip(*[
        dataset[p.item()]
        for p in helpers.progress(perm, every=25_000, desc="examples to re-init b_dec")
    ])
    vit_acts = torch.stack(examples)
    for sae in saes:
        sae.init_b_dec(vit_acts[: sae.cfg.n_reinit_samples])


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

        ckpt_fpath = os.path.join(cfg.ckpt_path, id, "sae.pt")
        nn.dump(ckpt_fpath, sae)
        logger.info("Dumped checkpoint to '%s'.", ckpt_fpath)
        cfg_fpath = os.path.join(cfg.ckpt_path, id, "config.json")
        with open(cfg_fpath, "w") as fd:
            json.dump(dataclasses.asdict(cfg), fd, indent=4)

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
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    dataloader = saev.data.iterable.DataLoader(cfg.data)
    saes, objectives, param_groups = make_saes([(c.sae, c.objective) for c in cfgs])

    mode = "online" if cfg.track else "disabled"
    tags = [cfg.tag] if cfg.tag else []
    run = saev.utils.wandb.ParallelWandbRun(
        cfg.wandb_project, [dataclasses.asdict(c) for c in cfgs], mode, tags
    )

    optimizer = torch.optim.Adam(param_groups, fused=True)
    lr_schedulers = [
        saev.utils.scheduling.Warmup(0.0, c.lr, c.n_lr_warmup) for c in cfgs
    ]
    sparsity_schedulers = [
        saev.utils.scheduling.Warmup(
            0.0, c.objective.sparsity_coeff, c.n_sparsity_warmup
        )
        for c in cfgs
    ]

    dataloader = saev.utils.scheduling.BatchLimiter(dataloader, cfg.n_patches)

    saes.train()
    saes = saes.to(cfg.device)
    objectives.train()
    objectives = objectives.to(cfg.device)

    global_step, n_patches_seen = 0, 0

    for batch in helpers.progress(dataloader, every=cfg.log_every):
        acts_BD = batch["act"].to(cfg.device, non_blocking=True)
        for sae in saes:
            sae.normalize_w_dec()
        # Forward passes and loss calculations.
        losses = []
        for sae, objective in zip(saes, objectives):
            x_hat, f_x = sae(acts_BD)
            losses.append(objective(acts_BD, f_x, x_hat))

        n_patches_seen += len(acts_BD)

        with torch.no_grad():
            if (global_step + 1) % cfg.log_every == 0:
                metrics = [
                    {
                        **loss.metrics(),
                        "progress/n_patches_seen": n_patches_seen,
                        "progress/learning_rate": group["lr"],
                        "progress/sparsity_coeff": objective.sparsity_coeff,
                    }
                    for loss, sae, objective, group in zip(
                        losses, saes, objectives, optimizer.param_groups
                    )
                ]
                run.log(metrics, step=global_step)

                logger.info(
                    ", ".join(
                        f"{key}: {value:.5f}"
                        for key, value in losses[0].metrics().items()
                    )
                )

        for loss in losses:
            loss.loss.backward()

        for sae in saes:
            sae.remove_parallel_grads()

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

    dataloader = saev.data.iterable.DataLoader(cfg.data)

    n_fired = torch.zeros((len(cfgs), saes[0].cfg.d_sae))
    values = torch.zeros((len(cfgs), saes[0].cfg.d_sae))
    total_l0 = torch.zeros(len(cfgs))
    total_l1 = torch.zeros(len(cfgs))
    total_mse = torch.zeros(len(cfgs))

    for batch in helpers.progress(dataloader, desc="eval", every=cfg.log_every):
        acts_BD = batch["act"].to(cfg.device, non_blocking=True)
        for i, (sae, objective) in enumerate(zip(saes, objectives)):
            x_hat_BD, f_x_BS = sae(acts_BD)
            loss = objective(acts_BD, f_x_BS, x_hat_BD)
            n_fired[i] += einops.reduce(f_x_BS > 0, "batch d_sae -> d_sae", "sum").cpu()
            values[i] += einops.reduce(f_x_BS, "batch d_sae -> d_sae", "sum").cpu()
            total_l0[i] += loss.l0.cpu()
            total_l1[i] += loss.l1.cpu()
            total_mse[i] += loss.mse.cpu()

    mean_values = values / n_fired
    freqs = n_fired / len(dataset)

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


##########
# SWEEPS #
##########


@beartype.beartype
def grid(cfg: Config, sweep_dct: dict[str, object]) -> tuple[list[Config], list[str]]:
    cfgs, errs = [], []
    for d, dct in enumerate(expand(sweep_dct)):
        # .sae is a nested field that cannot be naively expanded.
        sae_dct = dct.pop("sae")
        if sae_dct:
            sae_dct["seed"] = sae_dct.pop("seed", cfg.sae.seed) + cfg.seed + d
            dct["sae"] = dataclasses.replace(cfg.sae, **sae_dct)

        # .objective is a nested field that cannot be naively expanded.
        objective_dct = dct.pop("objective")
        if objective_dct:
            dct["objective"] = dataclasses.replace(cfg.objective, **objective_dct)

        # .data is a nested field that cannot be naively expanded.
        data_dct = dct.pop("data")
        if data_dct:
            dct["data"] = dataclasses.replace(cfg.data, **data_dct)

        try:
            cfgs.append(dataclasses.replace(cfg, **dct, seed=cfg.seed + d))
        except Exception as err:
            errs.append(str(err))

    return cfgs, errs


@beartype.beartype
def expand(config: dict[str, object]) -> collections.abc.Iterator[dict[str, object]]:
    """
    Expands dicts with (nested) lists into a list of (nested) dicts.
    """
    yield from _expand_discrete(config)


@beartype.beartype
def _expand_discrete(
    config: dict[str, object],
) -> collections.abc.Iterator[dict[str, object]]:
    """
    Expands any (possibly nested) list values in `config`
    """
    if not config:
        yield config
        return

    key, value = config.popitem()

    if isinstance(value, list):
        # Expand
        for c in _expand_discrete(config):
            for v in value:
                yield {**c, key: v}
    elif isinstance(value, dict):
        # Expand
        for c, v in itertools.product(
            _expand_discrete(config), _expand_discrete(value)
        ):
            yield {**c, key: v}
    else:
        for c in _expand_discrete(config):
            yield {**c, key: value}


#####################
# Parallel Training #
#####################


CANNOT_PARALLELIZE = set([
    "data",
    "n_workers",
    "n_patches",
    "sae_batch_size",
    "track",
    "wandb_project",
    "tag",
    "log_every",
    "ckpt_path",
    "device",
    "slurm_acct",
    "slurm_partition",
    "n_hours",
    "log_to",
    "sae.exp_factor",
    "sae.d_vit",
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
            key_values.append((key, make_hashable(helpers.get(dct, key))))
        group_key = tuple(key_values)

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(cfg)

    # Convert groups dict to list of lists
    return list(groups.values())


def make_hashable(obj):
    return json.dumps(obj, sort_keys=True)


@beartype.beartype
def main(
    cfg: typing.Annotated[Config, tyro.conf.arg(name="")], sweep: str | None = None
):
    """
    Train an SAE over activations, optionally running a parallel grid search over a set of hyperparameters.

    Args:
        cfg: Baseline config for training an SAE.
        sweep: Path to .toml file defining the sweep parameters.
    """
    import submitit

    if sweep is not None:
        with open(sweep, "rb") as fd:
            cfgs, errs = grid(cfg, tomllib.load(fd))

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
        executor.update_parameters(
            time=int(cfg.n_hours * 60),
            partition=cfg.slurm_partition,
            gpus_per_node=1,
            ntasks_per_node=1,
            cpus_per_task=cfg.n_workers + 4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=cfg.log_to)

    jobs = [executor.submit(worker_fn, group) for group in cfgs]
    for job in jobs:
        job.result()


if __name__ == "__main__":
    tyro.cli(main)
