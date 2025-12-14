import dataclasses
import logging
import pathlib
import random
import typing as tp

import beartype
import numpy as np
import polars as pl
import scipy.sparse
import torch
import tyro
from jaxtyping import Float, Float32, jaxtyped
from torch import Tensor

import saev.data
import saev.disk
import saev.helpers
import saev.viz

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("visuals")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for unified activation computation."""

    run: pathlib.Path = pathlib.Path("./runs/016lmihg")
    """Run directory."""
    shards: pathlib.Path = pathlib.Path("./shards/abcdef01")
    """Activations."""

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """Which accelerator to use."""
    sae_batch_size: int = 1024 * 8
    """Batch size for SAE inference."""

    log_freq_range: tuple[float, float] = (-6.0, 1.0)
    """Log10 frequency range for which to save images."""
    log_value_range: tuple[float, float] = (-3.0, 3.0)
    """Log10 frequency range for which to save images."""
    latents: list[int] = dataclasses.field(default_factory=list)
    """Latents to always include, no matter what."""
    n_latents: int = 400
    """Number of latents to save audio clips for."""
    n_clips: int = 20
    """Number of audio clips to generate per feature."""
    seed: int = 42
    """Random seed."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Example:
    waveform: Float32[np.ndarray, "1 time"]
    sample_rate: float
    tokens: Float[np.ndarray, " content_tokens_per_example"]
    # Metadata
    idx: int
    target: int
    label: str
    extra: dict[str, object] = dataclasses.field(default_factory=dict)


@beartype.beartype
def safe_load(path: pathlib.Path) -> Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


@beartype.beartype
@torch.inference_mode()
def cli(cfg: tp.Annotated[Config, tyro.conf.arg(name="")]):
    """Generate visual outputs for particular latents."""

    try:
        run = saev.disk.Run(cfg.run)
        d_sae = run.config["sae"]["d_sae"]
        token_acts = scipy.sparse.load_npz(
            run.inference / cfg.shards.name / "token_acts.npz"
        )
        mean_values_s = safe_load(run.inference / cfg.shards.name / "mean_values.pt")
        sparsity_s = safe_load(run.inference / cfg.shards.name / "sparsity.pt")
    except FileNotFoundError as err:
        logger.error("Required activation files not found: %s. Run inference.py", err)
        return

    # Create indexed activations dataset for efficient patch retrieval
    md = saev.data.Metadata.load(cfg.shards)
    transformer = saev.data.models.load_model_cls(md.family)(md.ckpt)
    audio_cfg = md.make_data_cfg()
    audio_ds = saev.data.datasets.get_dataset(audio_cfg)

    logger.info("Loaded data.")

    # obs_df = pl.DataFrame(
    #     [img_ds.get_metadata(i) for i in range(len(img_ds))], infer_schema_length=None
    # )
    # obs_fpath = os.path.join(cfg.root, "obs.parquet")
    # obs_df.write_parquet(obs_fpath)
    # logger.info("Saved obs.parquet with %d rows to '%s'.", obs_df.height, obs_fpath)

    topk_example_idx = (
        saev.helpers.csr_topk(token_acts, k=cfg.top_k, axis=0).indices
        // md.content_tokens_per_example
    )

    var_df = pl.DataFrame({
        "feature": range(d_sae),
        "log10_freq": torch.log10(sparsity_s).tolist(),
        "log10_value": torch.log10(mean_values_s).tolist(),
        "topk_example_idx": topk_example_idx.T.tolist(),
    })
    var_fpath = run.inference / cfg.shards.name / "var.parquet"
    var_df.write_parquet(var_fpath)
    logger.info("Saved var.parquet with %d rows to '%s'.", var_df.height, var_fpath)

    min_log_freq, max_log_freq = cfg.log_freq_range
    min_log_value, max_log_value = cfg.log_value_range

    mask = (
        (min_log_freq < torch.log10(sparsity_s))
        & (torch.log10(sparsity_s) < max_log_freq)
        & (min_log_value < torch.log10(mean_values_s))
        & (torch.log10(mean_values_s) < max_log_value)
    )

    features = cfg.latents
    random_features = torch.arange(d_sae)[mask.cpu()].tolist()
    random.seed(cfg.seed)
    random.shuffle(random_features)
    features += random_features[: cfg.n_latents]

    topk_example_idx = np.stack(var_df.get_column("topk_example_idx").to_numpy())
    assert topk_example_idx.shape == (d_sae, cfg.top_k)
    topk_example_idx = topk_example_idx[features]
    topk_token_idx = (
        topk_example_idx[:, :, None] * md.content_tokens_per_example
        + np.arange(md.content_tokens_per_example)[None, None, :]
    )
    assert topk_token_idx.max() < token_acts.shape[0]
    logger.info("Calculated top-k for each latent.")

    patch_size = int(transformer.patch_size * cfg.img_scale)

    for f_i, f in enumerate(
        saev.helpers.progress(features, desc="saving clips", every=1)
    ):
        feature_dir = run.inference / cfg.shards.name / "clips" / str(f)
        feature_dir.mkdir(exist_ok=True, parents=True)

        f_token_idx = topk_token_idx[f_i]
        token_values_kp = token_acts[f_token_idx.ravel()][:, f].reshape(cfg.top_k, -1)
        token_values_kp = token_values_kp.toarray()
        examples = []

        seen_example_idx = set()
        for example_idx, token_values_p in zip(
            topk_example_idx[f_i].tolist(), token_values_kp
        ):
            if example_idx in seen_example_idx:
                continue
            sample = audio_ds[example_idx]

            example = Example(
                waveform=sample["audio"],
                sample_rate=sample["sample_rate"],
                tokens=token_values_p,
                idx=example_idx,
                target=sample["target"],
                label=sample["label"],
            )
            examples.append(example)
            seen_example_idx.add(example_idx)

        # How to scale values.
        upper = token_values_kp.max().item()

        for j, example in enumerate(examples):
            # 1. Save original spectogram under {j}_spectogram.png
            # 2. Save SAE-highlighted spectogram under {j}_sae_spectogram.png
            # 3. Save time-cropped audio under {j}_time_clip.ogg
            # 4. Save time- and freq-cropped audio under {j}_time_freq_clip.ogg

            # Example of saving with highlights for a natural image
            #
            # img_with_highlights = saev.viz.add_highlights(
            #     example.img, example.tokens, patch_size, upper=upper
            # )
            # img_with_highlights.save(feature_dir / f"{j}_sae_img.png")
            pass
