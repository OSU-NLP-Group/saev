import dataclasses
import logging
import pathlib
import random
import typing as tp

import beartype
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.sparse
import soundfile as sf
import torch
import tyro
from jaxtyping import Float32, jaxtyped
from torch import Tensor

import saev.data
import saev.data.datasets as datasets
import saev.disk
import saev.helpers
from saev.data import bird_mae, models

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
    top_k: int = 32
    """Number of top examples per latent."""

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
    seed: int = 42
    """Random seed."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Example:
    waveform: Float32[np.ndarray, " samples"]
    sample_rate: int
    tokens: Float32[np.ndarray, " content_tokens_per_example"]
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
        sae_cfg_obj = run.config.get("sae")
        assert isinstance(sae_cfg_obj, dict), type(sae_cfg_obj)
        sae_cfg = tp.cast(dict[str, object], sae_cfg_obj)
        d_sae_obj = sae_cfg.get("d_sae")
        assert isinstance(d_sae_obj, int), type(d_sae_obj)
        d_sae = d_sae_obj
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
    data_transform, _ = models.load_model_cls(md.family).make_transforms(
        md.ckpt, md.content_tokens_per_example
    )

    audio_ds = datasets.get_dataset(md.make_data_cfg(), data_transform=lambda x: x)

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

    topk_example_idx = np.asarray(
        var_df.get_column("topk_example_idx").to_list(), dtype=np.int64
    )
    assert topk_example_idx.shape == (d_sae, cfg.top_k)
    topk_example_idx = topk_example_idx[features]
    topk_token_idx = (
        topk_example_idx[:, :, None] * md.content_tokens_per_example
        + np.arange(md.content_tokens_per_example)[None, None, :]
    )
    assert topk_token_idx.max() < token_acts.shape[0]
    logger.info("Calculated top-k for each latent.")

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

            waveform = sample["data"]
            sample_rate = sample["sample_rate"]

            example = Example(
                waveform=waveform.astype(np.float32),
                sample_rate=sample_rate,
                tokens=token_values_p,
                idx=example_idx,
                target=sample["target"],
                label=sample["label"],
            )
            examples.append(example)
            seen_example_idx.add(example_idx)

        for j, example in enumerate(examples):
            # We want to save 4 files for each example:
            # 1. {j}_spectrogram.png - Original log-mel spectrogram as an image
            # 2. {j}_sae_spectrogram.png - Spectrogram with SAE-highlighted patches overlaid
            # 3. {j}_time_clip.ogg - Audio cropped to time segments with high activations
            # 4. {j}_time_freq_clip.ogg - Audio cropped in time AND frequency-filtered
            #
            # Available data in `example`:
            #   - example.waveform: np.ndarray of shape [samples], raw audio at example.sample_rate Hz
            #   - example.sample_rate: int, should be 32000 for Bird-MAE
            #   - example.tokens: np.ndarray of shape [256], SAE activation values per patch
            #     Patch indexing: patch i -> time_patch = i // 8, mel_patch = i % 8
            #     So patches 0-7 are time=0 (all 8 mel bins), patches 8-15 are time=1, etc.
            #   - example.idx, example.target, example.label: metadata
            #
            # ========== STEP 1: Generate spectrogram ==========
            # Use data_transform() to convert waveform to log-mel spectrogram.
            # Input: numpy array of shape [samples]
            # Output: torch.Tensor of shape [512, 128] (time_frames, mel_bins)
            #
            # spectrogram = bird_mae.transform(example.waveform)  # [512, 128]
            #
            # To save as PNG, use matplotlib:
            #   import matplotlib.pyplot as plt
            #   fig, ax = plt.subplots()
            #   ax.imshow(spectrogram.T, aspect='auto', origin='lower', cmap='magma')
            #   ax.set_xlabel('Time frame')
            #   ax.set_ylabel('Mel bin')
            #   fig.savefig(feature_dir / f"{j}_spectrogram.png")
            #   plt.close(fig)
            #
            # ========== STEP 2: Highlighted spectrogram ==========
            # Overlay SAE activation values on the spectrogram.
            # example.tokens has shape [256] with float activation values.
            # Reshape to [32, 8] (time_patches x mel_patches) for visualization.
            #
            # Note: saev.viz.add_highlights() expects square images and won't work here.
            # Instead, create a custom overlay:
            #   patch_values = example.tokens.reshape(32, 8)  # [time, mel]
            #   # Upsample to match spectrogram resolution: 32->512 in time, 8->128 in mel
            #   # Each patch covers 16 frames and 16 mel bins
            #   overlay = np.repeat(np.repeat(patch_values, 16, axis=0), 16, axis=1)  # [512, 128]
            #   # Normalize overlay to [0, 1] for alpha blending
            #   overlay_norm = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
            #   # Blend with spectrogram using matplotlib
            #
            # ========== STEP 3: Time-cropped audio ==========
            # Use bird_mae.filter_audio() with mode="time" to extract audio segments
            # corresponding to patches with high SAE activation.
            #
            # First, convert float activations to a boolean mask:
            #   patches_bool = example.tokens > 0  # shape [256], dtype bool
            #
            # Then filter:
            #   from saev.data import bird_mae
            #   time_clip = bird_mae.filter_audio(
            #       torch.from_numpy(example.waveform),
            #       example.sample_rate,
            #       torch.from_numpy(patches_bool),
            #       mode="time"
            #   )
            #
            # Save as .ogg using soundfile:
            #   import soundfile as sf
            #   sf.write(feature_dir / f"{j}_time_clip.ogg", time_clip, example.sample_rate)
            #
            # ========== STEP 4: Time + frequency filtered audio ==========
            # Same as step 3 but with mode="time+freq":
            #   time_freq_clip = bird_mae.filter_audio(
            #       torch.from_numpy(example.waveform),
            #       example.sample_rate,
            #       torch.from_numpy(patches_bool),
            #       mode="time+freq"
            #   )
            #   sf.write(feature_dir / f"{j}_time_freq_clip.ogg", time_freq_clip, example.sample_rate)
            #

            spectrogram_tm = data_transform(example.waveform)
            assert isinstance(spectrogram_tm, torch.Tensor), type(spectrogram_tm)
            assert spectrogram_tm.shape == (512, 128), tuple(spectrogram_tm.shape)

            fig, ax = plt.subplots(figsize=(10, 4), dpi=150, layout="constrained")
            ax.imshow(
                spectrogram_tm.T.numpy(), aspect="auto", origin="lower", cmap="magma"
            )
            ax.set_xlabel("Time frame")
            ax.set_ylabel("Mel bin")
            fig.savefig(feature_dir / f"{j}_spectrogram.png")
            plt.close(fig)

            patch_values = example.tokens.reshape(32, 8)
            overlay = np.repeat(np.repeat(patch_values, 16, axis=0), 16, axis=1)
            assert overlay.shape == (512, 128), overlay.shape

            overlay_min = float(overlay.min())
            overlay_max = float(overlay.max())
            overlay_norm = (overlay - overlay_min) / (overlay_max - overlay_min + 1e-8)

            fig, ax = plt.subplots(figsize=(10, 4), dpi=150, layout="constrained")
            ax.imshow(
                spectrogram_tm.T.numpy(), aspect="auto", origin="lower", cmap="magma"
            )
            ax.imshow(
                overlay_norm.T,
                aspect="auto",
                origin="lower",
                cmap="Reds",
                alpha=0.5 * overlay_norm.T,
            )
            ax.set_xlabel("Time frame")
            ax.set_ylabel("Mel bin")
            fig.savefig(feature_dir / f"{j}_sae_spectrogram.png")
            plt.close(fig)

            patches_bool = torch.from_numpy(example.tokens > 0)
            waveform_t = torch.from_numpy(example.waveform)

            time_clip = bird_mae.filter_audio(
                waveform_t, example.sample_rate, patches_bool, mode="time"
            )
            if time_clip.numel() == 0:
                logger.warning("Empty time clip for example idx=%d.", example.idx)
                time_clip = torch.zeros(1, dtype=torch.float32)
            sf.write(
                feature_dir / f"{j}_time_clip.ogg",
                time_clip.numpy(),
                example.sample_rate,
            )

            time_freq_clip = bird_mae.filter_audio(
                waveform_t, example.sample_rate, patches_bool, mode="time+freq"
            )
            if time_freq_clip.numel() == 0:
                logger.warning("Empty time+freq clip for example idx=%d.", example.idx)
                time_freq_clip = torch.zeros(1, dtype=torch.float32)
            sf.write(
                feature_dir / f"{j}_time_freq_clip.ogg",
                time_freq_clip.numpy(),
                example.sample_rate,
            )
