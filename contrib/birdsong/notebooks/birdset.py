import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import opensoundscape as opso
    import os
    import datasets
    import marimo as mo

    import pathlib
    import soundfile as sf
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    import saev.data.datasets
    import torchaudio
    import torch
    import torch.nn.functional as F
    from torchaudio.compliance.kaldi import fbank

    from jaxtyping import Float, jaxtyped
    import beartype
    from torch import Tensor
    return (
        F,
        Float,
        Tensor,
        beartype,
        fbank,
        jaxtyped,
        librosa,
        mo,
        np,
        pathlib,
        plt,
        saev,
        torch,
        torchaudio,
    )


@app.cell
def _(pathlib):
    root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/datasets/birdclef-2025")
    return (root,)


@app.cell
def _(root, saev):
    cfg = saev.data.datasets.BirdClef2025(root)
    ds = saev.data.datasets.BirdClef2025Dataset(cfg)
    return (ds,)


@app.cell
def _(ds):
    ds[0]
    return


@app.cell
def _(ds, mo):
    mo.audio(ds[0]["audio"][: 5 * 32_000], rate=32_000)
    return


@app.cell
def _(ds, torchaudio):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=32_000, n_fft=2048, hop_length=512, n_mels=128
    )

    audio = ds[0]["audio"]
    sr = ds[0]["sample_rate"]
    return audio, sr


@app.cell
def _(F, Float, Tensor, audio, beartype, fbank, jaxtyped, plt, torch):
    BIRDMAE_MEAN = -7.2
    BIRDMAE_STD = 4.43
    SR = 32_000
    CLIP_SEC = 5
    TARGET_T = 512
    N_MELS = 128


    @jaxtyped(typechecker=beartype.beartype)
    def birdmae_fbank_from_waveform(
        waveform: Float[Tensor, " samples"],
    ) -> Float[Tensor, "time mels"]:
        """
        waveform: 1D tensor [samples]
        returns: 2D tensor [512, 128] as expected by Bird-MAE
        """

        (n_samples,) = waveform.shape
        # 1) pad/truncate to exactly 5 s
        max_len = SR * CLIP_SEC
        if n_samples < max_len:
            pad = max_len - n_samples
            waveform = F.pad(waveform, (0, pad))
        else:
            waveform = waveform[:max_len]

        # 2) mean-center (per clip)
        waveform = waveform - waveform.mean(dim=0, keepdim=True)
        print(waveform.shape)

        # 3) Kaldi fbank: [T, 128]
        fb = fbank(
            waveform[None, :],
            htk_compat=True,
            sample_frequency=SR,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=N_MELS,
            dither=0.0,
            frame_shift=10.0,
        )  # [T, 128]

        # 4) pad to 512 frames with min value
        t, _ = fb.shape
        if t < TARGET_T:
            diff = TARGET_T - t
            min_val = fb.min()
            fb = F.pad(fb, (0, 0, 0, diff), value=min_val.item())
        elif t > TARGET_T:
            fb = fb[:TARGET_T]

        fb = (fb - BIRDMAE_MEAN) / (BIRDMAE_STD * 2.0)

        return fb


    def _():
        spectogram = birdmae_fbank_from_waveform(torch.from_numpy(audio))
        fig, ax = plt.subplots()
        ax.imshow(spectogram.squeeze().numpy())
        return fig


    _()
    return


@app.cell
def _(audio, librosa, np, plt, root, sr):
    soundscape_path = next((root / "train_audio").glob("**/*.ogg"))


    print(audio.shape, sr)

    # Take the first 5 s chunk
    win_s = 5
    clip = audio[: win_s * sr]

    # Compute mel spectrogram (example params; you can swap in your own)
    n_fft = 2048
    hop_length = 512
    n_mels = 128

    S = librosa.feature.melspectrogram(
        y=clip,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 6), nrows=2, layout="constrained", dpi=300)

    ax1.plot(np.linspace(0, win_s, len(clip)), clip)
    ax1.set_title("Waveform (first 5s)")
    ax1.set_xlabel("Time")
    ax1.set_xlim(0, 5)


    librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax2
    )
    # ax2.set_colorbar(label="dB")
    ax2.set_title("Mel spectrogram")
    ax2.set_xlim(0, 5)

    fig
    return


@app.cell
def _(mo):
    mo.audio(
        src="/fs/ess/PAS2136/samuelstevens/datasets/birdclef-2025/train_audio/21211/XC882652.ogg"
    )
    return


@app.cell
def _():
    # ds = datasets.load_dataset("samuelstevens/BirdSet", "XCM")
    return


if __name__ == "__main__":
    app.run()
