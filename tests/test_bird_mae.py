import librosa
import pytest
import torch
import transformers

from saev.data import bird_mae

CKPTS = [
    "Bird-MAE-Base",
    "Bird-MAE-Large",
    "Bird-MAE-Huge",
]
DTYPE = torch.float32
ATOL, RTOL = 1e-5, 1e-4


@pytest.fixture(scope="session", params=CKPTS)
def models(request):
    ckpt = request.param
    hf = transformers.AutoModel.from_pretrained(
        f"DBD-research-group/{ckpt}", trust_remote_code=True
    ).eval()
    bio = bird_mae.Transformer(ckpt).eval().to(DTYPE)
    return hf, bio


def _rand(*, batch: int = 1):
    torch.manual_seed(0)
    return torch.rand(batch, 1, 512, 128, dtype=DTYPE)


def test_same_shape_single(models):
    ref, ours = models
    batch = _rand()
    h = ref(batch, output_hidden_states=True)
    o = ours(batch)
    assert h.hidden_states[-1][:, 1:, :].shape == o[:, 1:, :].shape
    assert h.last_hidden_state.shape == o[:, 0, :].shape


def test_values_close_single(models):
    ref, ours = models
    batch = _rand()
    h = ref(batch, output_hidden_states=True)
    o = ours(batch)
    torch.testing.assert_close(
        h.hidden_states[-1][:, 1:], o[:, 1:, :], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(h.last_hidden_state, o[:, 0, :], atol=ATOL, rtol=RTOL)


def test_values_close_batch(models):
    ref, ours = models
    batch = _rand(batch=4)
    h = ref(batch, output_hidden_states=True)
    o = ours(batch)
    torch.testing.assert_close(
        h.hidden_states[-1][:, 1:], o[:, 1:, :], atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(h.last_hidden_state, o[:, 0, :], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("ckpt", CKPTS)
def test_transform_matches_hf_feature_extractor(ckpt):
    waveform, _ = librosa.load(librosa.ex("robin"), sr=32_000)

    hf_extractor = transformers.AutoFeatureExtractor.from_pretrained(
        f"DBD-research-group/{ckpt}", trust_remote_code=True
    )
    hf_mel = hf_extractor(waveform).squeeze()
    ours_mel = bird_mae.transform(waveform)

    torch.testing.assert_close(hf_mel, ours_mel, atol=ATOL, rtol=RTOL)


# ===================== #
# filter_audio tests    #
# ===================== #

# Constants for filter_audio tests
SR = 32_000
N_TIME_PATCHES = 32
N_MEL_PATCHES = 8
N_PATCHES = N_TIME_PATCHES * N_MEL_PATCHES  # 256
SAMPLES_PER_TIME_PATCH = 16 * 320  # 16 frames * 320 samples/frame = 5,120


@pytest.fixture
def waveform_5s():
    """5 seconds of audio at 32kHz with recognizable pattern."""
    n_samples = SR * 5  # 160,000
    # Use a simple ramp so we can verify which samples are extracted
    return torch.arange(n_samples, dtype=torch.float32) / n_samples


def _make_patches(time_indices: list[int]) -> torch.Tensor:
    """Create a boolean patch mask with all mel patches activated for given time indices."""
    patches = torch.zeros(N_PATCHES, dtype=torch.bool)
    for t in time_indices:
        patches[t * N_MEL_PATCHES : (t + 1) * N_MEL_PATCHES] = True
    return patches


def test_filter_audio_time_single_first_patch(waveform_5s):
    """Activate first time patch (t=0), expect samples from start of audio."""
    patches = _make_patches([0])
    result = bird_mae.filter_audio(waveform_5s, SR, patches, mode="time")

    assert result.shape[0] == SAMPLES_PER_TIME_PATCH
    # First time patch should be first 5,120 samples
    assert torch.equal(result, waveform_5s[:SAMPLES_PER_TIME_PATCH])


def test_filter_audio_time_single_middle_patch(waveform_5s):
    """Activate middle time patch (t=15), expect samples from middle of audio."""
    patches = _make_patches([15])
    result = bird_mae.filter_audio(waveform_5s, SR, patches, mode="time")

    assert result.shape[0] == SAMPLES_PER_TIME_PATCH
    start = 15 * SAMPLES_PER_TIME_PATCH
    end = 16 * SAMPLES_PER_TIME_PATCH
    assert torch.equal(result, waveform_5s[start:end])


def test_filter_audio_time_single_last_patch(waveform_5s):
    """Activate last time patch (t=31), expect samples from end of audio."""
    patches = _make_patches([31])
    result = bird_mae.filter_audio(waveform_5s, SR, patches, mode="time")

    start = 31 * SAMPLES_PER_TIME_PATCH
    # Last patch may be truncated to audio length
    expected = waveform_5s[start:]
    assert result.shape[0] == expected.shape[0]
    assert torch.equal(result, expected)


def test_filter_audio_time_consecutive_patches(waveform_5s):
    """Activate two consecutive time patches, expect contiguous segment."""
    patches = _make_patches([10, 11])
    result = bird_mae.filter_audio(waveform_5s, SR, patches, mode="time")

    assert result.shape[0] == 2 * SAMPLES_PER_TIME_PATCH
    start = 10 * SAMPLES_PER_TIME_PATCH
    end = 12 * SAMPLES_PER_TIME_PATCH
    assert torch.equal(result, waveform_5s[start:end])


def test_filter_audio_time_non_consecutive_patches(waveform_5s):
    """Activate two non-consecutive time patches, expect concatenated segments."""
    patches = _make_patches([5, 20])
    result = bird_mae.filter_audio(waveform_5s, SR, patches, mode="time")

    assert result.shape[0] == 2 * SAMPLES_PER_TIME_PATCH
    seg1 = waveform_5s[5 * SAMPLES_PER_TIME_PATCH : 6 * SAMPLES_PER_TIME_PATCH]
    seg2 = waveform_5s[20 * SAMPLES_PER_TIME_PATCH : 21 * SAMPLES_PER_TIME_PATCH]
    expected = torch.cat([seg1, seg2], dim=0)
    assert torch.equal(result, expected)


def test_filter_audio_time_single_mel_patch_activates_full_time(waveform_5s):
    """Activate only one mel patch in a time slot, should still extract full time segment."""
    patches = torch.zeros(N_PATCHES, dtype=torch.bool)
    patches[10 * N_MEL_PATCHES + 3] = True  # time=10, mel=3
    result = bird_mae.filter_audio(waveform_5s, SR, patches, mode="time")

    assert result.shape[0] == SAMPLES_PER_TIME_PATCH
    start = 10 * SAMPLES_PER_TIME_PATCH
    end = 11 * SAMPLES_PER_TIME_PATCH
    assert torch.equal(result, waveform_5s[start:end])


def test_filter_audio_time_all_patches(waveform_5s):
    """Activate all patches, expect full audio returned."""
    patches = torch.ones(N_PATCHES, dtype=torch.bool)
    result = bird_mae.filter_audio(waveform_5s, SR, patches, mode="time")

    # Full audio, possibly truncated at 32 * 5120 = 163,840 but audio is 160,000
    assert torch.equal(result, waveform_5s)


def test_filter_audio_time_no_patches(waveform_5s):
    """Activate no patches, expect empty array."""
    patches = torch.zeros(N_PATCHES, dtype=torch.bool)
    result = bird_mae.filter_audio(waveform_5s, SR, patches, mode="time")

    assert result.shape[0] == 0


def test_filter_audio_time_first_and_last_patches(waveform_5s):
    """Activate first and last time patches, expect two segments concatenated."""
    patches = _make_patches([0, 31])
    result = bird_mae.filter_audio(waveform_5s, SR, patches, mode="time")

    seg1 = waveform_5s[:SAMPLES_PER_TIME_PATCH]
    seg2 = waveform_5s[31 * SAMPLES_PER_TIME_PATCH :]
    expected = torch.cat([seg1, seg2], dim=0)
    assert torch.equal(result, expected)


def test_filter_audio_time_freq_handles_non_aligned_length():
    """time+freq should not error when waveform length is not hop-aligned."""
    # Length chosen to be non-multiple of hop_length (320)
    n_samples = SR * 5 + 123
    waveform = torch.arange(n_samples, dtype=torch.float32) / n_samples
    patches = _make_patches([0])  # activate first time patch

    out = bird_mae.filter_audio(waveform, SR, patches, mode="time+freq")
    assert isinstance(out, torch.Tensor)
    assert out.numel() > 0


def test_filter_audio_clips_to_5s_when_longer():
    """filter_audio should truncate/pad to same 5s window as transform()."""
    n_samples = SR * 7  # longer than 5s
    waveform = torch.arange(n_samples, dtype=torch.float32) / n_samples
    patches = _make_patches([0, 31])

    out = bird_mae.filter_audio(waveform, SR, patches, mode="time")
    # Patch 0 is full-length; patch 31 is truncated to remaining samples in the 5s window.
    expected_tail = waveform[31 * SAMPLES_PER_TIME_PATCH : SR * 5]
    assert out.numel() == SAMPLES_PER_TIME_PATCH + expected_tail.numel()
    assert torch.equal(out[:SAMPLES_PER_TIME_PATCH], waveform[:SAMPLES_PER_TIME_PATCH])
    assert torch.equal(out[SAMPLES_PER_TIME_PATCH:], expected_tail)


def test_filter_audio_time_freq_handles_much_longer():
    """time+freq should not error when waveform is much longer than 5s (clipped inside)."""
    n_samples = SR * 10 + 77
    waveform = torch.arange(n_samples, dtype=torch.float32) / n_samples
    patches = _make_patches([0, 10, 20])

    out = bird_mae.filter_audio(waveform, SR, patches, mode="time+freq")
    assert isinstance(out, torch.Tensor)
    assert out.numel() > 0
