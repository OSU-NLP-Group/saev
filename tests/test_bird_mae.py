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
