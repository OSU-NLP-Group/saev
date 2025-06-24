import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given, settings

from .. import config
from . import modeling


def test_factories():
    assert isinstance(modeling.get_activation(config.Relu()), torch.nn.ReLU)


@st.composite
def relu_cfgs(draw):
    d_vit = draw(st.sampled_from([32, 64, 128]))
    exp = draw(st.sampled_from([2, 4]))
    return config.Relu(d_vit=d_vit, exp_factor=exp)


@settings(deadline=None)
@given(cfg=relu_cfgs(), batch=st.integers(min_value=1, max_value=4))
def test_sae_shapes(cfg, batch):
    sae = modeling.SparseAutoencoder(cfg)
    x = torch.randn(batch, cfg.d_vit)
    x_hat, f = sae(x)
    assert x_hat.shape == (batch, cfg.d_vit)
    assert f.shape == (batch, cfg.d_sae)


hf_ckpts = [
    "osunlp/SAE_BioCLIP_24K_ViT-B-16_iNat21",
    "osunlp/SAE_CLIP_24K_ViT-B-16_IN1K",
    "osunlp/SAE_DINOv2_24K_ViT-B-14_IN1K",
]


@pytest.mark.parametrize("repo_id", hf_ckpts)
@pytest.mark.slow
def test_load_bioclip_checkpoint(repo_id, tmp_path):
    pytest.importorskip("huggingface_hub")

    import huggingface_hub

    ckpt_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename="sae.pt", cache_dir=tmp_path
    )

    model = modeling.load(ckpt_path)

    # Smoke-test shapes & numerics
    x = torch.randn(2, model.cfg.d_vit)
    x_hat, f_x = model(x)
    assert x_hat.shape == x.shape
    assert f_x.shape[1] == model.cfg.d_sae
    # reconstruction shouldn’t be exactly identical, but should have finite values
    assert torch.isfinite(x_hat).all()


roundtrip_cases = [
    config.Relu(d_vit=512, exp_factor=8, seed=0),
    config.Relu(d_vit=768, exp_factor=16, seed=1),
    config.Relu(d_vit=1024, exp_factor=32, seed=2),
]


@pytest.mark.parametrize("sae_cfg", roundtrip_cases)
def test_dump_load_roundtrip(tmp_path, sae_cfg):
    """Write → load → verify state-dict & cfg equality."""
    sae = modeling.SparseAutoencoder(sae_cfg)
    _ = sae(torch.randn(2, sae_cfg.d_vit))  # touch all params once

    ckpt = tmp_path / "sae.pt"
    modeling.dump(str(ckpt), sae)
    sae_loaded = modeling.load(str(ckpt))

    # configs identical
    assert sae_cfg == sae_loaded.cfg

    # tensors identical
    for k, v in sae.state_dict().items():
        torch.testing.assert_close(v, sae_loaded.state_dict()[k])
