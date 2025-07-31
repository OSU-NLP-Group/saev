import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from scipy.stats import kstest, pareto

from saev.nn import modeling


def test_factories():
    assert isinstance(modeling.get_activation(modeling.Relu()), torch.nn.ReLU)
    assert isinstance(modeling.get_activation(modeling.TopK()), modeling.TopKActivation)
    assert isinstance(
        modeling.get_activation(modeling.BatchTopK()), modeling.BatchTopKActivation
    )


@st.composite
def relu_cfgs(draw):
    d_vit = draw(st.sampled_from([32, 64, 128]))
    exp = draw(st.sampled_from([2, 4]))
    return modeling.Relu(d_vit=d_vit, exp_factor=exp)


@st.composite
def topk_cfgs(draw):
    d_vit = draw(st.sampled_from([32, 64, 128]))
    exp = draw(st.sampled_from([2, 4]))
    k = draw(st.sampled_from([1, 2, 4, 8]))
    return modeling.TopK(d_vit=d_vit, exp_factor=exp, top_k=k)


@st.composite
def batch_topk_cfgs(draw):
    d_vit = draw(st.sampled_from([32, 64, 128]))
    exp = draw(st.sampled_from([2, 4]))
    k = draw(st.sampled_from([1, 2, 4, 8]))
    return modeling.BatchTopK(d_vit=d_vit, exp_factor=exp, top_k=k)


@given(cfg=topk_cfgs(), batch=st.integers(min_value=1, max_value=4))
def test_topk_activation(cfg, batch):
    act = modeling.get_activation(cfg)
    x = torch.randn(batch, cfg.d_vit * cfg.exp_factor)
    y = act(x)
    assert y.shape == (batch, cfg.d_vit * cfg.exp_factor)
    # Check that only k elements are non-zero per sample
    assert (y != 0).sum(dim=1).eq(cfg.top_k).all()


@given(cfg=batch_topk_cfgs(), batch=st.integers(min_value=1, max_value=4))
def test_batch_topk_activation(cfg, batch):
    act = modeling.get_activation(cfg)
    x = torch.randn(batch, cfg.d_vit * cfg.exp_factor)
    y = act(x)
    assert y.shape == (batch, cfg.d_vit * cfg.exp_factor)
    # Check that only k elements are non-zero per batch
    assert (y != 0).sum(dim=1).sum(dim=0).eq(cfg.top_k)


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
    modeling.Relu(d_vit=512, exp_factor=8, seed=0),
    modeling.Relu(d_vit=768, exp_factor=16, seed=1),
    modeling.Relu(d_vit=1024, exp_factor=32, seed=2),
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


@given(cfg=relu_cfgs(), n=st.integers(20, 100), pareto_power=st.floats(0.1, 0.8))
def test_sample_prefix(cfg, n, pareto_power):
    """Check to make sure that the prefix sampling follows a pareto dist."""
    sae = modeling.MatryoshkaSparseAutoencoder(cfg)
    assert isinstance(sae, modeling.MatryoshkaSparseAutoencoder)

    prefixes = sae.sample_prefixes(
        sae.cfg.d_sae, n, pareto_power=pareto_power, replacement=True
    )

    # Test against pareto distribution with the same params
    # cdf is scaled according to the max value of the cdf (at d_sae)
    # we also need to discretize the distribution since we are sampling integers.
    # Testing this using the equation is a bit tricky, so we sample integers instead.
    def pareto_cdf(x):
        return pareto.cdf(x, b=pareto_power) / pareto.cdf(sae.cfg.d_sae, b=pareto_power)

    pareto_pdf = np.diff(
        np.concatenate(([0], [pareto_cdf(i) for i in range(1, sae.cfg.d_sae + 1)]))
    )

    pareto_prefixes = np.random.choice(
        np.arange(1, sae.cfg.d_sae), size=n, replace=True, p=pareto_pdf[1:]
    )

    statistic, p_value = kstest(prefixes, pareto_prefixes)

    assert p_value > 0.01
