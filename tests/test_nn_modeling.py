import itertools
import typing as tp

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given, settings

from saev.nn import modeling


def sae_cfgs():
    return st.builds(
        lambda d_model, expansion: modeling.SparseAutoencoderConfig(
            d_model=d_model,
            d_sae=d_model * expansion,
        ),
        d_model=st.sampled_from([32, 64, 128]),
        expansion=st.sampled_from([2, 4]),
    )


def sae_cfgs_comprehensive():
    """Comprehensive SAE config strategy for testing various combinations."""
    # Define activation strategies
    relu_strategy = st.builds(modeling.Relu)
    topk_strategy = st.builds(
        modeling.TopK, top_k=st.sampled_from([8, 16, 32, 64, 128])
    )
    batch_topk_strategy = st.builds(
        modeling.BatchTopK, top_k=st.sampled_from([8, 16, 32, 64, 128])
    )

    activation_strategy = st.one_of(relu_strategy, topk_strategy, batch_topk_strategy)

    return st.builds(
        lambda d_model, expansion, seed, normalize_w_dec, remove_parallel_grads, activation: (
            modeling.SparseAutoencoderConfig(
                d_model=d_model,
                d_sae=d_model * expansion,
                normalize_w_dec=normalize_w_dec,
                remove_parallel_grads=remove_parallel_grads,
                activation=activation,
            )
        ),
        d_model=st.sampled_from([256, 384, 512, 768, 1024]),
        expansion=st.sampled_from([2, 4, 8, 16, 32]),
        seed=st.integers(min_value=0, max_value=100),
        normalize_w_dec=st.booleans(),
        remove_parallel_grads=st.booleans(),
        activation=activation_strategy,
    )


@settings(deadline=None)
@given(cfg=sae_cfgs(), batch=st.integers(min_value=1, max_value=4))
def test_sae_shapes(cfg, batch):
    sae = modeling.SparseAutoencoder(cfg)
    x = torch.randn(batch, cfg.d_model)
    out = sae(x)
    assert out.x_hats.shape == (batch, 1, cfg.d_model)
    assert out.f_x.shape == (batch, cfg.d_sae)


hf_ckpts = [
    "osunlp/SAE_BioCLIP_24K_ViT-B-16_iNat21",
    "osunlp/SAE_CLIP_24K_ViT-B-16_IN1K",
    "osunlp/SAE_DINOv2_24K_ViT-B-14_IN1K",
]


@pytest.mark.parametrize("repo_id", hf_ckpts)
@pytest.mark.slow
def test_load_existing_checkpoint(repo_id, tmp_path):
    pytest.importorskip("huggingface_hub")

    import huggingface_hub

    ckpt_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename="sae.pt", cache_dir=tmp_path
    )

    model = modeling.load(ckpt_path)

    # Smoke-test shapes & numerics
    x = torch.randn(2, model.cfg.d_model)
    out = model(x)
    assert out.x_hats.shape == x[:, None, :].shape
    assert out.f_x.shape[1] == model.cfg.d_sae
    # reconstruction shouldn’t be exactly identical, but should have finite values
    assert torch.isfinite(out.x_hats).all()


def test_dump_load_roundtrip_exhaustive(tmp_path):
    """Test dump/load roundtrip for all combinations of activations and various configs."""
    # Test all activation types with different configurations
    activation_cfgs = [cls() for cls in tp.get_args(modeling.ActivationConfig)]

    # Various SAE configurations - reduced set for faster testing
    sae_cfgs = [
        {"d_model": 256, "d_sae": 1024},
        {"d_model": 512, "d_sae": 4096, "normalize_w_dec": False},
        {"d_model": 384, "d_sae": 4608, "remove_parallel_grads": False},
    ]

    for i, (act_cfg, cfg_args) in enumerate(
        itertools.product(activation_cfgs, sae_cfgs)
    ):
        sae_cfg = modeling.SparseAutoencoderConfig(**cfg_args, activation=act_cfg)
        sae = modeling.SparseAutoencoder(sae_cfg)
        _ = sae(torch.randn(2, sae_cfg.d_model))  # touch all params once

        ckpt = tmp_path / f"sae_{i}.pt"
        modeling.dump(ckpt, sae)
        sae_loaded = modeling.load(ckpt)

        # configs identical
        assert sae_cfg == sae_loaded.cfg

        # tensors identical
        for k, v in sae.state_dict().items():
            torch.testing.assert_close(v, sae_loaded.state_dict()[k])


@pytest.mark.parametrize(
    "sae_cfg",
    [
        modeling.SparseAutoencoderConfig(d_model=512, d_sae=4096),
        modeling.SparseAutoencoderConfig(d_model=768, d_sae=12288),
        modeling.SparseAutoencoderConfig(d_model=1024, d_sae=32768),
    ],
)
def test_dump_load_roundtrip_simple(tmp_path, sae_cfg):
    """Write → load → verify state-dict & cfg equality."""
    sae = modeling.SparseAutoencoder(sae_cfg)
    _ = sae(torch.randn(2, sae_cfg.d_model))  # touch all params once

    ckpt = tmp_path / "sae.pt"
    modeling.dump(ckpt, sae)
    sae_loaded = modeling.load(ckpt)

    # configs identical
    assert sae_cfg == sae_loaded.cfg

    # tensors identical
    for k, v in sae.state_dict().items():
        torch.testing.assert_close(v, sae_loaded.state_dict()[k])


@given(sae_cfg=sae_cfgs_comprehensive())
@settings(deadline=None, max_examples=10)
def test_dump_load_roundtrip_hypothesis(sae_cfg):
    """Property-based test for dump/load roundtrip with random configurations."""
    import tempfile

    # Create SAE and test roundtrip
    sae = modeling.SparseAutoencoder(sae_cfg)
    _ = sae(torch.randn(2, sae_cfg.d_model))  # touch all params once

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt = f"{tmp_dir}/sae.pt"
        modeling.dump(ckpt, sae)
        sae_loaded = modeling.load(ckpt)

        # configs identical
        assert sae_cfg == sae_loaded.cfg

        # tensors identical
        for k, v in sae.state_dict().items():
            torch.testing.assert_close(v, sae_loaded.state_dict()[k])


def test_load_local_checkpoint(request):
    """Test loading a checkpoint from a local path specified via --ckpt-path."""
    ckpt_path = request.config.getoption("--ckpt-path")
    if ckpt_path is None:
        pytest.skip("No checkpoint path provided. Use --ckpt-path to specify one.")

    # Test loading the checkpoint
    model = modeling.load(ckpt_path)

    # Basic smoke tests
    assert isinstance(model, modeling.SparseAutoencoder)
    assert hasattr(model, "cfg")
    assert isinstance(model.cfg, modeling.SparseAutoencoderConfig)

    # Test forward pass
    x = torch.randn(2, model.cfg.d_model)
    x_hat, f_x = model(x)

    # Check shapes
    assert x_hat.shape == x.shape
    assert f_x.shape == (2, model.cfg.d_sae)

    # Check that outputs are finite
    assert torch.isfinite(x_hat).all()
    assert torch.isfinite(f_x).all()

    # Print some info about the loaded checkpoint
    print(f"\nLoaded checkpoint from: {ckpt_path}")
    print(f"Model config: d_model={model.cfg.d_model}, d_sae={model.cfg.d_sae}")
    print(f"Activation: {type(model.cfg.activation).__name__}")
    if hasattr(model.cfg.activation, "top_k"):
        print(f"Top-k value: {model.cfg.activation.top_k}")


schema4_ckpt = "/fs/ess/PAS2136/samuelstevens/saev/runs/3zih0tpa/checkpoint/sae.pt"


def test_load_schema4_with_seed(tmp_path):
    """Schema-4 checkpoints contain 'seed' in cfg which is not a SparseAutoencoderConfig field. The loader must strip it."""
    import json
    import pathlib

    fpath = pathlib.Path(schema4_ckpt)
    if not fpath.exists():
        pytest.skip(f"Schema-4 checkpoint not available at {fpath}")

    # Verify the checkpoint actually has the seed key (test precondition).
    with open(fpath, "rb") as fd:
        header = json.loads(fd.readline())
    assert header["schema"] == 4
    assert "seed" in header["cfg"], "Test precondition: checkpoint must contain 'seed'"

    # This is the actual test: loading should succeed, not raise TypeError.
    sae = modeling.load(fpath)
    assert isinstance(sae, modeling.SparseAutoencoder)
    assert sae.cfg.d_model == header["cfg"]["d_model"]


def test_remove_parallel_grads_handles_non_normalized_rows():
    cfg = modeling.SparseAutoencoderConfig(
        d_model=4, d_sae=4, normalize_w_dec=False, remove_parallel_grads=True
    )
    # Disable automatic normalization but keep removal on
    sae = modeling.SparseAutoencoder(cfg)

    # Artificial W_dec and grad
    with torch.no_grad():
        sae.W_dec.copy_(torch.randn_like(sae.W_dec))
    sae.W_dec.grad = torch.randn_like(sae.W_dec)

    # After removal, each row grad should be orthogonal to its row of W_dec
    sae.remove_parallel_grads()
    row_dots = (sae.W_dec.grad * sae.W_dec).sum(dim=1)
    assert torch.allclose(row_dots, torch.zeros_like(row_dots), atol=1e-6)
