import torch

from saev.nn import modeling, objectives


def _make_identity_sae(d_model: int, alpha: float = 1.0, k_aux: int = 2):
    cfg = modeling.SparseAutoencoderConfig(
        d_model=d_model,
        d_sae=d_model,
        normalize_w_dec=False,
        remove_parallel_grads=False,
        activation=modeling.TopK(
            top_k=d_model, aux=modeling.AuxK(k_aux=k_aux, alpha=alpha)
        ),
    )
    sae = modeling.SparseAutoencoder(cfg)
    with torch.no_grad():
        sae.W_dec.copy_(torch.eye(d_model))
        sae.W_enc.copy_(torch.eye(d_model))
        sae.b_dec.zero_()
        sae.b_enc.zero_()
    return sae


def test_auxk_zero_dead_returns_zero_and_no_grad():
    sae = _make_identity_sae(4)
    x = torch.zeros(2, 4)
    pre = torch.ones(2, 4, requires_grad=True)
    out = modeling.SparseAutoencoder.DecodeOutput(
        pre_acts=pre, acts=pre, x_hats=torch.zeros(2, 1, 4)
    )
    dead_mask = torch.zeros(4, dtype=torch.bool)

    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    assert loss.item() == 0.0
    assert not loss.requires_grad or loss.grad_fn is None


def test_auxk_topk_value_matches_manual():
    sae = _make_identity_sae(4, alpha=1.0, k_aux=2)
    x = torch.zeros(1, 4)
    pre = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = modeling.SparseAutoencoder.DecodeOutput(
        pre_acts=pre, acts=pre, x_hats=torch.zeros(1, 1, 4)
    )
    dead_mask = torch.ones(4, dtype=torch.bool)

    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    expected = (3.0**2 + 4.0**2) / 4  # mean over d_model
    assert torch.allclose(loss, torch.tensor(expected))


def test_auxk_alpha_scales_loss():
    base = _make_identity_sae(4, alpha=1.0, k_aux=2)
    scaled = _make_identity_sae(4, alpha=0.5, k_aux=2)
    x = torch.zeros(1, 4)
    pre = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = modeling.SparseAutoencoder.DecodeOutput(
        pre_acts=pre, acts=pre, x_hats=torch.zeros(1, 1, 4)
    )
    dead_mask = torch.ones(4, dtype=torch.bool)
    base_loss = base.cfg.activation.aux.loss(
        sae=base, x=x, out=out, dead_mask=dead_mask
    )
    scaled_loss = scaled.cfg.activation.aux.loss(
        sae=scaled, x=x, out=out, dead_mask=dead_mask
    )
    assert torch.allclose(scaled_loss, base_loss * 0.5)


def test_auxk_clamps_k_to_dead_count():
    sae = _make_identity_sae(4, k_aux=8)
    x = torch.zeros(1, 4)
    pre = torch.tensor([[0.0, 0.0, 5.0, 0.0]])
    out = modeling.SparseAutoencoder.DecodeOutput(
        pre_acts=pre, acts=pre, x_hats=torch.zeros(1, 1, 4)
    )
    dead_mask = torch.tensor([False, True, True, False])
    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    expected = (5.0**2) / 4
    assert torch.allclose(loss, torch.tensor(expected))


def test_auxk_gradients_only_on_dead_selected_latents():
    sae = _make_identity_sae(4, k_aux=1)
    x = torch.zeros(1, 4)
    pre = torch.tensor([[1.0, 2.0, 3.0, 0.5]], requires_grad=True)
    out = modeling.SparseAutoencoder.DecodeOutput(
        pre_acts=pre, acts=pre, x_hats=torch.zeros(1, 1, 4)
    )
    dead_mask = torch.tensor([False, True, True, False])
    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    loss.backward()

    assert pre.grad[0, 2] != 0  # top dead latent selected
    assert pre.grad[0, 1] == 0  # dead but not selected by topk
    assert pre.grad[0, 0] == 0  # live latent
    assert pre.grad[0, 3] == 0  # live latent


def test_auxk_gradients_flow_to_decoder_dead_rows_only():
    sae = _make_identity_sae(4, k_aux=1)
    x = torch.zeros(1, 4)
    pre = torch.tensor([[1.0, 0.0, 3.0, 0.0]])
    out = modeling.SparseAutoencoder.DecodeOutput(
        pre_acts=pre, acts=pre, x_hats=torch.zeros(1, 1, 4)
    )
    dead_mask = torch.tensor([True, True, False, False])

    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    loss.backward()
    assert sae.W_dec.grad[0].abs().sum() > 0  # dead & selected
    assert sae.W_dec.grad[2].abs().sum() == 0  # live, should get no aux grad


def test_decode_returns_decodeoutput_with_prefix_dim():
    sae = _make_identity_sae(3)
    f = torch.ones(2, 3)
    dec = sae.decode(f)
    assert isinstance(dec, modeling.SparseAutoencoder.DecodeOutput)
    assert dec.x_hats.shape == (2, 1, 3)


def test_objective_aux_included_in_total_loss():
    sae = _make_identity_sae(4)
    obj = objectives.Matryoshka(n_prefixes=1)
    objective = objectives.get_objective(obj)
    x = torch.randn(2, 4)
    loss = objective(sae, x)
    assert torch.allclose(loss.loss, loss.mse + loss.sparsity + loss.aux)


def test_objective_aux_zero_for_relu():
    cfg = modeling.SparseAutoencoderConfig(
        d_model=4, d_sae=4, activation=modeling.Relu()
    )
    sae = modeling.SparseAutoencoder(cfg)
    obj = objectives.Matryoshka(n_prefixes=1)
    objective = objectives.get_objective(obj)
    x = torch.randn(2, 4)
    loss = objective(sae, x)
    assert torch.allclose(loss.aux, torch.zeros_like(loss.aux))


def test_full_backward_updates_mse_and_aux_paths():
    sae = _make_identity_sae(4, k_aux=1)
    obj = objectives.Matryoshka(n_prefixes=1)
    objective = objectives.get_objective(obj)
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    loss = objective(sae, x)
    loss.loss.backward()
    # Gradients should hit decoder and encoder
    assert sae.W_dec.grad is not None
    assert sae.W_enc.grad is not None
    assert x.grad is not None
