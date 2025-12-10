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
    out = modeling.SparseAutoencoder.Output(
        h_x=pre, f_x=pre, x_hats=torch.zeros(2, 1, 4)
    )
    dead_mask = torch.zeros(4, dtype=torch.bool)

    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    assert loss.item() == 0.0
    assert not loss.requires_grad or loss.grad_fn is None


def test_auxk_topk_value_matches_manual():
    sae = _make_identity_sae(4, alpha=1.0, k_aux=2)
    x = torch.zeros(1, 4)
    pre = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = modeling.SparseAutoencoder.Output(
        h_x=pre, f_x=pre, x_hats=torch.zeros(1, 1, 4)
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
    out = modeling.SparseAutoencoder.Output(
        h_x=pre, f_x=pre, x_hats=torch.zeros(1, 1, 4)
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
    out = modeling.SparseAutoencoder.Output(
        h_x=pre, f_x=pre, x_hats=torch.zeros(1, 1, 4)
    )
    dead_mask = torch.tensor([False, True, True, False])
    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    expected = (5.0**2) / 4
    assert torch.allclose(loss, torch.tensor(expected))


def test_auxk_gradients_only_on_dead_selected_latents():
    sae = _make_identity_sae(4, k_aux=1)
    x = torch.zeros(1, 4)
    pre = torch.tensor([[1.0, 2.0, 3.0, 0.5]], requires_grad=True)
    out = modeling.SparseAutoencoder.Output(
        h_x=pre, f_x=pre, x_hats=torch.zeros(1, 1, 4)
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
    out = modeling.SparseAutoencoder.Output(
        h_x=pre, f_x=pre, x_hats=torch.zeros(1, 1, 4)
    )
    dead_mask = torch.tensor([True, True, False, False])

    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    loss.backward()
    assert sae.W_dec.grad[0].abs().sum() > 0  # dead & selected
    assert sae.W_dec.grad[2].abs().sum() == 0  # live, should get no aux grad


def test_decode_returns_tensor_with_prefix_dim():
    sae = _make_identity_sae(3)
    f = torch.ones(2, 3)
    x_hats = sae.decode(f)
    assert isinstance(x_hats, torch.Tensor)
    assert x_hats.shape == (2, 1, 3)


def test_objective_aux_included_in_total_loss():
    sae = _make_identity_sae(4)
    obj = objectives.Matryoshka(n_prefixes=1)
    objective = objectives.get_objective(obj)
    x = torch.randn(2, 4)
    loss, _ = objective(sae, x)
    assert torch.allclose(loss.loss, loss.mse + loss.sparsity + loss.aux)


def test_objective_aux_zero_for_relu():
    cfg = modeling.SparseAutoencoderConfig(
        d_model=4, d_sae=4, activation=modeling.Relu()
    )
    sae = modeling.SparseAutoencoder(cfg)
    obj = objectives.Matryoshka(n_prefixes=1)
    objective = objectives.get_objective(obj)
    x = torch.randn(2, 4)
    loss, _ = objective(sae, x)
    assert torch.allclose(loss.aux, torch.zeros_like(loss.aux))


def test_full_backward_updates_mse_and_aux_paths():
    sae = _make_identity_sae(4, k_aux=1)
    obj = objectives.Matryoshka(n_prefixes=1)
    objective = objectives.get_objective(obj)
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    loss, _ = objective(sae, x)
    loss.loss.backward()
    # Gradients should hit decoder and encoder
    assert sae.W_dec.grad is not None
    assert sae.W_enc.grad is not None
    assert x.grad is not None


def test_auxk_with_nonzero_residual():
    """Test AuxK loss formula ||e - e_hat||^2 when residual e != 0."""
    sae = _make_identity_sae(4, alpha=1.0, k_aux=2)
    # x and x_hat differ, so residual e = x - x_hat is non-zero
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    x_hat = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    # residual e = [1, 2, 0, 0]
    pre = torch.tensor([[0.0, 0.0, 3.0, 4.0]])
    out = modeling.SparseAutoencoder.Output(h_x=pre, f_x=pre, x_hats=x_hat.unsqueeze(1))
    dead_mask = torch.ones(4, dtype=torch.bool)

    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    # With identity decoder and k_aux=2, selects indices 2,3 with values 3,4
    # e_hat = [0, 0, 3, 4]
    # loss = ||e - e_hat||^2.mean() = ||[1,2,0,0] - [0,0,3,4]||^2 / 4
    #      = (1^2 + 2^2 + 9 + 16) / 4 = 30/4 = 7.5
    expected = (1.0 + 4.0 + 9.0 + 16.0) / 4
    assert torch.allclose(loss, torch.tensor(expected))


def test_auxk_detaches_residual_from_live_path():
    """Verify gradients don't flow back through x_hat to live latent computation."""
    sae = _make_identity_sae(4, alpha=1.0, k_aux=2)
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    # x_hat with grad tracking to verify it gets detached
    x_hat = torch.tensor([[0.5, 0.5, 0.0, 0.0]], requires_grad=True)
    pre = torch.tensor([[0.0, 0.0, 3.0, 4.0]], requires_grad=True)
    out = modeling.SparseAutoencoder.Output(h_x=pre, f_x=pre, x_hats=x_hat.unsqueeze(1))
    dead_mask = torch.ones(4, dtype=torch.bool)

    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    loss.backward()

    # x_hat should have no gradient because residual is detached
    assert x_hat.grad is None or torch.allclose(x_hat.grad, torch.zeros_like(x_hat))
    # pre should have gradient because it flows through aux reconstruction
    assert pre.grad is not None
    assert pre.grad.abs().sum() > 0


def test_auxk_uses_preacts_not_postacts():
    """With realistic TopK, verify AuxK uses pre-activation values."""
    # SAE with top_k=2, so only 2 latents survive activation
    cfg = modeling.SparseAutoencoderConfig(
        d_model=4,
        d_sae=4,
        normalize_w_dec=False,
        remove_parallel_grads=False,
        activation=modeling.TopK(top_k=2, aux=modeling.AuxK(k_aux=2, alpha=1.0)),
    )
    sae = modeling.SparseAutoencoder(cfg)
    with torch.no_grad():
        sae.W_dec.copy_(torch.eye(4))
        sae.W_enc.copy_(torch.eye(4))
        sae.b_dec.zero_()
        sae.b_enc.zero_()

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    enc = sae.encode(x)
    # TopK(2) keeps indices 2,3 (values 3,4), zeros out indices 0,1
    assert enc.f_x[0, 0] == 0.0  # zeroed by TopK
    assert enc.f_x[0, 1] == 0.0  # zeroed by TopK
    assert enc.f_x[0, 2] == 3.0  # kept
    assert enc.f_x[0, 3] == 4.0  # kept
    # But pre-activations should have original values
    assert enc.h_x[0, 0] == 1.0
    assert enc.h_x[0, 1] == 2.0

    # Now test AuxK with dead_mask marking indices 0,1 as dead
    x_hat = sae.decode(enc.f_x)
    out = modeling.SparseAutoencoder.Output(h_x=enc.h_x, f_x=enc.f_x, x_hats=x_hat)
    dead_mask = torch.tensor([True, True, False, False])

    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    # AuxK should use pre-acts [1,2] for dead latents, not post-acts [0,0]
    # With k_aux=2, selects both dead latents with pre-act values 1 and 2
    # e_hat from dead latents = [1, 2, 0, 0]
    # x_hat from live latents = [0, 0, 3, 4]
    # residual e = x - x_hat = [1, 2, 0, 0]
    # loss = ||e - e_hat||^2.mean() = ||[1,2,0,0] - [1,2,0,0]||^2 / 4 = 0
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)


def test_auxk_batch_aggregation():
    """Test loss is correctly aggregated across batch dimension."""
    sae = _make_identity_sae(4, alpha=1.0, k_aux=1)
    # Two examples with different top dead latents
    x = torch.zeros(2, 4)
    pre = torch.tensor([
        [0.0, 0.0, 3.0, 1.0],  # top dead: index 2 (value 3)
        [0.0, 0.0, 1.0, 5.0],  # top dead: index 3 (value 5)
    ])
    out = modeling.SparseAutoencoder.Output(
        h_x=pre, f_x=pre, x_hats=torch.zeros(2, 1, 4)
    )
    dead_mask = torch.ones(4, dtype=torch.bool)

    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)
    # Example 0: e_hat = [0,0,3,0], loss = 9/4
    # Example 1: e_hat = [0,0,0,5], loss = 25/4
    # Mean over batch and d_model: (9 + 25) / (2 * 4) = 34/8 = 4.25
    expected = (9.0 + 25.0) / 8
    assert torch.allclose(loss, torch.tensor(expected))


def test_auxk_eval_mode_returns_zero_with_none_mask():
    """In eval mode, AuxK returns zero loss when dead_mask is None."""
    sae = _make_identity_sae(4, alpha=1.0, k_aux=2)
    sae.eval()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pre = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = modeling.SparseAutoencoder.Output(
        h_x=pre, f_x=pre, x_hats=torch.zeros(1, 1, 4)
    )

    loss = sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=None)
    assert loss.item() == 0.0


def test_auxk_eval_mode_asserts_on_dead_mask():
    """In eval mode, AuxK asserts if dead_mask is provided."""
    import pytest

    sae = _make_identity_sae(4, alpha=1.0, k_aux=2)
    sae.eval()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pre = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = modeling.SparseAutoencoder.Output(
        h_x=pre, f_x=pre, x_hats=torch.zeros(1, 1, 4)
    )
    dead_mask = torch.ones(4, dtype=torch.bool)

    with pytest.raises(AssertionError, match="must be None during eval"):
        sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=dead_mask)


def test_auxk_train_mode_asserts_on_none_mask():
    """In train mode, AuxK asserts if dead_mask is None."""
    import pytest

    sae = _make_identity_sae(4, alpha=1.0, k_aux=2)
    sae.train()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pre = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = modeling.SparseAutoencoder.Output(
        h_x=pre, f_x=pre, x_hats=torch.zeros(1, 1, 4)
    )

    with pytest.raises(AssertionError, match="required during training"):
        sae.cfg.activation.aux.loss(sae=sae, x=x, out=out, dead_mask=None)


def test_n_dead_tracks_dead_latents():
    """Test that n_dead correctly counts latents that haven't fired in threshold tokens."""
    # Use small threshold for easy testing
    threshold = 10
    obj_cfg = objectives.Matryoshka(n_prefixes=1, dead_threshold_tokens=threshold)
    objective = objectives.get_objective(obj_cfg)

    # Create SAE with top_k=2 so only 2 latents fire per example
    cfg = modeling.SparseAutoencoderConfig(
        d_model=4,
        d_sae=4,
        normalize_w_dec=False,
        remove_parallel_grads=False,
        activation=modeling.TopK(top_k=2, aux=modeling.AuxK()),
    )
    sae = modeling.SparseAutoencoder(cfg)
    with torch.no_grad():
        sae.W_dec.copy_(torch.eye(4))
        sae.W_enc.copy_(torch.eye(4))
        sae.b_dec.zero_()
        sae.b_enc.zero_()
    sae.train()

    # Input where latents 0,1 have highest pre-activations, so TopK selects them
    # Latents 2,3 will never fire
    x = torch.tensor([[2.0, 1.0, 0.0, -1.0], [2.0, 1.0, 0.0, -1.0]])
    batch_size = x.shape[0]

    # First forward: no latents are dead yet (tracker starts at 0)
    loss, _ = objective(sae, x)
    assert loss.n_dead == 0

    # Run enough batches to exceed threshold for inactive latents
    n_batches = (threshold // batch_size) + 1
    for _ in range(n_batches):
        loss, _ = objective(sae, x)

    # After threshold tokens, latents 2 and 3 should be dead
    assert loss.n_dead == 2

    # Now make latent 2 the highest - it should fire and no longer be dead
    x_activate_2 = torch.tensor([[0.0, 1.0, 3.0, -1.0], [0.0, 1.0, 3.0, -1.0]])
    loss, _ = objective(sae, x_activate_2)
    assert loss.n_dead == 1  # Only latent 3 is dead now
