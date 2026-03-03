"""Unit tests for the MiniBatchSemiNMF baseline."""

import itertools

import torch
import torch.nn.functional as F
from tdiscovery.baselines import MiniBatchSemiNMF, TrainConfig, dump, load

import saev.data
import saev.disk


def _generate_semi_nmf_data(
    *,
    n_samples: int = 128,
    n_features: int = 6,
    n_concepts: int = 4,
    noise_std: float = 0.01,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    z = torch.rand(n_samples, n_concepts)
    D = torch.randn(n_concepts, n_features)
    noise = noise_std * torch.randn(n_samples, n_features)
    acts = z @ D + noise
    return acts.to(torch.float32), z.to(torch.float32), D.to(torch.float32)


def _fit_model(
    acts: torch.Tensor,
    *,
    n_concepts: int,
    batch_size: int,
    epochs: int,
    seed: int,
    z_iters: int = 4,
    encode_iters: int = 20,
    ridge: float = 1e-6,
    eps: float = 1e-8,
    forget_factor: float = 0.7,
    d_update_every: int = 1,
) -> MiniBatchSemiNMF:
    torch.manual_seed(seed)
    model = MiniBatchSemiNMF(
        n_concepts=n_concepts,
        device="cpu",
        z_iters=z_iters,
        encode_iters=encode_iters,
        batch_size=batch_size,
        ridge=ridge,
        eps=eps,
        forget_factor=forget_factor,
        d_update_every=d_update_every,
    )
    for _ in range(epochs):
        perm = torch.randperm(acts.shape[0])
        for batch in acts[perm].split(batch_size):
            model.partial_fit(batch)
    return model


def _match_components(similarity: torch.Tensor) -> tuple[int, ...]:
    best_score = float("-inf")
    best_perm: tuple[int, ...] | None = None
    rows, cols = similarity.shape
    assert rows == cols, "Similarity must be square"
    indices = range(cols)
    for perm in itertools.permutations(indices, rows):
        diag = similarity[torch.arange(rows), torch.tensor(perm)]
        score = float(diag.sum().item())
        if score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None
    return best_perm


def test_minibatch_semi_nmf_basic_fit_and_transform_nonneg() -> None:
    acts, _, _ = _generate_semi_nmf_data()
    n_concepts = 4
    epochs = 3
    model = _fit_model(
        acts,
        n_concepts=n_concepts,
        batch_size=32,
        epochs=epochs,
        seed=0,
        z_iters=4,
        encode_iters=20,
        d_update_every=1,
    )

    assert model.D_ is not None
    assert model.D_.shape == (n_concepts, acts.shape[1])
    assert model.n_features_in_ == acts.shape[1]
    assert model.n_samples_seen_ == acts.shape[0] * epochs

    codes = model.transform(acts)
    assert codes.shape == (acts.shape[0], n_concepts)
    assert torch.all(codes >= 0)


def test_minibatch_semi_nmf_round_trip(tmp_path) -> None:
    train_shards = tmp_path / "train-shards"
    val_shards = tmp_path / "val-shards"
    train_shards.mkdir(parents=True, exist_ok=True)
    val_shards.mkdir(parents=True, exist_ok=True)
    runs_root = tmp_path / "saev" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        method="semi-nmf",
        k=3,
        device="cpu",
        runs_root=runs_root,
        train_data=saev.data.ShuffledConfig(shards=train_shards),
        val_data=saev.data.ShuffledConfig(shards=val_shards),
        z_iters=7,
        encode_iters=11,
        ridge=1e-5,
        eps=1e-6,
        forget_factor=0.5,
        d_update_every=2,
    )
    run = saev.disk.Run.new(
        "semi-nmf-round-trip",
        train_shards_dir=train_shards,
        val_shards_dir=val_shards,
        runs_root=cfg.runs_root,
    )

    model = MiniBatchSemiNMF(
        n_concepts=cfg.k,
        device="cpu",
        z_iters=cfg.z_iters,
        encode_iters=cfg.encode_iters,
        batch_size=cfg.train_data.batch_size,
        ridge=cfg.ridge,
        eps=cfg.eps,
        forget_factor=cfg.forget_factor,
        d_update_every=cfg.d_update_every,
    )
    model.D_ = torch.randn(cfg.k, 5)
    model.n_features_in_ = 5
    model.n_samples_seen_ = 128
    model.n_steps_ = 7

    dump(run, cfg, model=model)
    loaded = load(run, device="cpu")

    assert isinstance(loaded, MiniBatchSemiNMF)
    assert loaded.D_ is not None
    assert torch.allclose(loaded.D_, model.D_)
    assert loaded.n_features_in_ == model.n_features_in_
    assert loaded.n_samples_seen_ == model.n_samples_seen_
    assert loaded.n_steps_ == model.n_steps_
    assert loaded.z_iters == cfg.z_iters
    assert loaded.encode_iters == cfg.encode_iters
    assert loaded.ridge == cfg.ridge
    assert loaded.eps == cfg.eps
    assert loaded.forget_factor == cfg.forget_factor
    assert loaded.d_update_every == cfg.d_update_every


def test_minibatch_semi_nmf_reproducible_with_seed() -> None:
    acts, _, _ = _generate_semi_nmf_data(seed=123)
    model_a = _fit_model(
        acts,
        n_concepts=4,
        batch_size=32,
        epochs=2,
        seed=111,
        z_iters=4,
        encode_iters=20,
        d_update_every=1,
    )
    model_b = _fit_model(
        acts,
        n_concepts=4,
        batch_size=32,
        epochs=2,
        seed=111,
        z_iters=4,
        encode_iters=20,
        d_update_every=1,
    )

    assert model_a.D_ is not None
    assert model_b.D_ is not None
    assert torch.allclose(model_a.D_, model_b.D_)

    codes_a = model_a.transform(acts)
    codes_b = model_b.transform(acts)
    assert torch.allclose(codes_a, codes_b)


def test_minibatch_semi_nmf_matches_full_batch() -> None:
    acts, _, _ = _generate_semi_nmf_data(seed=7)
    n_concepts = 4
    full = _fit_model(
        acts,
        n_concepts=n_concepts,
        batch_size=acts.shape[0],
        epochs=4,
        seed=222,
        z_iters=4,
        encode_iters=20,
        d_update_every=1,
    )
    mini = _fit_model(
        acts,
        n_concepts=n_concepts,
        batch_size=32,
        epochs=4,
        seed=222,
        z_iters=4,
        encode_iters=20,
        d_update_every=1,
    )

    assert full.D_ is not None
    assert mini.D_ is not None

    full_norm = F.normalize(full.D_.cpu(), dim=1)
    mini_norm = F.normalize(mini.D_.cpu(), dim=1)
    similarity = torch.abs(full_norm @ mini_norm.T)
    perm = _match_components(similarity)
    aligned = similarity[torch.arange(n_concepts), torch.tensor(perm)]
    assert float(aligned.mean().item()) > 0.7
    assert float(aligned.min().item()) > 0.4
