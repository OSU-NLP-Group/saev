import torch

from saev import config, training


def test_split_cfgs_on_single_key():
    cfgs = [config.Train(n_workers=12), config.Train(n_workers=16)]
    expected = [[config.Train(n_workers=12)], [config.Train(n_workers=16)]]

    actual = training.split_cfgs(cfgs)

    assert actual == expected


def test_split_cfgs_on_single_key_with_multiple_per_key():
    cfgs = [
        config.Train(n_patches=12),
        config.Train(n_patches=16),
        config.Train(n_patches=16),
        config.Train(n_patches=16),
    ]
    expected = [
        [config.Train(n_patches=12)],
        [
            config.Train(n_patches=16),
            config.Train(n_patches=16),
            config.Train(n_patches=16),
        ],
    ]

    actual = training.split_cfgs(cfgs)

    assert actual == expected


def test_split_cfgs_on_multiple_keys_with_multiple_per_key():
    cfgs = [
        config.Train(n_patches=12, track=False),
        config.Train(n_patches=12, track=True),
        config.Train(n_patches=16, track=True),
        config.Train(n_patches=16, track=True),
        config.Train(n_patches=16, track=False),
    ]
    expected = [
        [config.Train(n_patches=12, track=False)],
        [config.Train(n_patches=12, track=True)],
        [
            config.Train(n_patches=16, track=True),
            config.Train(n_patches=16, track=True),
        ],
        [config.Train(n_patches=16, track=False)],
    ]

    actual = training.split_cfgs(cfgs)

    assert actual == expected


def test_split_cfgs_no_bad_keys():
    cfgs = [
        config.Train(n_patches=12, objective=config.Vanilla(sparsity_coeff=1e-4)),
        config.Train(n_patches=12, objective=config.Vanilla(sparsity_coeff=2e-4)),
        config.Train(n_patches=12, objective=config.Vanilla(sparsity_coeff=3e-4)),
        config.Train(n_patches=12, objective=config.Vanilla(sparsity_coeff=4e-4)),
        config.Train(n_patches=12, objective=config.Vanilla(sparsity_coeff=5e-4)),
    ]
    expected = [
        [
            config.Train(n_patches=12, objective=config.Vanilla(sparsity_coeff=1e-4)),
            config.Train(n_patches=12, objective=config.Vanilla(sparsity_coeff=2e-4)),
            config.Train(n_patches=12, objective=config.Vanilla(sparsity_coeff=3e-4)),
            config.Train(n_patches=12, objective=config.Vanilla(sparsity_coeff=4e-4)),
            config.Train(n_patches=12, objective=config.Vanilla(sparsity_coeff=5e-4)),
        ]
    ]

    actual = training.split_cfgs(cfgs)

    assert actual == expected


class DummyDS(torch.utils.data.Dataset):
    def __init__(self, n, d):
        self.x = torch.randn(n, d)

    def __getitem__(self, i):
        return dict(act=self.x[i])

    def __len__(self):
        return len(self.x)


def test_one_training_step(monkeypatch):
    cfg = config.Train(
        track=False, sae_batch_size=8, data=config.DataLoad(), n_patches=64
    )

    # monkey-patch Dataset/loader used in activations module
    from . import activations

    monkeypatch.setattr(activations, "Dataset", lambda _: DummyDS(32, cfg.sae.d_vit))
    monkeypatch.setattr(activations, "get_dataloader", lambda *a, **k: None)  # not used

    # run a single loop
    from . import training

    ids = training.main([cfg])  # should not raise
    assert len(ids) == 1


def test_one_training_step_matryoshka(monkeypatch):
    """A minimal end-to-end training-loop smoke test for the Matryoshka objective."""

    # configuration that uses Matryoshka
    cfg = config.Train(
        track=False,
        sae_batch_size=8,
        n_patches=64,  # make the run fast.
        data=config.DataLoad(),
        objective=config.Matryoshka(),
    )

    # stub out expensive I/O
    from . import activations

    monkeypatch.setattr(activations, "Dataset", lambda *_: DummyDS(32, cfg.sae.d_vit))
    monkeypatch.setattr(activations, "get_dataloader", lambda *_1, **_2: None)

    # run one training job
    from saev import training

    ids = training.main([cfg])
    assert len(ids) == 1
