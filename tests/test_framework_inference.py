import pytest

import saev.data
import saev.data.datasets
import saev.framework.inference
import saev.framework.train


def test_img_folder_inference():
    """Tests that image folders (non-segmentation) work with inference.py

    1. Save activations to disk (in a temporary location)
    2. Train an SAE for 0 tokens (we don't care about the weights)
    3. Run the inference script.
    """
    with (
        pytest.helpers.tmp_shards_root() as shards_root,
        pytest.helpers.tmp_runs_root() as runs_root,
    ):
        # Save activations to disk
        shards_dir = pytest.helpers.write_shards(
            shards_root, data=saev.data.datasets.FakeImg(n_examples=32)
        )
        md = saev.data.Metadata.load(shards_dir)

        # Train for 0 tokens.
        train_cfg = saev.framework.train.Config(
            train_data=saev.data.ShuffledConfig(shards=shards_dir, layer=0),
            val_data=saev.data.ShuffledConfig(shards=shards_dir, layer=0),
            n_train=0,
            n_val=0,
            runs_root=runs_root,
            device="cpu",
            sae=saev.nn.SparseAutoencoderConfig(
                d_model=md.d_model, d_sae=md.d_model * 8
            ),
        )
        (run_id,) = saev.framework.train.worker_fn([train_cfg])

        inference_cfg = saev.framework.inference.Config(
            run=runs_root / run_id,
            data=saev.data.OrderedConfig(shards=shards_dir, layer=0),
            device="cpu",
        )
        saev.framework.inference.worker_fn(inference_cfg)

        run = saev.disk.Run(runs_root / run_id)
        fpaths = saev.framework.inference.Filepaths.from_run(run, md)

        # Assert all files exist and have non-zero size
        for fpath in fpaths:
            assert fpath.exists()
            assert fpath.stat().st_size > 0
