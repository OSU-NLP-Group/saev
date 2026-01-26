import base64
import os
import pathlib
import pickle

import pytest

from saev.data import datasets, shards


@pytest.fixture
def make_shards_dir(tmp_path: pathlib.Path):
    def _make(
        n_shards: int = 3,
        *,
        empty_shards: list[int] | None = None,
        missing_shards: list[int] | None = None,
        unreadable_shards: list[int] | None = None,
        dir_shards: list[int] | None = None,
    ) -> pathlib.Path:
        shards_root_dpath = tmp_path / "saev" / "shards"
        shards_root_dpath.mkdir(parents=True, exist_ok=True)

        data_cfg = datasets.FakeImg(n_examples=n_shards)
        md = shards.Metadata(
            family="clip",
            ckpt="hf-hub:hf-internal-testing/tiny-open-clip-model",
            layers=(0,),
            content_tokens_per_example=1,
            cls_token=True,
            d_model=4,
            n_examples=n_shards,
            max_tokens_per_shard=8,
            data=base64.b64encode(pickle.dumps(data_cfg)).decode("utf8"),
            dataset=data_cfg.root,
        )
        md.dump(shards_root_dpath)
        shards_dir_dpath = shards_root_dpath / md.hash

        empty_shards_set = set(empty_shards or [])
        missing_shards_set = set(missing_shards or [])
        unreadable_shards_set = set(unreadable_shards or [])
        dir_shards_set = set(dir_shards or [])

        shard_entries: list[shards.Shard] = []
        for shard_idx in range(n_shards):
            acts_fname = f"acts{shard_idx:06}.bin"
            shard_entries.append(shards.Shard(name=acts_fname, n_examples=1))
            acts_fpath = shards_dir_dpath / acts_fname

            if shard_idx in missing_shards_set:
                continue
            if shard_idx in dir_shards_set:
                acts_fpath.mkdir()
                continue

            if shard_idx in empty_shards_set:
                acts_fpath.write_bytes(b"")
            else:
                acts_fpath.write_bytes(b"data")

            if shard_idx in unreadable_shards_set and acts_fpath.is_file():
                acts_fpath.chmod(0)

        shards.ShardInfo(shard_entries).dump(shards_dir_dpath)
        return shards_dir_dpath

    return _make


def _can_make_unreadable() -> bool:
    if os.name == "nt":
        return False
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return False
    return True


def test_validate_all_shards_present_and_nonempty(make_shards_dir):
    shards_dir_dpath = make_shards_dir()
    shard_info = shards.ShardInfo.load(shards_dir_dpath)
    shard_info.validate(shards_dir_dpath)


def test_validate_missing_shard_file(make_shards_dir):
    shards_dir_dpath = make_shards_dir(missing_shards=[1])
    shard_info = shards.ShardInfo.load(shards_dir_dpath)

    with pytest.raises(FileNotFoundError) as excinfo:
        shard_info.validate(shards_dir_dpath)

    missing_fpath = (shards_dir_dpath / "acts000001.bin").resolve()
    msg = str(excinfo.value)
    assert "Missing files (1):" in msg
    assert str(missing_fpath) in msg
    assert missing_fpath.is_absolute()


def test_validate_empty_shard_file(make_shards_dir):
    shards_dir_dpath = make_shards_dir(empty_shards=[0])
    shard_info = shards.ShardInfo.load(shards_dir_dpath)

    with pytest.raises(FileNotFoundError) as excinfo:
        shard_info.validate(shards_dir_dpath)

    empty_fpath = (shards_dir_dpath / "acts000000.bin").resolve()
    msg = str(excinfo.value)
    assert "Empty files (1):" in msg
    assert str(empty_fpath) in msg
    assert empty_fpath.is_absolute()


def test_validate_multiple_problems(make_shards_dir):
    shards_dir_dpath = make_shards_dir(missing_shards=[1], empty_shards=[2])
    shard_info = shards.ShardInfo.load(shards_dir_dpath)

    with pytest.raises(FileNotFoundError) as excinfo:
        shard_info.validate(shards_dir_dpath)

    missing_fpath = (shards_dir_dpath / "acts000001.bin").resolve()
    empty_fpath = (shards_dir_dpath / "acts000002.bin").resolve()
    msg = str(excinfo.value)
    assert "Missing files (1):" in msg
    assert "Empty files (1):" in msg
    assert str(missing_fpath) in msg
    assert str(empty_fpath) in msg
    assert msg.index("Missing files") < msg.index("Empty files")


def test_validate_no_shards(make_shards_dir):
    shards_dir_dpath = make_shards_dir(n_shards=0)
    shard_info = shards.ShardInfo.load(shards_dir_dpath)
    shard_info.validate(shards_dir_dpath)


def test_validate_unreadable_file(make_shards_dir):
    if not _can_make_unreadable():
        pytest.skip(
            "chmod 0 does not reliably block access on this platform or as root"
        )

    shards_dir_dpath = make_shards_dir(unreadable_shards=[0])
    shard_info = shards.ShardInfo.load(shards_dir_dpath)

    with pytest.raises(FileNotFoundError) as excinfo:
        shard_info.validate(shards_dir_dpath)

    unreadable_fpath = (shards_dir_dpath / "acts000000.bin").resolve()
    msg = str(excinfo.value)
    assert "Unreadable files (1):" in msg
    assert str(unreadable_fpath) in msg


def test_validate_directory_instead_of_file(make_shards_dir):
    shards_dir_dpath = make_shards_dir(dir_shards=[0])
    shard_info = shards.ShardInfo.load(shards_dir_dpath)

    with pytest.raises(FileNotFoundError) as excinfo:
        shard_info.validate(shards_dir_dpath)

    dir_fpath = (shards_dir_dpath / "acts000000.bin").resolve()
    msg = str(excinfo.value)
    assert "Not regular files (1):" in msg
    assert str(dir_fpath) in msg


def test_validate_accepts_str_path(make_shards_dir):
    shards_dir_dpath = make_shards_dir()
    shard_info = shards.ShardInfo.load(shards_dir_dpath)
    shard_info.validate(str(shards_dir_dpath))
