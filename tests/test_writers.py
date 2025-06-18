"""
On OSC, with the fish shell:

for shards in /fs/scratch/PAS2136/samuelstevens/cache/saev/*; uv run pytest tests/test_writers.py --shards $shards; end
"""

import glob
import os

import pytest
from hypothesis import given
from hypothesis import strategies as st

from saev.data.writers import IndexLookup, Metadata


@pytest.fixture(scope="session")
def shard_root(pytestconfig):
    p = pytestconfig.getoption("--shards")
    if p is None:
        pytest.skip("--shards not supplied")
    return p


def test_metadata_n_shards_matches_disk(shard_root):
    """Test that Metadata.n_shards matches the actual number of shards on disk.
    Shards are on disk with this sort of file structure:
    $ ls
    acts000000.bin  acts000005.bin  acts000010.bin  acts000015.bin  acts000020.bin
    acts000001.bin  acts000006.bin  acts000011.bin  acts000016.bin  acts000021.bin
    acts000002.bin  acts000007.bin  acts000012.bin  acts000017.bin  acts000022.bin
    acts000003.bin  acts000008.bin  acts000013.bin  acts000018.bin  acts000023.bin
    acts000004.bin  acts000009.bin  acts000014.bin  acts000019.bin  metadata.json
    So here, the number of shards would be 24.
    """
    # Load metadata from the directory
    metadata_path = os.path.join(shard_root, "metadata.json")
    assert os.path.exists(metadata_path), f"Metadata file not found at {metadata_path}"

    metadata = Metadata.load(metadata_path)

    # Count actual shards on disk
    shard_pattern = os.path.join(shard_root, "acts*.bin")
    actual_shards = len(glob.glob(shard_pattern))

    # Verify the calculated n_shards matches the actual count
    assert metadata.n_shards == actual_shards


@st.composite
def metadatas(draw) -> Metadata:
    return Metadata(
        vit_family="family",
        vit_ckpt="ckpt",
        layers=tuple(
            sorted(
                draw(
                    st.sets(
                        st.integers(min_value=0, max_value=24), min_size=1, max_size=24
                    )
                )
            )
        ),
        n_patches_per_img=draw(st.integers(min_value=1, max_value=512)),
        cls_token=draw(st.booleans()),
        d_vit=512,
        seed=0,
        n_imgs=draw(st.integers(min_value=1, max_value=10_000_000)),
        n_patches_per_shard=draw(st.integers(min_value=1, max_value=200_000_000)),
        data="test",
    )


@given(metadatas())
def test_api_surfaces(metadata):
    il = IndexLookup(metadata)
    assert hasattr(il, "map")
    assert hasattr(il, "length")


@given(st.data())
def test_roundtrip(data):
    metadata = data.draw(metadatas())
    patches = data.draw(st.sampled_from(["cls", "all", "patches"]))
    layer = data.draw(st.sampled_from([0, "all"]))
    il = IndexLookup(metadata)

    length = il.length(patches, layer)
    i = data.draw(st.integers(min_value=0, max_value=length - 1))

    sh, i_in_sh, g_img, g_patch = il.map(i, patches, layer)

    # Basic invariants
    assert 0 <= sh < metadata.n_shards
    assert 0 <= i_in_sh < metadata.n_patches_per_shard
    assert 0 <= g_img < metadata.n_imgs


@given(metadata=metadatas())
def test_negative_index_raises(metadata):
    il = IndexLookup(metadata)
    with pytest.raises(IndexError):
        il.map(-1, "cls", 0)


@given(metadata=metadatas())
def test_index_equal_length_raises(metadata):
    il = IndexLookup(metadata)
    length = il.length("cls", 0)
    with pytest.raises(IndexError):
        il.map(length, "cls", 0)


@given(metadata=metadatas())
def test_invalid_mode_and_layer(metadata):
    il = IndexLookup(metadata)
    with pytest.raises(AssertionError):
        il.length("foo", 0)
    with pytest.raises(AssertionError):
        il.map(0, "cls", "bogus")


def test_singleton_dataset():
    meta = Metadata(
        n_imgs=1,
        n_patches_per_img=1,
        layers=(0,),
        n_patches_per_shard=1,
        vit_family="family",
        vit_ckpt="ckpt",
        cls_token=True,
        d_vit=512,
        seed=0,
        data="test",
    )
    il = IndexLookup(meta)
    assert il.length("cls", 0) == 1
    out = il.map(0, "cls", 0)
    assert out[0] == 0  # shard
    assert out[1] == 0  # img in shard


def test_second_img():
    meta = Metadata(
        n_imgs=10_000,
        n_patches_per_img=196,
        layers=(-1,),
        n_patches_per_shard=200_000_000,
        vit_family="family",
        vit_ckpt="ckpt",
        cls_token=False,
        d_vit=512,
        seed=0,
        data="test",
    )
    il = IndexLookup(meta)
    assert il.length("patches", 0) == 1_960_000
    sh, i_in_sh, g_img, g_patch = il.map(196, "patches", 0)
    assert sh == 0
    assert i_in_sh == 1
    assert g_img == 1
    assert g_patch == 196
