import glob
import os

import pytest

from saev.data.shards import Metadata, ShardInfo


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
    acts000003.bin  acts000008.bin  acts000013.bin  acts000018.bin  metadata.json
    acts000004.bin  acts000009.bin  acts000014.bin  acts000019.bin  shards.json
    So here, the number of shards would be 23.
    """
    metadata = Metadata.load(shard_root)

    # Count actual shards on disk
    shard_pattern = os.path.join(shard_root, "acts*.bin")
    actual_shards = len(glob.glob(shard_pattern))

    # Verify the calculated n_shards matches the actual count
    assert metadata.n_shards == actual_shards


def test_metadata_n_imgs_per_shard_matches_disk(shard_root):
    """Test that Metadata.n_imgs_per_shard matches both our math and shards.json."""
    # Load metadata from the directory
    metadata = Metadata.load(shard_root)
    shard_info = ShardInfo.load(shard_root)

    for shard in shard_info[:-1]:
        assert metadata.n_imgs_per_shard == shard.n_imgs
    assert metadata.n_imgs_per_shard >= shard_info[-1].n_imgs
