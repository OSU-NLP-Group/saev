# We need to find the true total number of shards. They are on disk with this sort of file-structure:
# ascend-login01 /f/s/P/o/a/406094ac63e6de7a592d5ddbaa581e147d51a2596e39732dafe2171cfe15225b  ls
# acts000000.bin  acts000005.bin  acts000010.bin  acts000015.bin  acts000020.bin  acts000025.bin
# acts000001.bin  acts000006.bin  acts000011.bin  acts000016.bin  acts000021.bin  acts000026.bin
# acts000002.bin  acts000007.bin  acts000012.bin  acts000017.bin  acts000022.bin  acts000027.bin
# acts000003.bin  acts000008.bin  acts000013.bin  acts000018.bin  acts000023.bin  acts000028.bin
# acts000004.bin  acts000009.bin  acts000014.bin  acts000019.bin  acts000024.bin  metadata.json
# So here, the number of shards would be 29.

import os
import glob
import pytest
from saev.data.writers import Metadata


@pytest.fixture(scope="session")
def shard_root(pytestconfig):
    p = pytestconfig.getoption("--shards")
    if p is None:
        pytest.skip("--shards not supplied")
    return p


def test_metadata_n_shards_matches_disk(shard_root):
    """Test that Metadata.n_shards matches the actual number of shards on disk."""
    # Load metadata from the directory
    metadata_path = os.path.join(shard_root, "metadata.json")
    assert os.path.exists(metadata_path), f"Metadata file not found at {metadata_path}"
    
    metadata = Metadata.load(metadata_path)
    
    # Count actual shards on disk
    shard_pattern = os.path.join(shard_root, "acts*.bin")
    actual_shards = len(glob.glob(shard_pattern))
    
    # Verify the calculated n_shards matches the actual count
    assert metadata.n_shards == actual_shards, (
        f"Metadata.n_shards ({metadata.n_shards}) doesn't match "
        f"actual shard count ({actual_shards})"
    )
