def pytest_addoption(parser):
    parser.addoption(
        "--shards",
        action="store",
        default=None,
        help="Root directory with activation shards",
    )
    parser.addoption(
        "--ckpt-path",
        action="store",
        default=None,
        help="Path to a checkpoint file for testing load functionality",
    )
