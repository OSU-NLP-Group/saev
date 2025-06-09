def pytest_addoption(parser):
    parser.addoption(
        "--shards",
        action="store",
        default=None,
        help="Root directory with activation shards",
    )
