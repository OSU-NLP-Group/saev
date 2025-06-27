def pytest_addoption(parser):
    parser.addoption(
        "--cub-root",
        action="store",
        default=None,
        help="CUB 200 ImageFolder directory with train/, test/ and metadata/.",
    )
