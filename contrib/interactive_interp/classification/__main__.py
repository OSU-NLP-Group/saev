import logging
import tomllib
import typing

import beartype
import tyro

from . import config

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("contrib.semseg")


@beartype.beartype
def train(
    cfg: typing.Annotated[config.Train, tyro.conf.arg(name="")],
    sweep: str | None = None,
):
    import submitit

    from . import training

    if sweep is not None:
        with open(sweep, "rb") as fd:
            cfgs, errs = config.grid(cfg, tomllib.load(fd))

        if errs:
            for err in errs:
                logger.warning("Error in config: %s", err)
            return

    else:
        cfgs = [cfg]

    logger.info("Running %d training jobs.", len(cfgs))

    executor = submitit.DebugExecutor(folder=cfg.log_to)

    job = executor.submit(training.main, cfgs)
    job.result()


if __name__ == "__main__":
    tyro.cli(train)
