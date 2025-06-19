import logging
import typing

import beartype
import tyro

from . import config

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger("saev")


@beartype.beartype
def visuals(cfg: typing.Annotated[config.Visuals, tyro.conf.arg(name="")]):
    """
    Save maximally activating images for each SAE latent.

    Args:
        cfg: Config
    """
    from . import visuals

    visuals.main(cfg)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"visuals": visuals})
    logger.info("Done.")
