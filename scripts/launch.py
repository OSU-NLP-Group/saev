"""
Launcher script for activation computation with submitit support.
"""

import inference
import shards
import tyro.extras

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "inference": inference.main,
        "shards": shards.main,
    })
