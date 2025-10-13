"""
Launcher script for activation computation with submitit support.
"""

import tyro.extras

import saev.framework.inference
import saev.framework.shards

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "inference": saev.framework.inference.main,
        "shards": saev.framework.shards.main,
    })
