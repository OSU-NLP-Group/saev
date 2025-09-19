"""
Launcher script for activation computation with submitit support.
"""

import tyro.extras

import saev.scripts.dump

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"dump": saev.scripts.dump.main})
