"""
Launcher script for activation computation with submitit support.
"""

import tyro.extras
import visuals

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"visuals": visuals.main})
