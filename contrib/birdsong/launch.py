"""
Launcher script.
"""

import birdsong.visuals
import tyro.extras

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "visuals": birdsong.visuals.cli,
    })
