"""
Launcher script for activation computation with submitit support.
"""

import tdiscovery.metrics
import tdiscovery.probe1d
import tdiscovery.visuals
import tyro.extras

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "visuals": tdiscovery.visuals.cli,
        "probe1d": tdiscovery.probe1d.cli,
        "metrics": tdiscovery.metrics.cli,
    })
