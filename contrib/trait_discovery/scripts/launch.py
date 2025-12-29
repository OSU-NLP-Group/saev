"""
Launcher script for activation computation with submitit support.
"""

import tdiscovery.baselines
import tdiscovery.classification
import tdiscovery.metrics
import tdiscovery.probe1d
import tdiscovery.visuals
import tyro.extras

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "baseline::train": tdiscovery.baselines.train_cli,
        "baseline::inference": tdiscovery.baselines.inference_cli,
        "cls::train": tdiscovery.classification.train_cli,
        "cls::eval": tdiscovery.classification.eval_cli,
        "metrics": tdiscovery.metrics.cli,
        "probe1d": tdiscovery.probe1d.cli,
        "visuals": tdiscovery.visuals.cli,
    })
