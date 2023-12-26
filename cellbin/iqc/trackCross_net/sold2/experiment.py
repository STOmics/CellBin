"""
Main file to launch training and testing experiments.
"""

import yaml
import os
import torch

# Pytorch configurations
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


def load_config(config_path):
    """ Load configurations from a given yaml file. """
    # Check file exists
    if not os.path.exists(config_path):
        print(config_path)
        raise ValueError("[Error] The provided config path is not valid.")

    # Load the configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config

