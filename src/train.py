"""
Capstone elements: **Model Design/Building**, **Model Training**, and
**Model Optimization** (stub).

This module is a CLI scaffold for training custom 3D segmentation models
on Lung1 (e.g., MONAI 3D U-Net or V-Net). It loads `configs/default.yaml`
to discover which models and hyperparameters to use, and is designed to be
extended with dataloaders, a training loop, and optimizer/scheduler logic
in line with the nnU-Net baseline.
"""

from __future__ import annotations

import argparse
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Train models (bootstrap stub).")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    print("Training is bootstrapped.")
    print("Next: implement dataloaders, MONAI 3D U-Net/V-Net, training loop.")
    print(f"Configured models: {cfg.get('train', {}).get('models', [])}")


if __name__ == "__main__":
    main()


