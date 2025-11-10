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


