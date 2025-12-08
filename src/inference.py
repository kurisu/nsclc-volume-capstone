"""
Capstone element: **Model Analysis / Inference** (stub).

This CLI is intended to load the best-performing trained model, run
segmentation on new Lung1-like CT volumes, and emit both masks and
volume reports. Together with `src/evaluate.py`, it will support the
end-to-end “predict and analyze volumes” phase of the capstone.
"""

from __future__ import annotations

import argparse
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Model inference (bootstrap stub).")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        _ = yaml.safe_load(f)
    print("Inference is bootstrapped.")
    print("Next: implement loading best model and generating masks + volume report.")


if __name__ == "__main__":
    main()


