from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate models (bootstrap stub).")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    reports_dir = Path(cfg.get("paths", {}).get("reports", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    print("Evaluation is bootstrapped.")
    print("Next: implement metric computations, CCC, stratified analysis, and plots.")
    print(f"Reports will be saved under: {reports_dir}")


if __name__ == "__main__":
    main()


