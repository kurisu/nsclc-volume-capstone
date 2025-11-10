from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Lung1 data (stubs).")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", {})

    # Create target directories if missing
    for key in ["interim", "processed", "metadata", "reports"]:
        if key in paths:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

    print("Data preparation is bootstrapped.")
    print("Next: implement RTSTRUCT→mask conversion and 5-fold split.")


if __name__ == "__main__":
    main()


