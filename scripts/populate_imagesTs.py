from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Populate imagesTs from imagesTr using test_index.json")
    p.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to DatasetXXX_Name folder containing imagesTr/labelsTr/imagesTs and test_index.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = args.dataset_dir
    images_tr = ds / "imagesTr"
    images_ts = ds / "imagesTs"
    test_index = ds / "test_index.json"
    if not test_index.exists():
        print(f"test_index.json not found at {test_index}; nothing to do.")
        return
    images_ts.mkdir(parents=True, exist_ok=True)
    with test_index.open() as f:
        idx = json.load(f)
    test_ids = idx.get("test_ids", [])
    copied = 0
    for cid in test_ids:
        src = images_tr / f"{cid}_0000.nii.gz"
        dst = images_ts / f"{cid}_0000.nii.gz"
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"WARNING: missing source {src}")
    print(f"Copied {copied} test cases into imagesTs.")


if __name__ == "__main__":
    main()


