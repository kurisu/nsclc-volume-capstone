from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create nnU-Net dataset.json for CT binary segmentation.")
    p.add_argument("--dataset_dir", type=Path, default=Path("data/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1"))
    p.add_argument("--name", type=str, default="NSCLC_Lung1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images_tr = args.dataset_dir / "imagesTr"
    labels_tr = args.dataset_dir / "labelsTr"
    images_ts = args.dataset_dir / "imagesTs"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    # minimal dataset.json
    ds = {
        "name": args.name,
        "description": "NSCLC Lung1 CT tumor segmentation",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "tensorImageSize": "3D",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": None,  # filled below
        "file_ending": ".nii.gz",
        "training": [],
        "test": [],
    }
    # populate training entries if present
    for img in sorted(images_tr.glob("*_0000.nii.gz")):
        case = img.name.replace("_0000.nii.gz", "")
        lab = labels_tr / f"{case}.nii.gz"
        if lab.exists():
            ds["training"].append({"image": f"./imagesTr/{img.name}", "label": f"./labelsTr/{case}.nii.gz"})
    ds["numTraining"] = len(ds["training"])
    # test list is optional for nnU-Net v2; leave empty or list by path
    with (args.dataset_dir / "dataset.json").open("w") as f:
        json.dump(ds, f, indent=2)
    print(f"Wrote dataset.json with {ds['numTraining']} training entries at {args.dataset_dir / 'dataset.json'}")


if __name__ == "__main__":
    main()


