from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib  # type: ignore
import numpy as np
import yaml

from src.utils.metrics import dice_coefficient


def derive_case_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate models and/or nnU-Net predictions.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--nnunet-preds",
        type=str,
        default=None,
        help="Optional path to nnU-Net prediction folder containing *.nii.gz",
    )
    p.add_argument(
        "--labels-dir",
        type=str,
        default="data/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/labelsTr",
        help="Labels directory to compare predictions against",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/processed/nnunet_fold0_metrics.csv",
        help="Output CSV for summary metrics",
    )
    return p.parse_args()


def load_nii(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).get_fdata())


def collect_pairs(pred_dir: Path, labels_dir: Path) -> List[Tuple[str, Path, Path]]:
    pairs: List[Tuple[str, Path, Path]] = []
    for pred in sorted(pred_dir.glob("*.nii.gz")):
        case = derive_case_id(pred)  # handle .nii.gz correctly
        label = labels_dir / f"{case}.nii.gz"
        if label.exists():
            pairs.append((case, pred, label))
    return pairs


def eval_nnunet_preds(pred_dir: Path, labels_dir: Path, out_csv: Path) -> Dict[str, float]:
    pairs = collect_pairs(pred_dir, labels_dir)
    if not pairs:
        print(f"No overlapping cases between {pred_dir} and {labels_dir}.")
        return {}
    rows: List[Dict[str, float]] = []
    for case, pred_p, label_p in pairs:
        pred = load_nii(pred_p)
        label = load_nii(label_p)
        # Binarize predictions in case of soft/prob outputs (safety)
        pred_bin = pred >= 0.5
        label_bin = label >= 0.5
        dsc = dice_coefficient(label_bin, pred_bin)
        rows.append({"case": case, "dice": float(dsc)})
        print(f"{case}: Dice={dsc:.4f}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "dice"])
        writer.writeheader()
        writer.writerows(rows)
    mean_dice = float(np.mean([r["dice"] for r in rows]))
    print(f"Mean Dice over {len(rows)} cases: {mean_dice:.4f}")
    return {"mean_dice": mean_dice}


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    reports_dir = Path(cfg.get("paths", {}).get("reports", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    # nnU-Net bridge (optional)
    if args.nnunet_preds:
        pred_dir = Path(args.nnunet_preds)
        labels_dir = Path(args.labels_dir)
        out_csv = Path(args.out)
        eval_nnunet_preds(pred_dir, labels_dir, out_csv)
    else:
        print("No --nnunet-preds provided. Skipping nnU-Net evaluation bridge.")
        print("You can run: make nnunet:evaluate or pass --nnunet-preds explicitly.")


if __name__ == "__main__":
    main()

