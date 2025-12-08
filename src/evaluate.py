from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib  # type: ignore
import numpy as np
import yaml

from src.utils.metrics import dice_coefficient, jaccard_index


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
        # load arrays
        pred = load_nii(pred_p)
        label = load_nii(label_p)
        # binarize (safety in case of probabilistic outputs)
        # Binarize predictions in case of soft/prob outputs (safety)
        pred_bin = pred >= 0.5
        label_bin = label >= 0.5

        # metrics
        dsc = dice_coefficient(label_bin, pred_bin)
        jac = jaccard_index(label_bin, pred_bin)

        # volumes (mm^3 and mL) from label header spacing
        label_img = nib.load(str(label_p))  # type: ignore
        zooms = label_img.header.get_zooms()[:3]  # type: ignore[call-arg]
        # guard against malformed headers
        if len(zooms) != 3 or any(z <= 0 for z in zooms):
            voxel_vol_mm3 = float(1.0)
        else:
            voxel_vol_mm3 = float(np.prod(zooms))

        label_voxels = int(label_bin.sum())
        pred_voxels = int(pred_bin.sum())
        label_vol_mm3 = float(label_voxels * voxel_vol_mm3)
        pred_vol_mm3 = float(pred_voxels * voxel_vol_mm3)
        label_vol_ml = label_vol_mm3 / 1000.0
        pred_vol_ml = pred_vol_mm3 / 1000.0
        vol_abs_diff_ml = abs(pred_vol_ml - label_vol_ml)
        vol_rel_err_pct = float(np.nan)
        if label_vol_ml > 0:
            vol_rel_err_pct = 100.0 * (pred_vol_ml - label_vol_ml) / label_vol_ml

        rows.append(
            {
                "case": case,
                "dice": float(dsc),
                "jaccard": float(jac),
                "label_voxels": float(label_voxels),
                "pred_voxels": float(pred_voxels),
                "label_volume_ml": float(label_vol_ml),
                "pred_volume_ml": float(pred_vol_ml),
                "volume_abs_diff_ml": float(vol_abs_diff_ml),
                "volume_rel_err_pct": float(vol_rel_err_pct),
            }
        )
        print(
            f"{case}: Dice={dsc:.4f} | Jaccard={jac:.4f} | "
            f"GT_vol={label_vol_ml:.2f} mL, Pred_vol={pred_vol_ml:.2f} mL, "
            f"|Δ|={vol_abs_diff_ml:.2f} mL"
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "dice",
                "jaccard",
                "label_voxels",
                "pred_voxels",
                "label_volume_ml",
                "pred_volume_ml",
                "volume_abs_diff_ml",
                "volume_rel_err_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    # summary
    mean_dice = float(np.mean([r["dice"] for r in rows]))
    mean_jacc = float(np.mean([r["jaccard"] for r in rows]))
    mae_vol_ml = float(np.mean([r["volume_abs_diff_ml"] for r in rows]))
    # exclude NaNs for rel error
    rel_errs = [r["volume_rel_err_pct"] for r in rows if not np.isnan(r["volume_rel_err_pct"])]
    mape_vol_pct = float(np.mean(np.abs(rel_errs))) if rel_errs else float("nan")
    print(
        f"Summary over {len(rows)} cases -> "
        f"Mean Dice={mean_dice:.4f}, Mean Jaccard={mean_jacc:.4f}, "
        f"MAE Volume={mae_vol_ml:.2f} mL, MAPE Volume={mape_vol_pct:.2f}%"
    )
    return {
        "mean_dice": mean_dice,
        "mean_jaccard": mean_jacc,
        "mae_volume_ml": mae_vol_ml,
        "mape_volume_pct": mape_vol_pct,
    }


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

