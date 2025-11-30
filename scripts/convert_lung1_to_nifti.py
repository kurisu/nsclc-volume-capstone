from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pydicom
import SimpleITK as sitk
from rt_utils import RTStructBuilder


ROI_NAME_KEYWORDS = (
    "gtv",
    "primary",
    "tumor",
    "tumour",
    "lesion",
)


def find_rtstruct(patient_dir: Path) -> Optional[Path]:
    for root, _, files in os.walk(patient_dir):
        for f in files:
            if not f.lower().endswith(".dcm"):
                continue
            fp = Path(root) / f
            try:
                ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
                if getattr(ds, "Modality", None) == "RTSTRUCT":
                    return fp
            except Exception:
                continue
    return None


def find_best_ct_series(patient_dir: Path) -> Optional[tuple[list[str], str]]:
    """
    Recursively search for the CT series with the most slices under patient_dir.
    Returns (file_names, series_dir) or None.
    """
    reader = sitk.ImageSeriesReader()
    best_files: list[str] = []
    best_dir: Optional[str] = None
    for root, dirs, files in os.walk(patient_dir):
        # Quick check: skip folders with no DICOM-looking files to keep it fast
        has_dcm = any(fn.lower().endswith(".dcm") for fn in files)
        if not has_dcm:
            continue
        try:
            series_ids = reader.GetGDCMSeriesIDs(root)
        except Exception:
            continue
        if not series_ids:
            continue
        for sid in series_ids:
            try:
                fns = reader.GetGDCMSeriesFileNames(root, sid)
            except Exception:
                continue
            # Prefer larger series (more slices)
            if len(fns) > len(best_files):
                best_files = fns
                best_dir = root
    if best_files:
        return best_files, best_dir or str(patient_dir)
    return None


def get_case_id(patient_dir: Path) -> str:
    # Prefer Lung1-XXX style if present, else fallback to folder name
    # search upwards in path parts
    for part in reversed(patient_dir.parts):
        m = re.match(r"(?i)l?ung1[-_]?(\d{3})", part)
        if m:
            return f"LUNG1-{m.group(1)}"
    name = patient_dir.name
    for part in name.replace("_", "-").split("-"):
        if part.isdigit() and len(part) == 3:
            return f"LUNG1-{part}"
    return name


def select_roi_names(all_roi_names: Iterable[str]) -> list[str]:
    selected = []
    for name in all_roi_names:
        n = (name or "").lower()
        if any(k in n for k in ROI_NAME_KEYWORDS):
            selected.append(name)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for n in selected:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def save_nifti_from_sitk(img: sitk.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))


def save_mask_from_numpy_like_ct(mask: np.ndarray, ct_img: sitk.Image, out_path: Path) -> None:
    seg = sitk.GetImageFromArray(mask.astype(np.uint8))
    seg.SetSpacing(ct_img.GetSpacing())
    seg.SetOrigin(ct_img.GetOrigin())
    seg.SetDirection(ct_img.GetDirection())
    save_nifti_from_sitk(seg, out_path)


def convert_patient(patient_dir: Path, interim_out: Path) -> dict:
    """
    Returns a small dict with status for logging.
    """
    case_id = get_case_id(patient_dir)
    out_dir = interim_out / case_id
    out_ct = out_dir / "ct.nii.gz"
    out_seg = out_dir / "seg.nii.gz"

    # Skip if both outputs exist
    if out_ct.exists() and out_seg.exists():
        return {"case_id": case_id, "status": "skipped_exists"}

    # Read CT as SimpleITK using series reader (recursively search for series)
    series = find_best_ct_series(patient_dir)
    if not series:
        return {"case_id": case_id, "status": "no_ct_series"}
    file_names, series_dir = series
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    try:
        ct_img = reader.Execute()
    except Exception as e:
        return {"case_id": case_id, "status": "ct_read_error", "error": str(e)}

    # Find RTSTRUCT
    rtstruct_path = find_rtstruct(patient_dir)
    if rtstruct_path is None:
        # Save CT at least; segmentation missing
        save_nifti_from_sitk(ct_img, out_ct)
        return {"case_id": case_id, "status": "no_rtstruct_saved_ct_only"}

    # Build mask(s) using rt-utils
    try:
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(patient_dir),
            rt_struct_path=str(rtstruct_path),
        )
        roi_names = rtstruct.get_roi_names()
        sel = select_roi_names(roi_names)
        if not sel and roi_names:
            # Fallback: if there is exactly one ROI, take it
            if len(roi_names) == 1:
                sel = [roi_names[0]]
        if not sel:
            save_nifti_from_sitk(ct_img, out_ct)
            return {"case_id": case_id, "status": "no_matching_roi_saved_ct_only", "rois": roi_names}
        # Combine selected ROI masks
        combined = None
        for rn in sel:
            m = rtstruct.get_mask_for_roi(rn)  # numpy ndarray (z, y, x)
            combined = m if combined is None else (combined | m)
        # Save outputs
        save_nifti_from_sitk(ct_img, out_ct)
        save_mask_from_numpy_like_ct(combined.astype(np.uint8), ct_img, out_seg)
        return {"case_id": case_id, "status": "ok", "rois": sel}
    except Exception as e:
        save_nifti_from_sitk(ct_img, out_ct)
        return {"case_id": case_id, "status": "rtstruct_error_saved_ct_only", "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Lung1 DICOM (CT + RTSTRUCT) to NIfTI ct.nii.gz / seg.nii.gz")
    ap.add_argument("--raw-root", type=Path, required=True, help="Root directory with Lung1 raw DICOM folders")
    ap.add_argument("--interim-out", type=Path, required=True, help="Output root for interim NIfTI pairs")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of patients (0=all)")
    ap.add_argument("--log-json", type=Path, default=None, help="Optional path to write a JSON conversion log")
    args = ap.parse_args()

    # Gather patient roots by scanning for directory names matching LUNG1-XXX anywhere under raw-root
    candidates: list[Path] = []
    for root, dirs, files in os.walk(args.raw_root):
        for d in dirs:
            if re.match(r"(?i)l?ung1[-_]?(\d{3})", d):
                candidates.append(Path(root) / d)
    # Fallback to direct children if pattern not found
    patients = candidates or [p for p in args.raw_root.iterdir() if p.is_dir()]
    patients.sort()
    if args.limit and args.limit > 0:
        patients = patients[: args.limit]
    report = {"raw_root": str(args.raw_root), "interim_out": str(args.interim_out), "num_patients": len(patients), "cases": []}
    for p in patients:
        res = convert_patient(p, args.interim_out)
        report["cases"].append(res)
        status = res.get("status", "")
        case_id = res.get("case_id", p.name)
        rois = res.get("rois", [])
        if status == "ok":
            print(f"[OK] {case_id} rois={rois}")
        else:
            print(f"[{status}] {case_id}")
    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        with args.log_json.open("w") as f:
            json.dump(report, f, indent=2)
    # Print short summary
    ok = sum(1 for c in report["cases"] if c.get("status") == "ok")
    ct_only = sum(1 for c in report["cases"] if "saved_ct_only" in c.get("status", ""))
    print(f"Converted: {ok} with seg, {ct_only} CT-only, total processed {len(report['cases'])}")


if __name__ == "__main__":
    main()


