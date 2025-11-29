from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple


def find_cases(source_root: Path) -> List[Tuple[str, Path, Path]]:
    """
    Find cases under source_root that contain ct.nii.gz and seg.nii.gz.
    Returns list of (case_id, ct_path, seg_path).
    """
    cases: List[Tuple[str, Path, Path]] = []
    if not source_root.exists():
        return cases
    for child in sorted(source_root.iterdir()):
        if not child.is_dir():
            continue
        ct = child / "ct.nii.gz"
        seg = child / "seg.nii.gz"
        if ct.exists() and seg.exists():
            cases.append((child.name, ct, seg))
    return cases


def write_dataset_json(
    task_dir: Path,
    dataset_name: str,
    cases: List[str],
    images_subdir: str = "imagesTr",
    labels_subdir: str = "labelsTr",
) -> None:
    """
    Write nnU-Net v2 dataset.json with minimal required fields.
    """
    training_entries = [
        {
            "image": f"./{images_subdir}/{cid}_0000.nii.gz",
            "label": f"./{labels_subdir}/{cid}.nii.gz",
        }
        for cid in cases
    ]
    ds = {
        "name": dataset_name,
        "description": "NSCLC Lung1 CT tumor segmentation",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "tensorImageSize": "3D",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(training_entries),
        "file_ending": ".nii.gz",
        "training": training_entries,
        "test": [],
    }
    with (task_dir / "dataset.json").open("w") as f:
        json.dump(ds, f, indent=2)


def sanity_check_nifti(ct_path: Path, seg_path: Path) -> None:
    """
    Basic sanity checks: load headers to ensure files are valid NIfTI.
    """
    try:
        import nibabel as nib  # type: ignore  # noqa: WPS433
    except Exception:
        # If nibabel is not available, skip sanity checks
        return
    try:
        nib.load(str(ct_path))  # type: ignore[attr-defined]
        nib.load(str(seg_path))  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"NIfTI sanity check failed for {ct_path} / {seg_path}: {e}")


def copy_case(ct: Path, seg: Path, images_tr: Path, labels_tr: Path, case_id: str) -> None:
    shutil.copy2(ct, images_tr / f"{case_id}_0000.nii.gz")
    shutil.copy2(seg, labels_tr / f"{case_id}.nii.gz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare nnU-Net v2 dataset from NIfTI pairs.")
    p.add_argument(
        "--source",
        type=Path,
        default=Path("data/interim"),
        help="Directory containing case folders with ct.nii.gz and seg.nii.gz",
    )
    p.add_argument(
        "--nnunet-raw",
        type=Path,
        default=Path("data/nnunet/nnUNet_raw"),
        help="Path to nnUNet_raw root",
    )
    p.add_argument("--dataset-id", type=int, default=501, help="Dataset ID number")
    p.add_argument("--dataset-name", type=str, default="NSCLC_Lung1", help="Dataset name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_key = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    task_dir = args.nnunet_raw / dataset_key
    images_tr = task_dir / "imagesTr"
    labels_tr = task_dir / "labelsTr"
    images_ts = task_dir / "imagesTs"

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    cases = find_cases(args.source)
    if not cases:
        print(f"No cases found in {args.source}. Expected subdirs with ct.nii.gz and seg.nii.gz.")
        return

    copied_ids: List[str] = []
    for case_id, ct_path, seg_path in cases:
        sanity_check_nifti(ct_path, seg_path)
        copy_case(ct_path, seg_path, images_tr, labels_tr, case_id)
        copied_ids.append(case_id)
        print(f"Added case: {case_id}")

    write_dataset_json(task_dir, args.dataset_name, copied_ids)
    print(f"Wrote dataset.json with {len(copied_ids)} training cases to {task_dir}")


if __name__ == "__main__":
    main()


