from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def get_available_patient_ids(dataset_dir: Path) -> List[str]:
    """
    Discover usable subjects based on presence of nnU-Net-ready NIfTI files.

    We treat any case with a label in labelsTr (or, as a fallback, an image in
    imagesTr) as available. This ensures that splits are always derived from
    the actually usable segmentation cohort, even if the upstream conversion
    step changes or fails for some subjects.
    """
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    ids: set[str] = set()

    if labels_tr.exists():
        for lab in labels_tr.glob("*.nii.gz"):
            # labels are <case_id>.nii.gz
            ids.add(lab.stem)
    elif images_tr.exists():
        for img in images_tr.glob("*_0000.nii.gz"):
            # images are <case_id>_0000.nii.gz
            ids.add(img.name.replace("_0000.nii.gz", ""))

    return sorted(ids)


def stratified_split(
    df: pd.DataFrame,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int = 1337,
    strata_cols: Tuple[str, ...] = ("vendor", "recon_kernel", "slice_thickness_bin", "tumor_size_bin"),
) -> Tuple[List[str], List[str], List[str]]:
    """Simple proportional sampling by strata; falls back to random if strata missing."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    for c in strata_cols:
        if c not in df.columns:
            df[c] = "NA"
    # shuffle
    df = df.sample(frac=1.0, random_state=seed)
    # proportional allocation by strata
    groups = df.groupby(list(strata_cols))
    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []
    remaining_train = train_n
    remaining_val = val_n
    remaining_test = test_n
    for _, g in groups:
        ids = g["patient_id"].tolist()
        if not ids:
            continue
        # split roughly into thirds proportional to requested totals
        n = len(ids)
        # compute proportional sizes
        t_take = min(n, int(round(n * (train_n / max(train_n + val_n + test_n, 1e-6)))))
        v_take = min(n - t_take, int(round(n * (val_n / max(train_n + val_n + test_n, 1e-6)))))
        r = n - t_take - v_take
        train_ids.extend(ids[:t_take])
        val_ids.extend(ids[t_take : t_take + v_take])
        test_ids.extend(ids[t_take + v_take : t_take + v_take + r])
        remaining_train -= t_take
        remaining_val -= v_take
        remaining_test -= r
    # If counts off due to rounding, top up randomly from leftovers
    all_ids = df["patient_id"].tolist()
    used = set(train_ids + val_ids + test_ids)
    leftovers = [i for i in all_ids if i not in used]
    rng.shuffle(leftovers)

    def take(lst: List[str], n_take: int) -> None:
        for _ in range(max(0, n_take)):
            if leftovers:
                lst.append(leftovers.pop())

    take(train_ids, remaining_train)
    take(val_ids, remaining_val)
    take(test_ids, remaining_test)
    # clip to exact requested
    train_ids = train_ids[:train_n]
    val_ids = val_ids[:val_n]
    test_ids = test_ids[:test_n]
    # ensure disjoint sets
    train_ids = [i for i in train_ids if i not in set(val_ids) and i not in set(test_ids)]
    val_ids = [i for i in val_ids if i not in set(train_ids) and i not in set(test_ids)]
    test_ids = [i for i in test_ids if i not in set(train_ids) and i not in set(val_ids)]
    return train_ids, val_ids, test_ids


def write_splits_final(preprocessed_base: Path, train_ids: List[str], val_ids: List[str]) -> None:
    splits = [{"train": train_ids, "val": val_ids}]
    out = preprocessed_base / "splits_final.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote splits_final.json at {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate stratified patient-level splits for nnU-Net.")
    p.add_argument("--clinical_csv", type=Path, default=Path("data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"))
    p.add_argument("--mapping_csv", type=Path, default=None, help="Optional mapping of patient_id to case_id filenames")
    p.add_argument("--dataset_dir", type=Path, default=Path("data/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1"))
    p.add_argument("--preprocessed_base", type=Path, default=Path("data/nnunet/nnUNet_preprocessed/Dataset501_NSCLC_Lung1"))
    p.add_argument("--train", type=int, default=120)
    p.add_argument("--val", type=int, default=20)
    p.add_argument("--test", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.clinical_csv)
    # basic normalization of columns; adapt as needed
    df = df.rename(columns={"PatientID": "patient_id", "Manufacturer": "vendor", "ConvolutionKernel": "recon_kernel", "SliceThickness": "slice_thickness"})
    # binning
    if "slice_thickness" in df.columns:
        df["slice_thickness_bin"] = pd.qcut(df["slice_thickness"].astype(float), q=4, labels=False, duplicates="drop").astype(str)
    else:
        df["slice_thickness_bin"] = "NA"
    # placeholder tumor size bins if not present
    if "TumorVolume" in df.columns:
        df["tumor_size_bin"] = pd.qcut(df["TumorVolume"].astype(float), q=4, labels=False, duplicates="drop").astype(str)
    else:
        df["tumor_size_bin"] = "NA"

    # Restrict to subjects that actually have usable ct+seg in the prepared nnU-Net dataset.
    # This keeps the splitting fully data-driven: if upstream conversion changes,
    # the splits automatically reflect the new usable cohort.
    available_ids = get_available_patient_ids(args.dataset_dir)
    if available_ids:
        before = df.shape[0]
        df = df[df["patient_id"].isin(available_ids)].copy()
        after = df.shape[0]
        print(
            f"Restricted clinical cohort to {after} subjects with usable ct+seg "
            f"(from {before} clinical rows, {len(available_ids)} available nnU-Net cases)."
        )
    else:
        print(
            f"WARNING: No usable cases discovered under {args.dataset_dir}. "
            "Using all clinical subjects for splits."
        )

    # derive splits (counts remain configurable; by default 120/20/20 as per script args)
    train_ids, val_ids, test_ids = stratified_split(df, args.train, args.val, args.test, args.seed)

    # Simple summary for reporting
    total_usable = len(df["patient_id"].unique())
    print(
        f"Split summary (usable subjects={total_usable}): "
        f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )

    # write splits_final.json (train/val)
    write_splits_final(args.preprocessed_base, train_ids, val_ids)
    # write a small index for test set to populate imagesTs externally
    test_index = args.dataset_dir / "test_index.json"
    with test_index.open("w") as f:
        json.dump({"test_ids": test_ids}, f, indent=2)
    print(f"Wrote test_index.json with {len(test_ids)} ids at {test_index}")


if __name__ == "__main__":
    main()