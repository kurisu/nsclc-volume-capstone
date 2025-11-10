# nnU-Net as External Baseline (not vendored)

We treat nnU-Net as an external baseline. Use upstream installation and training, then import predicted masks here for evaluation alongside other models.

## 1) Install nnU-Net (refer to upstream docs)
- Docs: `https://github.com/MIC-DKFZ/nnUNet`
- Typical install (conda or venv). Ensure CUDA if training on GPU.

## 2) Data Preparation for Lung1
Prepare the dataset in nnU-Net’s expected structure (Task directory). This repo does not convert to nnU-Net format; follow nnU-Net guidelines for dataset conversion.

Hints:
- Inputs: DICOM CT and RTSTRUCT (GTV). You may preconvert RTSTRUCT→NIfTI masks using your own tooling, then build the Task.
- Verify FrameOfReferenceUID and alignment.

## 3) Train nnU-Net
Example (pseudo):
```
nnUNetv2_plan_and_preprocess -d <TASK_ID>
nnUNetv2_train 3d_fullres nnUNetTrainer <TASK_ID> 0
```

## 4) Export Predictions
Generate predictions (NIfTI masks) for your evaluation split, then copy into this repo, e.g.:
```
data/processed/lung1/preds/nnunet/<patient_id>.nii.gz
```

## 5) Evaluate in This Repo
Point evaluation to the predictions directory via `configs/default.yaml` (you may add a field like `paths.predictions: data/processed/lung1/preds/nnunet`) and run:
```
make evaluate
```

Notes:
- Keep thresholding/post-processing consistent across models where applicable.
- Compute volume metrics in native spacing per the project plan.


