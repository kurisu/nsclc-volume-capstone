# nnU-Net Baseline (Apple Silicon, MPS)

This project integrates nnU-Net v2 as an external baseline and provides helper scripts to train locally on Apple Silicon (M1/M2/M3) using the PyTorch MPS backend.

## 1) Install (Apple Silicon)
- Create a conda/mamba env (Python 3.10–3.12).
- Install PyTorch (macOS wheels include MPS support) and nnU-Net v2:
```
pip install torch torchvision torchaudio
pip install nnunetv2 nibabel SimpleITK scikit-image pandas rich
```

## 2) Environment variables
Source the helper to set required nnU-Net paths and MPS fallback:
```
source scripts/nnunet_env.sh
```

## 3) Prepare dataset
Convert NIfTI pairs from `data/interim/<CASE>/ct.nii.gz` and `seg.nii.gz` to nnU-Net v2 raw layout:
```
make nnunet:prepare
```
This creates `data/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/{imagesTr,labelsTr,imagesTs}` and a `dataset.json`.

## 4) Plan and preprocess
```
make nnunet:preprocess
```

## 5) Train (fold 0, 3d_fullres) on MPS
```
make nnunet:train_fold0_mps
```
The wrapper `scripts/nnunet_train_mps.py` forces MPS if available and enables CPU fallback for unsupported kernels.

## 6) Predict and evaluate
- Place test images in `imagesTs` (optional).
- Predict:
```
make nnunet:predict
```
- Evaluate (bridges to repo evaluation tooling; extend as needed):
```
make nnunet:evaluate
```

Notes:
- Training on MPS is generally slower than CUDA but feasible for the baseline.
- If you hit OOM, nnU-Net auto-tunes batch size; you can also reduce workers.

