#!/usr/bin/env bash
set -euo pipefail

##
## Run nnU-Net v2 predictions using an external results directory (with your trained
## ResEncUNetL plans model) and then evaluate against the Lung1 labels using
## the project’s evaluation tooling.
##
## Assumptions:
## - You are running this from the repo root
## - Dependencies are installed via uv / pyproject:
##     uv pip install -e '.[dev,nnunet]'
## - nnUNet v2 CLI is on PATH (installed via the nnunet extra above).
##

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# 1) Configure nnU-Net env paths relative to this repo (raw / preprocessed)
source scripts/nnunet_env.sh

# 2) Point nnUNet paths to the external storage that contains the fully prepared
#    Dataset501_NSCLC_Lung1 (raw + preprocessed + results).
EXTERNAL_NNUNET_ROOT_DEFAULT="/Users/kai/workspace/capstone_transfer/capstone_storage/nnunet"
EXTERNAL_NNUNET_ROOT="${EXTERNAL_NNUNET_ROOT:-${EXTERNAL_NNUNET_ROOT_DEFAULT}}"

export nnUNet_raw="${EXTERNAL_NNUNET_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${EXTERNAL_NNUNET_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${EXTERNAL_NNUNET_ROOT}/nnUNet_results"

echo "Using nnUNet_raw          = ${nnUNet_raw}"
echo "Using nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "Using nnUNet_results      = ${nnUNet_results}"

RAW_DS_ROOT="${nnUNet_raw}/Dataset501_NSCLC_Lung1"

# 3) Predict on held-out test images in the *external* imagesTs using the
#    ResEncUNetL plans. Explicitly use checkpoint_best.pth (nnUNetv2_predict
#    defaults to checkpoint_final.pth).
#    We also run with -device mps and -npp/-nps 0 to avoid the MPS pin_memory
#    issue and still leverage the GPU.
nnUNetv2_predict \
  -i "${RAW_DS_ROOT}/imagesTs" \
  -o data/nnunet/predsTs \
  -d 501 \
  -c 3d_fullres \
  -f 0 \
  -p nnUNetResEncUNetLPlans \
  -chk checkpoint_best.pth \
  -device mps \
  -npp 0 \
  -nps 0

# 4) Evaluate predictions vs labelsTr using the project bridge
python -m src.evaluate \
  --config configs/default.yaml \
  --nnunet-preds data/nnunet/predsTs \
  --labels-dir "${RAW_DS_ROOT}/labelsTr" \
  --out data/processed/nnunet_fold0_metrics.csv

echo "nnU-Net external model evaluation complete."
echo "Per-case metrics written to data/processed/nnunet_fold0_metrics.csv"


