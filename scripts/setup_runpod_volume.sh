#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/setup_runpod_volume.sh [SSH_TARGET] [VOLUME_ROOT] [REPO_ROOT]
# Example:
#   ./scripts/setup_runpod_volume.sh cursor-dev /workspace/capstone_storage /workspace/capstone
#
# This script:
# 1) Rsyncs large assets from the local repo to the remote RunPod volume
# 2) Cleans macOS metadata files on the remote
# 3) Builds nnU-Net raw layout + dataset.json on the remote
# 4) Generates stratified splits and populates imagesTs
# 5) Prints sanity counts
#
# Assumptions:
# - SSH_TARGET resolves (e.g., an SSH config host for your dev env)
# - The repo exists on the remote at REPO_ROOT
# - You have NIfTI pairs under local data/interim/<CASE>/{ct.nii.gz, seg.nii.gz}

SSH_TARGET="${1:-cursor-dev}"
VOLUME_ROOT="${2:-/workspace/capstone_storage}"
REPO_ROOT="${3:-/workspace/capstone}"

LOCAL_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$(pwd)")"

echo "Local repo root: ${LOCAL_ROOT}"
echo "Remote target:   ${SSH_TARGET}"
echo "Volume root:     ${VOLUME_ROOT}"
echo "Repo root:       ${REPO_ROOT}"
echo

# Ensure remote directory structure exists (recursive) BEFORE rsync
ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc " \
  mkdir -p ${VOLUME_ROOT}/data/raw; \
  mkdir -p ${VOLUME_ROOT}/data/interim; \
  mkdir -p ${VOLUME_ROOT}/nnunet/nnUNet_raw; \
"

echo "Step 1/5: Rsync large assets to remote volume..."
rsync -avP \
  --exclude='._*' --exclude='.DS_Store' \
  "${LOCAL_ROOT}/data/interim/" "${SSH_TARGET}:${VOLUME_ROOT}/data/interim/" || true

# Optional: clinical CSV (if present locally)
if [[ -f "${LOCAL_ROOT}/data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv" ]]; then
  rsync -avP \
    --exclude='._*' --exclude='.DS_Store' \
    "${LOCAL_ROOT}/data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv" \
    "${SSH_TARGET}:${VOLUME_ROOT}/data/raw/"
fi

echo "Step 2/5: Clean AppleDouble and .DS_Store on remote..."
ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc " \
  mkdir -p ${VOLUME_ROOT}/data/{raw,interim} ${VOLUME_ROOT}/nnunet/nnUNet_raw; \
  find ${VOLUME_ROOT} -type f -name '._*' -delete; \
  find ${VOLUME_ROOT} -type f -name '.DS_Store' -delete; \
"

echo "Step 3/5: Build nnU-Net raw layout + dataset.json on remote..."
ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc " \
  set -euo pipefail; \
  cd ${REPO_ROOT}; \
  python scripts/prepare_nnunet_dataset.py \
    --source ${VOLUME_ROOT}/data/interim \
    --nnunet-raw ${VOLUME_ROOT}/nnunet/nnUNet_raw \
    --dataset-id 501 \
    --dataset-name NSCLC_Lung1 || true; \
  python scripts/build_dataset_json.py \
    --dataset_dir ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1; \
"

echo "Step 4/5: Generate splits and populate imagesTs..."
ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc " \
  set -euo pipefail; \
  cd ${REPO_ROOT}; \
  python scripts/generate_splits.py \
    --clinical_csv ${VOLUME_ROOT}/data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
    --dataset_dir ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1 \
    --preprocessed_base ${VOLUME_ROOT}/nnunet/nnUNet_preprocessed/Dataset501_NSCLC_Lung1 \
    --train 120 --val 20 --test 20; \
  python scripts/populate_imagesTs.py \
    --dataset_dir ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1; \
"

echo "Step 5/5: Sanity counts..."
ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc " \
  echo 'imagesTr count:'; \
  find ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/imagesTr -name '*_0000.nii.gz' | wc -l; \
  echo 'labelsTr count:'; \
  find ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/labelsTr -name '*.nii.gz' | wc -l; \
  echo 'imagesTs count:'; \
  find ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/imagesTs -name '*_0000.nii.gz' | wc -l; \
  echo 'Done.'; \
"

echo
echo "Volume setup complete. Next on remote:"
echo "  dstack apply -f preprocess.dstack.yml --yes"
echo "  dstack apply -f train.dstack.yml --yes"
echo "  dstack apply -f predict.dstack.yml --yes"
echo "  dstack apply -f eval.dstack.yml --yes"


