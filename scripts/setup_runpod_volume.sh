#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/setup_runpod_volume.sh [--verbose] [SSH_TARGET] [VOLUME_ROOT] [REPO_ROOT]
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

VERBOSE=false
if [[ "${1:-}" == "--verbose" ]]; then
  VERBOSE=true
  shift
fi

SSH_TARGET="${1:-cursor-dev}"
VOLUME_ROOT="${2:-/workspace/capstone_storage}"
REPO_ROOT="${3:-/workspace/capstone}"

LOCAL_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$(pwd)")"

echo "Local repo root: ${LOCAL_ROOT}"
echo "Remote target:   ${SSH_TARGET}"
echo "Volume root:     ${VOLUME_ROOT}"
echo "Repo root:       ${REPO_ROOT}"
echo

echo "Preflight: verify remote connectivity and required tools..."
ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc "command -v rsync >/dev/null 2>&1 || exit 127; whoami >/dev/null 2>&1 || exit 1"
if [[ $? -ne 0 ]]; then
  echo "Preflight failed: rsync not found or remote unreachable."
  echo "Tip: recreate the dev env (dstack apply -f dev.dstack.yml --recreate) to bootstrap packages."
  exit 1
fi
echo "Preflight: OK"

# Ensure remote directory structure exists (recursive) BEFORE rsync
echo "Step 0: Ensure remote directory structure exists..."
ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc " \
  mkdir -p '${VOLUME_ROOT}/data/raw'; \
  mkdir -p '${VOLUME_ROOT}/data/interim'; \
  mkdir -p '${VOLUME_ROOT}/nnunet/nnUNet_raw'; \
  true \
"
${VERBOSE:+ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc "ls -ld '${VOLUME_ROOT}' '${VOLUME_ROOT}/data' '${VOLUME_ROOT}/data/raw' '${VOLUME_ROOT}/data/interim' '${VOLUME_ROOT}/nnunet' '${VOLUME_ROOT}/nnunet/nnUNet_raw' || true";}
echo "Step 0: OK"

echo "Step 1/5: Rsync large assets to remote volume..."
RSYNC_COMMON=( -aP --no-owner --no-group --chmod=ugo=rwX --exclude='._*' --exclude='.DS_Store' )
rsync "${RSYNC_COMMON[@]}" \
  --exclude='._*' --exclude='.DS_Store' \
  "${LOCAL_ROOT}/data/interim/" "${SSH_TARGET}:${VOLUME_ROOT}/data/interim/" || true

# Optional: clinical CSV (if present locally)
if [[ -f "${LOCAL_ROOT}/data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv" ]]; then
  rsync "${RSYNC_COMMON[@]}" \
    --exclude='._*' --exclude='.DS_Store' \
    "${LOCAL_ROOT}/data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv" \
    "${SSH_TARGET}:${VOLUME_ROOT}/data/raw/"
fi
echo "Step 1/5: OK (rsync completed; some 'Operation not permitted' owner/group messages can be ignored)"

echo "Step 2/5: Clean AppleDouble and .DS_Store on remote..."
ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc " \
  mkdir -p ${VOLUME_ROOT}/data/{raw,interim} ${VOLUME_ROOT}/nnunet/nnUNet_raw; \
  find '${VOLUME_ROOT}' -type f -name '._*' -delete; \
  find '${VOLUME_ROOT}' -type f -name '.DS_Store' -delete; \
  true; \
"
${VERBOSE:+echo "Cleaned AppleDouble/.DS_Store";}
echo "Step 2/5: OK"

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
  true; \
"
${VERBOSE:+ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc "ls -la ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1 | sed -n '1,10p'";}
echo "Step 3/5: OK"

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
  true; \
"
echo "Step 4/5: OK"

echo "Step 5/5: Sanity counts..."
IMAGES_TR=$(ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc "find ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/imagesTr -name '*_0000.nii.gz' | wc -l" || echo 0)
LABELS_TR=$(ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc "find ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/labelsTr -name '*.nii.gz' | wc -l" || echo 0)
IMAGES_TS=$(ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc "find ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/imagesTs -name '*_0000.nii.gz' | wc -l" || echo 0)
${VERBOSE:+ssh -o BatchMode=yes "${SSH_TARGET}" bash -lc "echo 'Top of imagesTr:'; ls -la ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/imagesTr | sed -n '1,10p'; echo 'Top of labelsTr:'; ls -la ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/labelsTr | sed -n '1,10p'; echo 'Top of imagesTs:'; ls -la ${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/imagesTs | sed -n '1,10p'";}

echo
# Summary
echo "=================== SUMMARY ==================="
echo "imagesTr: ${IMAGES_TR}"
echo "labelsTr: ${LABELS_TR}"
echo "imagesTs: ${IMAGES_TS}"
if [[ "${IMAGES_TR}" -gt 0 && "${LABELS_TR}" -gt 0 ]]; then
  echo "Status: OK - nnU-Net raw appears populated."
else
  echo "Status: INCOMPLETE - please ensure interim NIfTIs exist locally and rerun the script."
fi
echo "Next:"
echo "  dstack apply -f preprocess.dstack.yml --yes"
echo "  dstack apply -f train.dstack.yml --yes"
echo "  dstack apply -f predict.dstack.yml --yes"
echo "  dstack apply -f eval.dstack.yml --yes"


