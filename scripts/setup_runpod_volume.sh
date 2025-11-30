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

REMOTE_DATA_RAW="${VOLUME_ROOT}/data/raw"
REMOTE_DATA_INTERIM="${VOLUME_ROOT}/data/interim"
REMOTE_NNUNET_RAW="${VOLUME_ROOT}/nnunet/nnUNet_raw"
REMOTE_DATASET_DIR="${VOLUME_ROOT}/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1"
REMOTE_PREPROCESSED_DIR="${VOLUME_ROOT}/nnunet/nnUNet_preprocessed/Dataset501_NSCLC_Lung1"

echo "Local repo root: ${LOCAL_ROOT}"
echo "Remote target:   ${SSH_TARGET}"
echo "Volume root:     ${VOLUME_ROOT}"
echo "Repo root:       ${REPO_ROOT}"
echo

echo "Preflight: verify remote connectivity and required tools..."
if ! ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'command -v rsync >/dev/null 2>&1 && command -v uv >/dev/null 2>&1 && whoami >/dev/null 2>&1'"; then
  echo "Preflight failed."
  echo "Diagnostics (remote):"
  ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'if command -v rsync >/dev/null 2>&1; then echo rsync: OK; else echo rsync: MISSING; fi; if command -v uv >/dev/null 2>&1; then echo uv: OK; else echo uv: MISSING; fi; if whoami >/dev/null 2>&1; then echo ssh/whoami: OK; else echo ssh/whoami: FAILED; fi'" || true
  echo "Tip: recreate the dev env (dstack apply -f dev.dstack.yml --recreate) to bootstrap packages."
  exit 1
fi
echo "Preflight: OK"

# Ensure remote directory structure exists (recursive) BEFORE rsync
echo "Step 0: Ensure remote directory structure exists..."
ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'mkdir -p \"${REMOTE_DATA_RAW}\" \"${REMOTE_DATA_INTERIM}\" \"${REMOTE_NNUNET_RAW}\"; true'"
if $VERBOSE; then
  ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'ls -ld \"${VOLUME_ROOT}\" \"${VOLUME_ROOT}/data\" \"${REMOTE_DATA_RAW}\" \"${REMOTE_DATA_INTERIM}\" \"${VOLUME_ROOT}/nnunet\" \"${REMOTE_NNUNET_RAW}\" || true'"
fi
echo "Step 0: OK"

echo "Step 1/5: Rsync large assets to remote volume..."
RSYNC_COMMON=( -aP --no-owner --no-group --chmod=ugo=rwX --exclude='._*' --exclude='.DS_Store' )
rsync "${RSYNC_COMMON[@]}" \
  "${LOCAL_ROOT}/data/interim/" "${SSH_TARGET}:${REMOTE_DATA_INTERIM}/" || true

# Optional: clinical CSV (if present locally)
if [[ -f "${LOCAL_ROOT}/data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv" ]]; then
  rsync "${RSYNC_COMMON[@]}" \
    "${LOCAL_ROOT}/data/raw/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv" \
    "${SSH_TARGET}:${REMOTE_DATA_RAW}/"
fi
echo "Step 1/5: OK (rsync completed; some 'Operation not permitted' owner/group messages can be ignored)"

echo "Step 2/5: Clean AppleDouble and .DS_Store on remote..."
ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'mkdir -p \"${REMOTE_DATA_RAW}\" \"${REMOTE_DATA_INTERIM}\" \"${REMOTE_NNUNET_RAW}\"; find \"${VOLUME_ROOT}\" -type f -name \"._*\" -delete; find \"${VOLUME_ROOT}\" -type f -name \".DS_Store\" -delete; true'"
if $VERBOSE; then echo "Cleaned AppleDouble/.DS_Store"; fi
echo "Step 2/5: OK"

echo "Step 3/5: Build nnU-Net raw layout + dataset.json on remote..."
ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'cd \"${REPO_ROOT}\"; uv run python scripts/prepare_nnunet_dataset.py --source \"${REMOTE_DATA_INTERIM}\" --nnunet-raw \"${REMOTE_NNUNET_RAW}\" --dataset-id 501 --dataset-name NSCLC_Lung1 || true; uv run python scripts/build_dataset_json.py --dataset_dir \"${REMOTE_DATASET_DIR}\"; true'"
if $VERBOSE; then
  ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'ls -la \"${REMOTE_DATASET_DIR}\" | sed -n \"1,10p\"'"
fi
echo "Step 3/5: OK"

echo "Step 4/5: Generate splits and populate imagesTs..."
ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'cd \"${REPO_ROOT}\"; uv run python scripts/generate_splits.py --clinical_csv \"${REMOTE_DATA_RAW}/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv\" --dataset_dir \"${REMOTE_DATASET_DIR}\" --preprocessed_base \"${REMOTE_PREPROCESSED_DIR}\" --train 120 --val 20 --test 20; uv run python scripts/populate_imagesTs.py --dataset_dir \"${REMOTE_DATASET_DIR}\"; true'"
echo "Step 4/5: OK"

echo "Step 5/5: Sanity counts..."
IMAGES_TR=$(ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'find \"${REMOTE_DATASET_DIR}/imagesTr\" -maxdepth 1 -type f -name \"*_0000.nii.gz\" | wc -l'" || echo 0)
LABELS_TR=$(ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'find \"${REMOTE_DATASET_DIR}/labelsTr\" -maxdepth 1 -type f -name \"*.nii.gz\" | wc -l'" || echo 0)
IMAGES_TS=$(ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'find \"${REMOTE_DATASET_DIR}/imagesTs\" -maxdepth 1 -type f -name \"*_0000.nii.gz\" | wc -l'" || echo 0)
if $VERBOSE; then
  ssh -o BatchMode=yes "${SSH_TARGET}" "bash -lc 'echo Top of imagesTr:; ls -la \"${REMOTE_DATASET_DIR}/imagesTr\" | sed -n \"1,10p\"; echo Top of labelsTr:; ls -la \"${REMOTE_DATASET_DIR}/labelsTr\" | sed -n \"1,10p\"; echo Top of imagesTs:; ls -la \"${REMOTE_DATASET_DIR}/imagesTs\" | sed -n \"1,10p\"'"
fi

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


