#!/usr/bin/env bash
# Usage: source scripts/nnunet_env.sh

# Enable CPU fallback for ops without MPS kernels
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Set nnU-Net paths relative to this repo
export nnUNet_raw="data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="data/nnunet/nnUNet_preprocessed"
export nnUNet_results="data/nnunet/nnUNet_results"

echo "Configured nnU-Net env:"
echo "  nnUNet_raw           = ${nnUNet_raw}"
echo "  nnUNet_preprocessed  = ${nnUNet_preprocessed}"
echo "  nnUNet_results       = ${nnUNet_results}"
echo "  PYTORCH_ENABLE_MPS_FALLBACK = ${PYTORCH_ENABLE_MPS_FALLBACK}"


