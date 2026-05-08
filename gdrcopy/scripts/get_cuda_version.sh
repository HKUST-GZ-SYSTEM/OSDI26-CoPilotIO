#!/bin/bash
# Usage: ./get_cuda_version.sh [nvcc_path]
NVCC="${1:-nvcc}"
CUDA_VERSION_FULL=$($NVCC --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
CUDA_VERSION_MAJOR=$(echo "$CUDA_VERSION_FULL" | cut -d'.' -f1)
CUDA_VERSION_MINOR=$(echo "$CUDA_VERSION_FULL" | cut -d'.' -f2)

if [ -z "$CUDA_VERSION_FULL" ]; then
    echo "Error: Could not detect CUDA version" >&2
    exit 1
fi

echo "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}"