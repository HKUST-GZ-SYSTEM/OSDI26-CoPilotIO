#!/bin/bash

# Get CUDA version from get_cuda_version.sh
NVCC="${1:-nvcc}"
DIR="$(dirname "$0")"
CUDA_VERSION_FULL=$($DIR/get_cuda_version.sh $NVCC)

if [ -z "$CUDA_VERSION_FULL" ]; then
    echo "Error: Could not detect CUDA version" >&2
    exit 1
fi

version_ge() {
    # returns 0 (true) if $1 >= $2, 1 (false) otherwise
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}
# Require CUDA >= 8.0
if ! version_ge "$CUDA_VERSION_FULL" "8.0"; then
    echo "Error: CUDA version must be >= 8.0" >&2
    exit 1
fi

# Initialize with Pascal architecture
COMPUTE_LIST="60 61 62"
SM_LIST="60 61 62"

# Add Volta (7.0) if CUDA >= 9.0
if version_ge "$CUDA_VERSION_FULL" "9.0"; then
    COMPUTE_LIST="$COMPUTE_LIST 70 72"
    SM_LIST="$SM_LIST 70 72"
fi

# Add Turing (7.5) if CUDA >= 10.0
if version_ge "$CUDA_VERSION_FULL" "10.0"; then
    COMPUTE_LIST="$COMPUTE_LIST 75"
    SM_LIST="$SM_LIST 75"
fi

# Add Ampere (8.0, 8.6, 8.7) if CUDA >= 11.1
if version_ge "$CUDA_VERSION_FULL" "11.1"; then
    COMPUTE_LIST="$COMPUTE_LIST 80 86 87"
    SM_LIST="$SM_LIST 80 86 87"
fi

# Add Ada Lovelace (8.9) if CUDA >= 11.8
if version_ge "$CUDA_VERSION_FULL" "11.8"; then
    COMPUTE_LIST="$COMPUTE_LIST 89"
    SM_LIST="$SM_LIST 89"
fi

# Add Hopper (9.0) if CUDA >= 12.0
if version_ge "$CUDA_VERSION_FULL" "12.0"; then
    COMPUTE_LIST="$COMPUTE_LIST 90"
    SM_LIST="$SM_LIST 90"
fi

# For CUDA >= 12.6, enable all supported architectures since this is the first version where nvcc drops sm50 support, and pplat requires atomics (available only in sm60+)
if version_ge "$CUDA_VERSION_FULL" "12.6"; then
    echo "-arch=all"
    exit 0
fi

# Generate NVCC flags
GENCODE_FLAGS=""
for compute in $COMPUTE_LIST; do
    GENCODE_FLAGS="$GENCODE_FLAGS -gencode arch=compute_$compute,code=compute_$compute"
done

for sm in $SM_LIST; do
    GENCODE_FLAGS="$GENCODE_FLAGS -gencode arch=compute_$sm,code=sm_$sm"
done

echo "$GENCODE_FLAGS"
