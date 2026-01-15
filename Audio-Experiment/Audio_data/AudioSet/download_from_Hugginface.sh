#!/bin/bash
# Download and extract the balanced AudioSet dataset from Hugging Face.
# Warning: needs hundreds of GBs of free space and stable internet.
# Run: bash download_audioset_balanced.sh

set -e

# Where to store everything
BASE_DIR="/leonardo_work/ICT25_ESP/sdigioia/Audio_data/AudioSet/balanced"
mkdir -p "${BASE_DIR}"
cd "${BASE_DIR}"

# Number of training and test tar shards
TRAIN_SHARDS=10
TEST_SHARDS=10

# Base URL for Hugging Face
BASE_URL="https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/balanced"

download_and_extract() {
  local SPLIT=$1
  local NUM_SHARDS=$2

  echo "=== Downloading ${SPLIT} split (${NUM_SHARDS} shards) ==="
  mkdir -p "${SPLIT}"
  cd "${SPLIT}"

  for i in $(seq -f "%05g" 0 9);
  do
    FILE="${SPLIT}-${i}-of-00010.tar"
    URL="${BASE_URL}/${FILE}"
    echo "Downloading ${FILE} ..."
    wget -q --show-progress "${URL}" -O "${FILE}"
    echo "Extracting ${FILE} ..."
    mkdir -p "${SPLIT}_${i}"
    tar -xf "${FILE}" -C "${SPLIT}_${i}"
    #rm "${FILE}"   # comment this line if you want to keep tarballs
  done

  cd ..
}

# Download train and test shards
download_and_extract "train" "${TRAIN_SHARDS}"
download_and_extract "test" "${TEST_SHARDS}"

echo "All balanced AudioSet shards downloaded and extracted to ${BASE_DIR}"

