#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)/models/SenseNova-MARS-8B"

export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-4096}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH:-${DEFAULT_MODEL_PATH}}" \
  --host "${MODEL_HOST:-0.0.0.0}" --port "${MODEL_PORT:-8888}" \
  --dtype "${MODEL_DTYPE:-bfloat16}" \
  --served-model-name "${SERVED_MODEL_NAME:-SenseNova-MARS-8B}" \
  --tp "${MODEL_TP:-1}" \
  --mem-fraction-static "${MODEL_MEM_FRACTION_STATIC:-0.45}" \
  --enable-deterministic-inference
