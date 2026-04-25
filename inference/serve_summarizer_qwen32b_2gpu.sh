#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)/models/Qwen3-32B"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}" python3 -m sglang.launch_server \
  --model-path "${SUMMARIZER_MODEL_PATH:-${DEFAULT_MODEL_PATH}}" \
  --served-model-name "${SUMMARIZER_MODEL:-qwen3-32b}" \
  --tp-size "${SUMMARIZER_TP_SIZE:-2}" \
  --dtype "${SUMMARIZER_DTYPE:-bfloat16}" \
  --host "${SUMMARIZER_HOST:-0.0.0.0}" --port "${SUMMARIZER_PORT:-8181}" \
  --mem-fraction-static "${SUMMARIZER_MEM_FRACTION_STATIC:-0.3}"
