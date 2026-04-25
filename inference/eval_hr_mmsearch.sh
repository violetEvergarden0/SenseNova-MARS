#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="$(cd "${REPO_DIR}/.." && pwd)"

export MODEL_BASE_URL="${MODEL_BASE_URL:-http://localhost:8888}"
export SUMMARIZER_BASE_URL="${SUMMARIZER_BASE_URL:-http://localhost:8181}"
export SUMMARIZER_MODEL="${SUMMARIZER_MODEL:-qwen3-32b}"

MODE="${MODE:-tool}"
JUDGE_CLIENT="${JUDGE_CLIENT:-azure}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-2024-11-20}"
MODEL_NAME="${MODEL_NAME:-SenseNova-MARS-8B}"
MODEL_TAG="${MODEL_NAME//\//_}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/outputs/hr_mmsearch_${MODEL_TAG}_${MODE}_$(date +%y%m%d%H%M%S)}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
SERPER_CONCURRENCY="${SERPER_CONCURRENCY:-1}"
MAX_TURNS="${MAX_TURNS:-50}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
SEARCH_CACHE_DIR="${SEARCH_CACHE_DIR:-${OUTPUT_DIR}/search_cache}"

if [ "${JUDGE_CLIENT}" = "azure" ]; then
  : "${AZURE_OPENAI_API_KEY:?Set AZURE_OPENAI_API_KEY for Azure judge}"
  : "${AZURE_OPENAI_BASE_URL:?Set AZURE_OPENAI_BASE_URL for Azure judge}"
  export AZURE_API_VERSION="${AZURE_API_VERSION:-2025-01-01-preview}"
elif [ "${JUDGE_CLIENT}" = "openai" ]; then
  : "${OPENAI_API_KEY:?Set OPENAI_API_KEY for OpenAI judge}"
  if [ -z "${OPENAI_BASE_URL:-}" ]; then
    echo "Warning: OPENAI_BASE_URL is not set; the judge will use the official OpenAI endpoint." >&2
  fi
else
  echo "Unsupported JUDGE_CLIENT: ${JUDGE_CLIENT}. Use azure or openai." >&2
  exit 1
fi

if [ "${MODE}" = "tool" ]; then
  : "${SERPER_API_KEY:?Set SERPER_API_KEY for tool mode}"
fi

mkdir -p "${OUTPUT_DIR}" "${SEARCH_CACHE_DIR}"

cd "${SCRIPT_DIR}"

ARGS=(
  --model-client openai
  --judge-client "${JUDGE_CLIENT}"
  --judge-model "${JUDGE_MODEL}"
  --model "${MODEL_NAME}"
  --mode "${MODE}"
  --datasets ../test_hr_mmsearch.json
  --data-root ..
  --output-dir "${OUTPUT_DIR}"
  --max-concurrent "${MAX_CONCURRENT}"
  --max-turns "${MAX_TURNS}"
  --serper-concurrency "${SERPER_CONCURRENCY}"
  --search-cache-dir "${SEARCH_CACHE_DIR}"
)

if [ "${MODE}" = "tool" ]; then
  ARGS+=(--tool-config tools_eval.yaml)
fi

if [ -n "${MAX_SAMPLES}" ]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}" --sample-offset "${SAMPLE_OFFSET}")
fi

python3 eval.py "${ARGS[@]}"
