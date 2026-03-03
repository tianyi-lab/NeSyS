#!/usr/bin/env bash
set -euo pipefail

# Generate neural evaluation summaries (logprob MCQ ranking) from:
# - Hugging Face dataset repo (3 tasks × 2 splits)
# - Hugging Face model adapters (one repo per env/model)
#
# Output files are written under: nesys/eval_results/
#
# Note: This script can be compute-heavy and may require a GPU.

DATASET_REPO_ID="${DATASET_REPO_ID:-cindermond/nesys-world-model-benchmark}"

# Base models
LLAMA_BASE="${LLAMA_BASE:-meta-llama/Llama-3.2-1B-Instruct}"
QWEN_BASE="${QWEN_BASE:-Qwen/Qwen3-4B}"

# Adapter repos (uploaded)
PLCRAFT_LLAMA_ADAPTER="${PLCRAFT_LLAMA_ADAPTER:-cindermond/world-model-plancraft-llama3-2-1b-instruct-filtered}"
PLCRAFT_QWEN_ADAPTER="${PLCRAFT_QWEN_ADAPTER:-cindermond/world-model-plancraft-qwen3-4b-filtered}"

SCI_LLAMA_ADAPTER="${SCI_LLAMA_ADAPTER:-cindermond/world-model-scienceworld-llama3-2-1b-instruct-filtered}"
SCI_QWEN_ADAPTER="${SCI_QWEN_ADAPTER:-cindermond/world-model-scienceworld-qwen3-4b-filtered}"

WS_LLAMA_ADAPTER="${WS_LLAMA_ADAPTER:-cindermond/world-model-webshop-llama3-2-1b-instruct-filtered}"
WS_QWEN_ADAPTER="${WS_QWEN_ADAPTER:-cindermond/world-model-webshop-qwen3-4b-filtered}"

run_one () {
  local env="$1"
  local split="$2"
  local base="$3"
  local adapter="$4"
  local out_prefix="$5"

  echo "=== env=${env} split=${split} base=${base} adapter=${adapter} ==="
  python3 eval_transition_mcq_logprob.py \
    --base_model "${base}" \
    --adapter "${adapter}" \
    --dataset_repo_id "${DATASET_REPO_ID}" \
    --task "${env}" \
    --split "${split}" \
    --output_prefix "eval_results/${out_prefix}_${split}"
}

# PlanCraft
run_one plancraft dev  "${LLAMA_BASE}" "${PLCRAFT_LLAMA_ADAPTER}" "plancraft_sft_llama3-2-1b-instruct_filtered"
run_one plancraft test "${LLAMA_BASE}" "${PLCRAFT_LLAMA_ADAPTER}" "plancraft_sft_llama3-2-1b-instruct_filtered"
run_one plancraft dev  "${QWEN_BASE}"  "${PLCRAFT_QWEN_ADAPTER}"  "plancraft_sft_qwen3-4b_filtered"
run_one plancraft test "${QWEN_BASE}"  "${PLCRAFT_QWEN_ADAPTER}"  "plancraft_sft_qwen3-4b_filtered"

# ScienceWorld
run_one scienceworld dev  "${LLAMA_BASE}" "${SCI_LLAMA_ADAPTER}" "scienceworld_sft_llama3-2-1b-instruct_filtered"
run_one scienceworld test "${LLAMA_BASE}" "${SCI_LLAMA_ADAPTER}" "scienceworld_sft_llama3-2-1b-instruct_filtered"
run_one scienceworld dev  "${QWEN_BASE}"  "${SCI_QWEN_ADAPTER}"  "scienceworld_sft_qwen3-4b_filtered"
run_one scienceworld test "${QWEN_BASE}"  "${SCI_QWEN_ADAPTER}"  "scienceworld_sft_qwen3-4b_filtered"

# WebShop
run_one webshop dev  "${LLAMA_BASE}" "${WS_LLAMA_ADAPTER}" "webshop_transition_qa_sft_llama3-2-1b-instruct_filtered"
run_one webshop test "${LLAMA_BASE}" "${WS_LLAMA_ADAPTER}" "webshop_transition_qa_sft_llama3-2-1b-instruct_filtered"
run_one webshop dev  "${QWEN_BASE}"  "${WS_QWEN_ADAPTER}"  "webshop_transition_qa_sft_qwen3-4b_filtered"
run_one webshop test "${QWEN_BASE}"  "${WS_QWEN_ADAPTER}"  "webshop_transition_qa_sft_qwen3-4b_filtered"

echo "Done. See nesys/eval_results/*_summary.json"

