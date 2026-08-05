#!/usr/bin/env bash
# Shared defaults for the two open-ended WebShop runs reported in the paper.
# Any value can be overridden through the environment before launching a run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NESYS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${NESYS_DIR}/.." && pwd)"

: "${WEBSHOP_ROOT:=${REPO_ROOT}/webshop}"
: "${WEBSHOP_AGENT_MODEL:=Jianwen/Webshop-7B-SFT}"
: "${WEBSHOP_TRANSITION_BASE:=meta-llama/Llama-3.2-1B-Instruct}"
: "${WEBSHOP_TRANSITION_ADAPTER:=cindermond/world-model-webshop-llama3-2-1b-instruct-filtered}"
: "${WEBSHOP_RULE_FILE:=${NESYS_DIR}/final_rules/rules_webshop_transition_qa_sft_llama3-2-1b-instruct_final.py}"
: "${WEBSHOP_NUM_SESSIONS:=100}"
: "${WEBSHOP_SESSION_OFFSET:=0}"
: "${WEBSHOP_MAX_STEPS:=15}"
: "${WEBSHOP_TOP_K_CANDIDATES:=10}"
: "${WEBSHOP_TOP_K_STATES_CONTEXT:=3}"
: "${WEBSHOP_SEED:=0}"
: "${WEBSHOP_CACHE_DIR:=}"

WEBSHOP_CACHE_ARGS=()
if [[ -n "${WEBSHOP_CACHE_DIR}" ]]; then
  WEBSHOP_CACHE_ARGS+=(--cache_dir "${WEBSHOP_CACHE_DIR}")
fi

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

export WEBSHOP_ROOT
export PYTHONPATH="${NESYS_DIR}:${WEBSHOP_ROOT}:${PYTHONPATH:-}"

