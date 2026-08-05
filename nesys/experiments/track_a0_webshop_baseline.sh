#!/usr/bin/env bash
# WebShop-7B-SFT baseline reported in the open-ended evaluation (0.2755).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-${NESYS_DIR}/eval_results_experiments/webshop_a0_baseline}"

cd "${NESYS_DIR}"
python eval_webshop_agent.py \
  --agent_model "${WEBSHOP_AGENT_MODEL}" \
  --modes naive \
  --num_sessions "${WEBSHOP_NUM_SESSIONS}" \
  --session_offset "${WEBSHOP_SESSION_OFFSET}" \
  --max_steps "${WEBSHOP_MAX_STEPS}" \
  --agent_temperature 0.2 \
  --seed "${WEBSHOP_SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  "${WEBSHOP_CACHE_ARGS[@]}" \
  "$@"

