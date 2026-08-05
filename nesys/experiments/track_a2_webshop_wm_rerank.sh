#!/usr/bin/env bash
# Rule-guided one-step lookahead reported in the open-ended evaluation (0.3276).
# The top action--successor pairs advise the agent; the WM does not execute its
# argmax directly.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-${NESYS_DIR}/eval_results_experiments/webshop_a2_wm_rerank}"

cd "${NESYS_DIR}"
python eval_webshop_agent.py \
  --agent_model "${WEBSHOP_AGENT_MODEL}" \
  --transition_base_model "${WEBSHOP_TRANSITION_BASE}" \
  --transition_adapter "${WEBSHOP_TRANSITION_ADAPTER}" \
  --rule_file "${WEBSHOP_RULE_FILE}" \
  --modes wm_rerank \
  --num_sessions "${WEBSHOP_NUM_SESSIONS}" \
  --session_offset "${WEBSHOP_SESSION_OFFSET}" \
  --max_steps "${WEBSHOP_MAX_STEPS}" \
  --top_k_candidates "${WEBSHOP_TOP_K_CANDIDATES}" \
  --top_k_states_context "${WEBSHOP_TOP_K_STATES_CONTEXT}" \
  --rule_scale 1.0 \
  --agent_temperature 0.2 \
  --transition_temperature 0.8 \
  --transition_top_p 0.95 \
  --seed "${WEBSHOP_SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  "${WEBSHOP_CACHE_ARGS[@]}" \
  "$@"

