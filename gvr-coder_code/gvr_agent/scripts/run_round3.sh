#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="xxx"
AGENT_DIR="${BASE_DIR}/gvr_agent"
CONFIG="${BASE_DIR}/config.yaml"

source "${AGENT_DIR}/scripts/common.sh"

OUT_DIR="${AGENT_DIR}/cover3w_rp3"
mkdir -p "${OUT_DIR}"

# =======================
# Repair 3
# =======================
run_repair_until_complete \
  "${OUT_DIR}" \
  "${AGENT_DIR}/3w_critic3.json" \
  "${OUT_DIR}/3w_rp3.jsonl" \
  "${CONFIG}" \
  "true" \
  50

# # =======================
# # Extract rp3 -> 3w_rp3.json
# # =======================
python "${AGENT_DIR}/extract_rpcode.py" \
  --input "${OUT_DIR}/3w_rp3.jsonl" \
  --output "${AGENT_DIR}/3w_rp3.json" \
  --output-dir "${OUT_DIR}"

# =======================
# Critic 4
# =======================
run_critic_until_complete \
  "${OUT_DIR}" \
  "${AGENT_DIR}/3w_rp3.json" \
  "${OUT_DIR}/3w_critic4.jsonl" \
  "${CONFIG}" \
  "${OUT_DIR}/eval_output_critic4.xlsx" \
  50

# =======================
# Extract critic4 -> 3w_critic4.json
# =======================
python "${AGENT_DIR}/extract_ap.py" \
  --input "${OUT_DIR}/3w_critic4.jsonl" \
  --output "${AGENT_DIR}/3w_critic4.json"
