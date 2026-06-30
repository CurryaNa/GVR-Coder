#!/usr/bin/env bash
set -euo pipefail



BASE_DIR="xxx"
AGENT_DIR="${BASE_DIR}/gvr_agent"
CONFIG="${BASE_DIR}/config.yaml"

source "${AGENT_DIR}/scripts/common.sh"

OUT_DIR="${AGENT_DIR}/cover3w_rp2"
mkdir -p "${OUT_DIR}"

# =======================
# Repair 2
# =======================
run_repair_until_complete \
  "${OUT_DIR}" \
  "${AGENT_DIR}/3w_critic2.json" \
  "${OUT_DIR}/3w_rp2.jsonl" \
  "${CONFIG}" \
  "true" \
  100

# # =======================
# Extract rp2 -> 5000_rp2.json
# =======================
python "${AGENT_DIR}/extract_rpcode.py" \
  --input "${OUT_DIR}/3w_rp2.jsonl" \
  --output "${AGENT_DIR}/3w_rp2.json" \
  --output-dir "${OUT_DIR}"

# =======================
# Critic 3
# =======================
run_critic_until_complete \
  "${OUT_DIR}" \
  "${AGENT_DIR}/3w_rp2.json" \
  "${OUT_DIR}/3w_critic3.jsonl" \
  "${CONFIG}" \
  "${OUT_DIR}/eval_output_critic3.xlsx" \
  50

# =======================
# Extract critic3 -> 5000_critic3.json
# =======================
python "${AGENT_DIR}/extract_ap.py" \
  --input "${OUT_DIR}/3w_critic3.jsonl" \
  --output "${AGENT_DIR}/3w_critic3.json"
