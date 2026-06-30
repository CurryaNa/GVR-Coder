#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="xxx"
AGENT_DIR="${BASE_DIR}/gvr_agent"
CONFIG="${BASE_DIR}/config.yaml"

source "${AGENT_DIR}/scripts/common.sh"

OUT_DIR="${AGENT_DIR}/cover3w_rp1"
mkdir -p "${OUT_DIR}"

run_repair_until_complete \
  "${OUT_DIR}" \
  "${AGENT_DIR}/critic1.json" \
  "${OUT_DIR}/3w_rp1.jsonl" \
  "${CONFIG}" \
  "true" \
  50

# Extract rp1 -> rp1.json
python "${AGENT_DIR}/extract_rpcode.py" \
  --input "${OUT_DIR}/3w_rp1.jsonl" \
  --output "${AGENT_DIR}/3w_rp1.json" \
  --output-dir "${OUT_DIR}"

# # Critic 2
run_critic_until_complete \
  "${OUT_DIR}" \
  "${AGENT_DIR}/3w_rp1.json" \
  "${OUT_DIR}/3w_critic2.jsonl" \
  "${CONFIG}" \
  "${OUT_DIR}/eval_output_critic2.xlsx" \
  50

# Extract critic2 -> critic2.json
python "${AGENT_DIR}/extract_ap.py" \
  --input "${OUT_DIR}/3w_critic2.jsonl" \
  --output "${AGENT_DIR}/3w_critic2.json"
