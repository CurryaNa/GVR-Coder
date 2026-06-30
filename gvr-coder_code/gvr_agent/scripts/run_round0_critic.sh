#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="xxx"
AGENT_DIR="${BASE_DIR}/cr_agent"
CONFIG="${BASE_DIR}/config.yaml"

OUT_DIR="${AGENT_DIR}/cover3w_rp1"
mkdir -p "${OUT_DIR}"

python "${AGENT_DIR}/critic.py" \
  --output-dir "${OUT_DIR}" \
  --data-path "${BASE_DIR}/cover.json" \
  --predict-path "${OUT_DIR}/critic1.jsonl" \
  --config-path "${CONFIG}" \
  --enable-infer \
  --excel-path "${OUT_DIR}/eval_output_critic1.xlsx"

python "${AGENT_DIR}/extract_ap.py" \
  --input "${OUT_DIR}/critic1.jsonl" \
  --output "${AGENT_DIR}/critic1.json"
