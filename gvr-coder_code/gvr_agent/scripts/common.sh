#!/usr/bin/env bash
set -euo pipefail

json_len() {
  local json_path="$1"
  python - <<PY
import json
p="${json_path}"
with open(p,"r",encoding="utf-8") as f:
    d=json.load(f)
print(len(d) if isinstance(d,list) else 1)
PY
}

jsonl_len() {
  local jsonl_path="$1"
  if [[ ! -f "$jsonl_path" ]]; then
    echo 0
    return
  fi
  python - <<PY
p="${jsonl_path}"
n=0
with open(p,"r",encoding="utf-8") as f:
    for line in f:
        if line.strip():
            n+=1
print(n)
PY
}

needs_repair_true_len() {
  local json_path="$1"
  python - <<PY
import json
p="${json_path}"
with open(p,"r",encoding="utf-8") as f:
    d=json.load(f)
if not isinstance(d,list):
    d=[d]
print(sum(1 for x in d if x.get("needs_repair") is True))
PY
}

run_critic_until_complete() {
  local output_dir="$1"
  local data_path="$2"
  local predict_path="$3"
  local config_path="$4"
  local excel_path="$5"
  local max_attempts="${6:-50}"

  local expect
  expect="$(json_len "$data_path")"

  echo "[critic] expect=${expect} data_path=${data_path}"
  for ((i=1; i<=max_attempts; i++)); do
    local got
    got="$(jsonl_len "$predict_path")"
    echo "[critic] attempt=$i got=${got} predict_path=${predict_path}"

    if [[ "$got" -ge "$expect" ]]; then
      echo "[critic] done (got >= expect)"
      return 0
    fi

    python "${output_dir%/}/../verify.py" \
      --output-dir "${output_dir}" \
      --data-path "${data_path}" \
      --predict-path "${predict_path}" \
      --config-path "${config_path}" \
      --enable-infer \
      --excel-path "${excel_path}"

    sleep 1
  done

  echo "[critic] ERROR: still incomplete after ${max_attempts} attempts" >&2
  return 1
}

run_repair_until_complete() {
  local output_dir="$1"
  local input_critic_json="$2"
  local predict_path="$3"
  local config_path="$4"
  local render="${5:-true}"
  local max_attempts="${6:-50}"

  local expect
  expect="$(needs_repair_true_len "$input_critic_json")"

  echo "[repair] expect(needs_repair=true)=${expect} input=${input_critic_json}"

  if [[ "$expect" -eq 0 ]]; then
    echo "[repair] no items need repair, skip"
    return 0
  fi

  for ((i=1; i<=max_attempts; i++)); do
    local got
    got="$(jsonl_len "$predict_path")"
    echo "[repair] attempt=$i got=${got} predict_path=${predict_path}"

    if [[ "$got" -ge "$expect" ]]; then
      echo "[repair] done (got >= expect)"
      return 0
    fi

    local render_flag=""
    if [[ "$render" == "true" ]]; then
      render_flag="--render"
    fi

    python "${output_dir%/}/../repair.py" \
      --input-critic-json "${input_critic_json}" \
      --output-dir "${output_dir}" \
      --predict-path "${predict_path}" \
      --config-path "${config_path}" \
      --enable-infer \
      ${render_flag}

    sleep 1
  done

  echo "[repair] ERROR: still incomplete after ${max_attempts} attempts" >&2
  return 1
}
