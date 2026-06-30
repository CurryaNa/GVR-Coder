#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

# bash "${DIR}/run_round0_critic.sh"
bash "${DIR}/run_round1.sh"
bash "${DIR}/run_round2.sh"
bash "${DIR}/run_round3.sh"

echo "✅ All rounds finished."
