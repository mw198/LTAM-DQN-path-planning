#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi
OUTPUT_ROOT="${PROJECT_DIR}/results_ltam_adjusted_gpu0"
LOG_FILE="${OUTPUT_ROOT}/ltam_adjusted_gpu0.log"
PID_FILE="${OUTPUT_ROOT}/ltam_adjusted_gpu0.pid"

cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

if [[ "${1:-}" == "--help-check" ]]; then
  exec "${PYTHON_BIN}" -u -m src.run_experiments --help
fi

mkdir -p "${OUTPUT_ROOT}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "already running pid=${old_pid}"
    echo "log=${LOG_FILE}"
    exit 0
  fi
fi

nohup "${PYTHON_BIN}" -u -m src.run_experiments \
  --stage formal \
  --methods DQN,DQN-AM,LTAM-DQN,LTAM-DQN-Adj \
  --train-curriculum mixed-hard \
  --validation-profile main-dense \
  --train-episodes 3000 \
  --train-seeds 0,1,2,3,4 \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda \
  > "${LOG_FILE}" 2>&1 &

pid="$!"
echo "${pid}" > "${PID_FILE}"
echo "started pid=${pid}"
echo "gpu=0"
echo "output=${OUTPUT_ROOT}"
echo "log=${LOG_FILE}"
