#!/usr/bin/env bash
set -u

# Run the two experiments independently on the same five-robot map. Override
# these for a multi-GPU host:
#   PHASE0_DEVICE=cuda:0 PHASE2_DEVICE=cuda:1 ./scripts/run_phase0_phase2.sh
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SEAC_DIR="${ROOT_DIR}/seac/seac"
RWARE_DIR="${ROOT_DIR}/robotic-warehouse"
PHASE0_DEVICE="${PHASE0_DEVICE:-cuda:0}"
PHASE2_DEVICE="${PHASE2_DEVICE:-cuda:0}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/results/parallel-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${RUN_DIR}/phase0" "${RUN_DIR}/phase2"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi

run_experiment() {
    local name="$1"
    local config="$2"
    local device="$3"
    local output_dir="$4"

    cd "${SEAC_DIR}"
    PYTHONPATH="${RWARE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        "${PYTHON_BIN}" train.py with "${config}" \
        algorithm.device="${device}" \
        save_dir="${output_dir}/models/{id}" \
        eval_dir="${output_dir}/video/{id}" \
        loss_dir="${output_dir}/loss/{id}" \
        >"${output_dir}/train.log" 2>&1
}

echo "Starting Phase 0 (original SEAC) and Phase 2 (routing SEAC)."
echo "Results: ${RUN_DIR}"

run_experiment phase0 rware_custom_5ag "${PHASE0_DEVICE}" "${RUN_DIR}/phase0" &
PHASE0_PID=$!
run_experiment phase2 rware_custom_5ag_routing "${PHASE2_DEVICE}" "${RUN_DIR}/phase2" &
PHASE2_PID=$!

set +e
wait "${PHASE0_PID}"
PHASE0_STATUS=$?
wait "${PHASE2_PID}"
PHASE2_STATUS=$?
set -e

echo "Phase 0 exit status: ${PHASE0_STATUS}"
echo "Phase 2 exit status: ${PHASE2_STATUS}"
echo "Logs: ${RUN_DIR}/phase0/train.log and ${RUN_DIR}/phase2/train.log"

if [[ "${PHASE0_STATUS}" -ne 0 || "${PHASE2_STATUS}" -ne 0 ]]; then
    exit 1
fi
