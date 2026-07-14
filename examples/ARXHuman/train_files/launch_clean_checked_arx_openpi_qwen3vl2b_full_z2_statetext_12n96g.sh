#!/bin/bash

set -euo pipefail

PARTITION=${PARTITION:-wam_critic}
TIME_LIMIT=${TIME_LIMIT:-10-00:00:00}
NNODES=${NNODES:-12}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-128}
JOB_NAME=${JOB_NAME:-arx_openpi_qwen3vl2b_full_z2_pd16_12n96g_cc}
REPO_ROOT=${REPO_ROOT:-/mnt/inspurfs/vla_coop/linzhanhui/projects/starVLA-ARX}
PAYLOAD=${PAYLOAD:-${REPO_ROOT}/examples/ARXHuman/train_files/run_clean_checked_arx_openpi_qwen3vl2b_full_z2_statetext_12n96g.sh}
LOG_DIR=${LOG_DIR:-/mnt/petrelfs/linzhanhui/slurm_logs}
LAUNCH_LOG=${LAUNCH_LOG:-${LOG_DIR}/${JOB_NAME}_launch_$(date +%Y%m%d_%H%M%S).log}

mkdir -p "${LOG_DIR}"
chmod +x "${PAYLOAD}"

echo "[launch] partition=${PARTITION}"
echo "[launch] nodes=${NNODES} gpus_per_node=${GPUS_PER_NODE}"
echo "[launch] payload=${PAYLOAD}"
echo "[launch] log=${LAUNCH_LOG}"

nohup salloc \
  -p "${PARTITION}" \
  --time "${TIME_LIMIT}" \
  --exclusive \
  -N "${NNODES}" \
  --ntasks-per-node=1 \
  --gres="gpu:${GPUS_PER_NODE}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --job-name "${JOB_NAME}" \
  bash -c "${PAYLOAD}" \
  >"${LAUNCH_LOG}" 2>&1 &

launcher_pid=$!
echo "[launch] launcher_pid=${launcher_pid}"

job_id=""
for _ in $(seq 1 60); do
  if [[ -f "${LAUNCH_LOG}" ]]; then
    job_id=$(grep -o 'job allocation [0-9]\+' "${LAUNCH_LOG}" | awk '{print $3}' | tail -n 1 || true)
    if [[ -n "${job_id}" ]]; then
      break
    fi
    if grep -qiE 'error|failed|invalid|not available|unable' "${LAUNCH_LOG}"; then
      break
    fi
  fi
  sleep 2
done

if [[ -n "${job_id}" ]]; then
  echo "[launch] job_id=${job_id}"
else
  echo "[launch] job_id not allocated yet; inspect ${LAUNCH_LOG}"
fi

echo "[launch] tail ${LAUNCH_LOG}:"
tail -n 40 "${LAUNCH_LOG}" || true
