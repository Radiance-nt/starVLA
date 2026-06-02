#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/hwfile/linzhanhui/projects/starVLA}"
PAYLOAD_SCRIPT="${PAYLOAD_SCRIPT:-${PROJECT_DIR}/examples/LIBERO/train_files/run_libero_clean_checked_payload.sh}"
STATE_DIR="${STATE_DIR:-/mnt/petrelfs/linzhanhui/runs_inspect/starVLA/watch_libero_clean_checked_generic_8g}"
PARTITION="${PARTITION:-eailab_link}"
ACCOUNT="${ACCOUNT:-research}"
TIME_LIMIT="${TIME_LIMIT:-5-00:00:00}"
WANDB_PROJECT="${WANDB_PROJECT:-starVLA_LIBERO}"
WANDB_ENTITY="${WANDB_ENTITY:-radiance}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/petrelfs/linzhanhui/runs_inspect/starVLA}"
PENDING_TIMEOUT_SECONDS="${PENDING_TIMEOUT_SECONDS:-1800}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-3000}"
DATA_MIX="${DATA_MIX:-libero_goal}"
CONFIG_YAML="${CONFIG_YAML:-examples/LIBERO/train_files/starvla_cotrain_libero.yaml}"
FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenOFT}"
BASE_VLM="${BASE_VLM:-/mnt/petrelfs/linzhanhui/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
ACCEL_CONFIG="${ACCEL_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-}"
ACTION_GOAL_LANG_PROB="${ACTION_GOAL_LANG_PROB:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-}"
EVAL_INTERVAL="${EVAL_INTERVAL:-}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-}"
TRAINER_IS_RESUME="${TRAINER_IS_RESUME:-false}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
RELOAD_MODULES="${RELOAD_MODULES:-}"
CONDA_BASE="${CONDA_BASE:-/mnt/petrelfs/linzhanhui/miniconda3}"
TOPOLOGY_TAG="${TOPOLOGY_TAG:-${NNODES}n$((NNODES * GPUS_PER_NODE))g}"
RUN_SUFFIX="${RUN_SUFFIX:-libero_goal_qwen3oft_${ATTN_IMPLEMENTATION}_pd${PER_DEVICE_BS}_ga${GRAD_ACCUM}_${TOPOLOGY_TAG}_cleanchecked}"
RUN_ID="${RUN_ID:-}"
JOB_TAG="${JOB_TAG:-${DATA_MIX}}"
JOB_NAME="${JOB_NAME:-lib-${JOB_TAG}-b${PER_DEVICE_BS}-ga${GRAD_ACCUM}-${TOPOLOGY_TAG}}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
DISABLE_WANDB_PROXY="${DISABLE_WANDB_PROXY:-0}"
WANDB_HTTP_PROXY="${WANDB_HTTP_PROXY:-}"
WANDB_HTTPS_PROXY="${WANDB_HTTPS_PROXY:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}"

mkdir -p "${STATE_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2
}

training_progress_detected() {
  local launch_log="$1"
  if grep -E 'Step [0-9]+, Loss:' "${launch_log}" >/dev/null 2>&1; then
    return 0
  fi
  if grep -E 'data_times=|model_times=' "${launch_log}" >/dev/null 2>&1; then
    return 0
  fi
  if grep -E 'wandb:.*Run history:' "${launch_log}" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

wandb_local_detected() {
  local run_dir="$1"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    return 1
  fi
  if find "${run_dir}/wandb" -maxdepth 2 -type d -name 'run-*' -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi
  if find "${run_dir}/wandb" -maxdepth 2 -type f -name 'wandb-metadata.json' -print -quit 2>/dev/null | grep -q .; then
    return 0
  fi
  return 1
}

cancel_stale_jobs() {
  local stale_ids
  stale_ids="$(
    squeue -u "${USER}" -h -o '%A %j %T' \
      | awk -v job_name="${JOB_NAME}" '$2 == job_name && ($3 == "PENDING" || $3 == "RUNNING") {print $1}' \
      | tr '\n' ' '
  )"
  if [[ -n "${stale_ids// }" ]]; then
    log "Cancelling stale jobs: ${stale_ids}"
    scancel ${stale_ids} || true
    sleep 5
  fi
}

launch_job() {
  local launch_log="${STATE_DIR}/launch_${TOPOLOGY_TAG}_$(date +%Y%m%d_%H%M%S).log"

  log "Launching ${JOB_NAME} via clean-checked salloc"
  if [[ -n "${EXCLUDE_NODES}" ]]; then
    nohup salloc \
      -p "${PARTITION}" \
      -A "${ACCOUNT}" \
      --time "${TIME_LIMIT}" \
      --exclude "${EXCLUDE_NODES}" \
      --exclusive \
      -N "${NNODES}" \
      --ntasks-per-node=1 \
      --gres "gpu:${GPUS_PER_NODE}" \
      --cpus-per-task "${CPUS_PER_TASK}" \
      --mem=0 \
      --job-name "${JOB_NAME}" \
      env \
        PROJECT_DIR="${PROJECT_DIR}" \
        PAYLOAD_SCRIPT="${PAYLOAD_SCRIPT}" \
        RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
        WANDB_PROJECT="${WANDB_PROJECT}" \
        WANDB_ENTITY="${WANDB_ENTITY}" \
        WANDB_HTTP_PROXY="${WANDB_HTTP_PROXY}" \
        WANDB_HTTPS_PROXY="${WANDB_HTTPS_PROXY}" \
        WANDB_RUN_ID="${WANDB_RUN_ID}" \
        WANDB_RESUME="${WANDB_RESUME}" \
        DISABLE_WANDB_PROXY="${DISABLE_WANDB_PROXY}" \
        GPUS_PER_NODE="${GPUS_PER_NODE}" \
        TOPOLOGY_TAG="${TOPOLOGY_TAG}" \
        DATA_MIX="${DATA_MIX}" \
        CONFIG_YAML="${CONFIG_YAML}" \
        FRAMEWORK_NAME="${FRAMEWORK_NAME}" \
        BASE_VLM="${BASE_VLM}" \
        ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
        ACCEL_CONFIG="${ACCEL_CONFIG}" \
        CONDA_BASE="${CONDA_BASE}" \
        PER_DEVICE_BS="${PER_DEVICE_BS}" \
        GRAD_ACCUM="${GRAD_ACCUM}" \
        MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
        NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS}" \
        ACTION_GOAL_LANG_PROB="${ACTION_GOAL_LANG_PROB}" \
        SAVE_INTERVAL="${SAVE_INTERVAL}" \
        LOGGING_FREQUENCY="${LOGGING_FREQUENCY}" \
        EVAL_INTERVAL="${EVAL_INTERVAL}" \
        SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
        TRAINER_IS_RESUME="${TRAINER_IS_RESUME}" \
        PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}" \
        RELOAD_MODULES="${RELOAD_MODULES}" \
        RUN_ID="${RUN_ID}" \
        RUN_SUFFIX="${RUN_SUFFIX}" \
        bash "${PAYLOAD_SCRIPT}" \
      >"${launch_log}" 2>&1 &
  else
    nohup salloc \
      -p "${PARTITION}" \
      -A "${ACCOUNT}" \
      --time "${TIME_LIMIT}" \
      --exclusive \
      -N "${NNODES}" \
      --ntasks-per-node=1 \
      --gres "gpu:${GPUS_PER_NODE}" \
      --cpus-per-task "${CPUS_PER_TASK}" \
      --mem=0 \
      --job-name "${JOB_NAME}" \
      env \
        PROJECT_DIR="${PROJECT_DIR}" \
        PAYLOAD_SCRIPT="${PAYLOAD_SCRIPT}" \
        RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
        WANDB_PROJECT="${WANDB_PROJECT}" \
        WANDB_ENTITY="${WANDB_ENTITY}" \
        WANDB_HTTP_PROXY="${WANDB_HTTP_PROXY}" \
        WANDB_HTTPS_PROXY="${WANDB_HTTPS_PROXY}" \
        WANDB_RUN_ID="${WANDB_RUN_ID}" \
        WANDB_RESUME="${WANDB_RESUME}" \
        DISABLE_WANDB_PROXY="${DISABLE_WANDB_PROXY}" \
        GPUS_PER_NODE="${GPUS_PER_NODE}" \
        TOPOLOGY_TAG="${TOPOLOGY_TAG}" \
        DATA_MIX="${DATA_MIX}" \
        CONFIG_YAML="${CONFIG_YAML}" \
        FRAMEWORK_NAME="${FRAMEWORK_NAME}" \
        BASE_VLM="${BASE_VLM}" \
        ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
        ACCEL_CONFIG="${ACCEL_CONFIG}" \
        CONDA_BASE="${CONDA_BASE}" \
        PER_DEVICE_BS="${PER_DEVICE_BS}" \
        GRAD_ACCUM="${GRAD_ACCUM}" \
        MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
        NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS}" \
        ACTION_GOAL_LANG_PROB="${ACTION_GOAL_LANG_PROB}" \
        SAVE_INTERVAL="${SAVE_INTERVAL}" \
        LOGGING_FREQUENCY="${LOGGING_FREQUENCY}" \
        EVAL_INTERVAL="${EVAL_INTERVAL}" \
        SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
        TRAINER_IS_RESUME="${TRAINER_IS_RESUME}" \
        PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT}" \
        RELOAD_MODULES="${RELOAD_MODULES}" \
        RUN_ID="${RUN_ID}" \
        RUN_SUFFIX="${RUN_SUFFIX}" \
        bash "${PAYLOAD_SCRIPT}" \
      >"${launch_log}" 2>&1 &
  fi

  echo "${launch_log}"
}

wait_for_job_id() {
  local launch_log="$1"
  local timeout_seconds="${2:-300}"
  local waited=0
  while (( waited < timeout_seconds )); do
    local job_id
    job_id="$(
      grep -Eo '[Gg]ranted job allocation [0-9]+' "${launch_log}" 2>/dev/null \
        | awk '{print $4}' | tail -n1
    )"
    if [[ -n "${job_id}" ]]; then
      echo "${job_id}"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  return 1
}

monitor_job() {
  local job_id="$1"
  local launch_log="$2"
  local pending_budget="$3"
  local startup_budget="$4"
  local start_ts
  start_ts="$(date +%s)"
  local run_dir=""

  while true; do
    if [[ -z "${run_dir}" ]]; then
      run_dir="$(grep '^\[RUN_DIR\]' "${launch_log}" 2>/dev/null | tail -n1 | sed 's/^\[RUN_DIR\] //')"
      if [[ -n "${run_dir}" ]]; then
        log "Detected run dir: ${run_dir}"
      fi
    fi

    local wandb_ok=0
    if [[ -n "${run_dir}" ]] && wandb_local_detected "${run_dir}"; then
      wandb_ok=1
    fi

    local loss_ok=0
    if training_progress_detected "${launch_log}"; then
      loss_ok=1
    fi

    if [[ "${wandb_ok}" == "1" && "${loss_ok}" == "1" ]]; then
      log "Training is healthy: W&B local artifacts and loss logs are present for job ${job_id}"
      return 0
    fi

    local state
    state="$(squeue -j "${job_id}" -h -o '%T' | head -n1)"
    local now_ts
    now_ts="$(date +%s)"
    local elapsed=$((now_ts - start_ts))

    if [[ -z "${state}" ]]; then
      local final_state
      final_state="$(sacct -j "${job_id}" --format=State -n -P 2>/dev/null | head -n1 | tr -d '[:space:]')"
      log "Job ${job_id} left queue before healthy startup. Final state: ${final_state:-unknown}"
      return 1
    fi

    if [[ "${state}" == "PENDING" && "${elapsed}" -ge "${pending_budget}" ]]; then
      log "Job ${job_id} stayed pending for ${elapsed}s; cancelling"
      scancel "${job_id}" || true
      return 2
    fi

    if [[ "${state}" == "RUNNING" && "${elapsed}" -ge "${startup_budget}" ]]; then
      log "Job ${job_id} exceeded startup budget without W&B/loss; cancelling"
      scancel "${job_id}" || true
      return 1
    fi

    sleep 60
  done
}

main() {
  cd "${PROJECT_DIR}"
  cancel_stale_jobs

  while true; do
    local launch_log
    launch_log="$(launch_job)"
    log "Launch log: ${launch_log}"

    local job_id=""
    if ! job_id="$(wait_for_job_id "${launch_log}")"; then
      log "Failed to obtain allocation id from ${launch_log}; retrying"
      continue
    fi
    log "Allocation granted: job ${job_id}"

    local result=0
    if monitor_job "${job_id}" "${launch_log}" "${PENDING_TIMEOUT_SECONDS}" "${STARTUP_TIMEOUT_SECONDS}"; then
      log "Watchdog finished successfully for job ${job_id}"
      return 0
    else
      result="$?"
    fi

    if [[ "${result}" == "2" ]]; then
      log "Job ${job_id} stayed pending too long; relaunching"
    else
      log "Job ${job_id} failed to start cleanly; relaunching"
    fi
  done
}

main "$@"
