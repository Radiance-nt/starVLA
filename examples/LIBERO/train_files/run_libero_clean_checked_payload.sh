#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/hwfile/linzhanhui/projects/starVLA}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/starVLA:${PYTHONPATH:-}"

export HOME="${HOME:-/mnt/petrelfs/linzhanhui}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_HTTP_PROXY="${WANDB_HTTP_PROXY:-}"
export WANDB_HTTPS_PROXY="${WANDB_HTTPS_PROXY:-}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-}"
export WANDB_RESUME="${WANDB_RESUME:-}"
WANDB_RUN_ID_VALUE="${WANDB_RUN_ID}"
WANDB_RESUME_VALUE="${WANDB_RESUME}"

if [[ -z "${WANDB_RUN_ID}" ]]; then
  unset WANDB_RUN_ID
fi
if [[ -z "${WANDB_RESUME}" ]]; then
  unset WANDB_RESUME
fi

configure_proxy_env() {
  if [[ "${DISABLE_WANDB_PROXY:-0}" == "1" ]]; then
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
    return 0
  fi
  if [[ -n "${WANDB_HTTP_PROXY:-}" ]]; then
    export http_proxy="${WANDB_HTTP_PROXY}"
    export HTTP_PROXY="${WANDB_HTTP_PROXY}"
  fi
  if [[ -n "${WANDB_HTTPS_PROXY:-}" ]]; then
    export https_proxy="${WANDB_HTTPS_PROXY}"
    export HTTPS_PROXY="${WANDB_HTTPS_PROXY}"
  fi
}

configure_proxy_env

load_wandb_env() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    return 0
  fi
  local netrc_path="${HOME}/.netrc"
  if [[ ! -f "${netrc_path}" ]]; then
    return 1
  fi
  local key
  key="$(
    awk '
      $1=="machine" && $2=="api.wandb.ai" {found=1; next}
      found && $1=="password" {print $2; exit}
    ' "${netrc_path}"
  )"
  if [[ -n "${key}" ]]; then
    export WANDB_API_KEY="${key}"
    return 0
  fi
  return 1
}

CONDA_BASE="${CONDA_BASE:-/mnt/petrelfs/linzhanhui/miniconda3}"
set +u
source "${CONDA_BASE}/bin/activate"
conda activate starVLA
set -u

load_wandb_env || {
  echo "[ERROR] WANDB_API_KEY is not set and could not be loaded from ${HOME}/.netrc"
  exit 21
}

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_4,mlx5_5}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-1440}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenOFT}"
BASE_VLM="${BASE_VLM:-/mnt/petrelfs/linzhanhui/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}"
CONFIG_YAML="${CONFIG_YAML:-examples/LIBERO/train_files/starvla_cotrain_libero.yaml}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${PROJECT_DIR}/playground/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_goal}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
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
WANDB_PROJECT="${WANDB_PROJECT:-starVLA_LIBERO}"
WANDB_ENTITY="${WANDB_ENTITY:-radiance}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/petrelfs/linzhanhui/runs_inspect/starVLA}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TOPOLOGY_TAG="${TOPOLOGY_TAG:-${SLURM_NNODES}n$((SLURM_NNODES * GPUS_PER_NODE))g}"
RUN_SUFFIX="${RUN_SUFFIX:-libero_goal_mgv_qwen3oft_${ATTN_IMPLEMENTATION}_pd${PER_DEVICE_BS}_ga${GRAD_ACCUM}_${TOPOLOGY_TAG}_cleanchecked}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID}_${RUN_SUFFIX}}"

TOTAL_GPUS="$((SLURM_NNODES * GPUS_PER_NODE))"
TOTAL_BATCH_SIZE="$((PER_DEVICE_BS * TOTAL_GPUS * GRAD_ACCUM))"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
ACCEL_CONFIG="${ACCEL_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HOME}/.triton/autotune"
cp "$0" "${OUTPUT_DIR}/"
config_copy_target="${OUTPUT_DIR}/$(basename "${CONFIG_YAML}")"
if [[ "$(realpath "${CONFIG_YAML}")" != "$(realpath "${config_copy_target}" 2>/dev/null || printf '%s' "${config_copy_target}")" ]]; then
  cp "${CONFIG_YAML}" "${OUTPUT_DIR}/"
fi

EXTRA_TRAIN_ARGS_STR=""
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
  EXTRA_TRAIN_ARGS_STR+=" --trainer.pretrained_checkpoint $(printf '%q' "${PRETRAINED_CHECKPOINT}")"
fi
if [[ -n "${RELOAD_MODULES}" ]]; then
  EXTRA_TRAIN_ARGS_STR+=" --trainer.reload_modules $(printf '%q' "${RELOAD_MODULES}")"
fi
if [[ -n "${SAVE_TOTAL_LIMIT}" ]]; then
  EXTRA_TRAIN_ARGS_STR+=" --trainer.save_total_limit $(printf '%q' "${SAVE_TOTAL_LIMIT}")"
fi
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
  EXTRA_TRAIN_ARGS_STR+=" --trainer.max_train_steps $(printf '%q' "${MAX_TRAIN_STEPS}")"
fi
if [[ -n "${NUM_WARMUP_STEPS}" ]]; then
  EXTRA_TRAIN_ARGS_STR+=" --trainer.num_warmup_steps $(printf '%q' "${NUM_WARMUP_STEPS}")"
fi
if [[ -n "${ACTION_GOAL_LANG_PROB}" ]]; then
  EXTRA_TRAIN_ARGS_STR+=" --framework.mgv.action_goal_lang_prob $(printf '%q' "${ACTION_GOAL_LANG_PROB}")"
fi
if [[ -n "${SAVE_INTERVAL}" ]]; then
  EXTRA_TRAIN_ARGS_STR+=" --trainer.save_interval $(printf '%q' "${SAVE_INTERVAL}")"
fi
if [[ -n "${LOGGING_FREQUENCY}" ]]; then
  EXTRA_TRAIN_ARGS_STR+=" --trainer.logging_frequency $(printf '%q' "${LOGGING_FREQUENCY}")"
fi
if [[ -n "${EVAL_INTERVAL}" ]]; then
  EXTRA_TRAIN_ARGS_STR+=" --trainer.eval_interval $(printf '%q' "${EVAL_INTERVAL}")"
fi

git rev-parse HEAD > "${OUTPUT_DIR}/git_rev.txt"
git status --short > "${OUTPUT_DIR}/git_status.txt"
scontrol show job "${SLURM_JOB_ID}" > "${OUTPUT_DIR}/scontrol_job.txt"
cat > "${OUTPUT_DIR}/batch_math.txt" <<EOF
per_device_batch_size=${PER_DEVICE_BS}
total_gpus=${TOTAL_GPUS}
gradient_accumulation_steps=${GRAD_ACCUM}
total_batch_size=${TOTAL_BATCH_SIZE}
EOF

echo "[RUN_DIR] ${OUTPUT_DIR}"
echo "[RUN_ID] ${RUN_ID}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR}"
echo "[INFO] MASTER_PORT=${MASTER_PORT}"
echo "[INFO] TOTAL_GPUS=${TOTAL_GPUS}"
echo "[INFO] TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}"
echo "[INFO] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[INFO] TRAINER_IS_RESUME=${TRAINER_IS_RESUME}"
echo "[INFO] PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}"
echo "[INFO] NUM_WARMUP_STEPS=${NUM_WARMUP_STEPS:-<yaml>}"
echo "[INFO] ACTION_GOAL_LANG_PROB=${ACTION_GOAL_LANG_PROB:-<yaml>}"
echo "[INFO] SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"

if ! srun --jobid "${SLURM_JOB_ID}" --ntasks "${SLURM_NNODES}" --ntasks-per-node=1 bash --noprofile --norc -c '
set -euo pipefail
host=$(hostname)
scope_args=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  scope_args=(--id "${CUDA_VISIBLE_DEVICES}")
  echo "[INFO] ${host}: checking allocated GPUs ${CUDA_VISIBLE_DEVICES}"
fi
for attempt in $(seq 1 10); do
  pids=$(nvidia-smi "${scope_args[@]}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk "NF" | sort -u)
  if [[ -n "${pids}" ]]; then
    while read -r pid; do
      [[ -n "${pid}" ]] || continue
      proc_user=$(ps -o user= -p "${pid}" 2>/dev/null | awk "{print \$1}" || true)
      [[ -n "${proc_user}" ]] || continue
      if [[ "${proc_user}" == "${USER}" ]]; then
        echo "[CLEANUP] ${host}: killing stale user GPU pid ${pid} (attempt ${attempt})"
        kill -9 "${pid}" 2>/dev/null || true
      fi
    done <<< "${pids}"
  fi

  foreign_apps=()
  own_apps=()
  apps=$(nvidia-smi "${scope_args[@]}" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk "NF")
  if [[ -n "${apps}" ]]; then
    while IFS= read -r app; do
      [[ -n "${app}" ]] || continue
      pid=$(echo "${app}" | cut -d, -f1 | tr -d " ")
      proc_user=$(ps -o user= -p "${pid}" 2>/dev/null | awk "{print \$1}" || true)
      if [[ -z "${proc_user}" ]]; then
        own_apps+=("${app}")
      elif [[ "${proc_user}" == "${USER}" ]]; then
        own_apps+=("${app}")
      else
        foreign_apps+=("${app}")
      fi
    done <<< "${apps}"
  fi

  if (( ${#foreign_apps[@]} > 0 )); then
    echo "[DIRTY] ${host}: foreign GPU compute processes detected"
    printf "%s\n" "${foreign_apps[@]}"
    exit 42
  fi

  if (( ${#own_apps[@]} == 0 )); then
    echo "[CLEAN] ${host}"
    exit 0
  fi

  sleep 3
done
echo "[DIRTY] ${host}: surviving user GPU compute processes detected after cleanup retries"
printf "%s\n" "${own_apps[@]}"
exit 42
'; then
  echo "[ERROR] Clean-check preflight failed"
  exit 42
fi

cat > "${OUTPUT_DIR}/launch_rank.sh" <<EOF
#!/bin/bash
set -euo pipefail

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/starVLA:\${PYTHONPATH:-}"
export HOME="${HOME}"
export WANDB_MODE="${WANDB_MODE}"
export WANDB_HTTP_PROXY="${WANDB_HTTP_PROXY}"
export WANDB_HTTPS_PROXY="${WANDB_HTTPS_PROXY}"
export WANDB_RUN_ID="${WANDB_RUN_ID_VALUE}"
export WANDB_RESUME="${WANDB_RESUME_VALUE}"
if [[ -z "\${WANDB_RUN_ID}" ]]; then
  unset WANDB_RUN_ID
fi
if [[ -z "\${WANDB_RESUME}" ]]; then
  unset WANDB_RESUME
fi

configure_proxy_env() {
  if [[ "\${DISABLE_WANDB_PROXY:-0}" == "1" ]]; then
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
    return 0
  fi
  if [[ -n "\${WANDB_HTTP_PROXY:-}" ]]; then
    export http_proxy="\${WANDB_HTTP_PROXY}"
    export HTTP_PROXY="\${WANDB_HTTP_PROXY}"
  fi
  if [[ -n "\${WANDB_HTTPS_PROXY:-}" ]]; then
    export https_proxy="\${WANDB_HTTPS_PROXY}"
    export HTTPS_PROXY="\${WANDB_HTTPS_PROXY}"
  fi
}

configure_proxy_env

load_wandb_env() {
  if [[ -n "\${WANDB_API_KEY:-}" ]]; then
    return 0
  fi
  local netrc_path="\${HOME}/.netrc"
  if [[ ! -f "\${netrc_path}" ]]; then
    return 1
  fi
  local key
  key="\$(
    awk '
      \$1=="machine" && \$2=="api.wandb.ai" {found=1; next}
      found && \$1=="password" {print \$2; exit}
    ' "\${netrc_path}"
  )"
  if [[ -n "\${key}" ]]; then
    export WANDB_API_KEY="\${key}"
    return 0
  fi
  return 1
}

set +u
source "${CONDA_BASE}/bin/activate"
conda activate starVLA
set -u
load_wandb_env

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
export NCCL_IB_HCA="${NCCL_IB_HCA}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"

echo "[INFO] Host=\$(hostname) SLURM_PROCID=\${SLURM_PROCID}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --machine_rank "\${SLURM_PROCID}" \
  --num_machines "${SLURM_NNODES}" \
  --num_processes "${TOTAL_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.qwenvl.attn_implementation "${ATTN_IMPLEMENTATION}" \
  --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vla_data.action_type delta_qpos \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BS}" \
  --datasets.vla_data.num_workers 1 \
  --datasets.vla_data.prefetch_factor 1 \
  --datasets.vla_data.persistent_workers false \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --trainer.is_resume "${TRAINER_IS_RESUME}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}"${EXTRA_TRAIN_ARGS_STR}
EOF

chmod +x "${OUTPUT_DIR}/launch_rank.sh"
srun --jobid "${SLURM_JOB_ID}" --ntasks "${SLURM_NNODES}" --ntasks-per-node=1 "${OUTPUT_DIR}/launch_rank.sh"
