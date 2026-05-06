#!/bin/bash

set -euo pipefail

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenOFT}"
FREEZE_MODULE_LIST="${FREEZE_MODULE_LIST:-}"
BASE_VLM="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
CONFIG_YAML="${CONFIG_YAML:-./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml}"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-$HOME/datasets}"
DATA_MIX="${DATA_MIX:-robotwin_unified_50}"
LEROBOT_VERSION="${LEROBOT_VERSION:-v3.0}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./results/Checkpoints}"
RUN_ID="${RUN_ID:-$(date +%m%d)_${DATA_MIX}_qwen3OFT_unified}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-150000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-starVLA_Robotwin}"
WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"

output_dir="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.data_root_dir "${ROBOTWIN_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.lerobot_version "${LEROBOT_VERSION}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BS}" \
  --trainer.freeze_modules "${FREEZE_MODULE_LIST}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}"
