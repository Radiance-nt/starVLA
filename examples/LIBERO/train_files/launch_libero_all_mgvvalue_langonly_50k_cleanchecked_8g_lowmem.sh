#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/hwfile/linzhanhui/projects/starVLA}"
WATCH_SCRIPT="${PROJECT_DIR}/examples/LIBERO/train_files/watch_libero_clean_checked_generic_8g.sh"

export PROJECT_DIR
export CONFIG_YAML="examples/LIBERO/train_files/starvla_cotrain_libero_all_mgvvalue_langonly_goalstartsplit_50k.yaml"
export DATA_MIX="libero_all"
export FRAMEWORK_NAME="QwenOFT"
export BASE_VLM="/mnt/petrelfs/linzhanhui/projects/starVLA/tmp/Qwen3-VL-4B-Instruct"
export RUN_ROOT_DIR="/mnt/petrelfs/linzhanhui/runs_inspect/starVLA"
export STATE_DIR="/mnt/petrelfs/linzhanhui/runs_inspect/starVLA/watch_libero_all_mgvvalue_langonly_50k_cleanchecked_8g_lowmem"

export PARTITION="eailab_link"
export ACCOUNT="research"
export TIME_LIMIT="5-00:00:00"
export NNODES="1"
export GPUS_PER_NODE="8"
export CPUS_PER_TASK="128"

# Lower host-memory pressure while keeping the same effective global batch size.
export PER_DEVICE_BS="2"
export GRAD_ACCUM="4"
export NUM_WORKERS="0"
export LOAD_ALL_DATA_FOR_TRAINING="false"
export MAX_TRAIN_STEPS="50000"
export NUM_WARMUP_STEPS="5000"
export ACTION_GOAL_LANG_PROB="1.0"
export SAVE_INTERVAL="5000"
export LOGGING_FREQUENCY="20"
export EVAL_INTERVAL="1000"
export SAVE_TOTAL_LIMIT="2"

export WANDB_PROJECT="starVLA_LIBERO_MGV"
export WANDB_ENTITY="radiance"
export DISABLE_WANDB_PROXY="0"
export WANDB_HTTP_PROXY="http://linzhanhui:WroVrZ2F6WSIO6BSpZQFTZxokukcYL2a9ufdIZnzD4G8HTzdaS4kamxiVA3l@10.1.20.51:23128"
export WANDB_HTTPS_PROXY="http://linzhanhui:WroVrZ2F6WSIO6BSpZQFTZxokukcYL2a9ufdIZnzD4G8HTzdaS4kamxiVA3l@10.1.20.51:23128"

export ATTN_IMPLEMENTATION="flash_attention_2"
export TOPOLOGY_TAG="1n8g"
export JOB_TAG="libero-all-mgvv50-lowmem"
export JOB_NAME="liball-mgvv50-b2-ga4-1n8g"
export RUN_SUFFIX="libero_all_mgvvalue_langonly_goalstartsplit_50k_qwen3oft_flash_attention_2_pd2_ga4_1n8g_cleanchecked"

cd "${PROJECT_DIR}"
exec bash "${WATCH_SCRIPT}"
