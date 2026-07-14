#!/bin/bash

set -euo pipefail

export HOME=/mnt/petrelfs/linzhanhui
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export TORCH_DIST_TIMEOUT_MINUTES=1440

export GPUS_PER_NODE=8
export TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))
export GRAD_ACCUM=1
export PER_DEVICE_BATCH_SIZE=16

export REPO_ROOT=/mnt/inspurfs/vla_coop/linzhanhui/projects/starVLA-ARX
export CONFIG_YAML=${REPO_ROOT}/examples/ARXHuman/train_files/starvla_train_arx_openpi_tube_human_v2_qwen3vl2b_pi_full_z2_statetext.yaml
export BASE_VLM=/mnt/inspurfs/vla_coop/public_model_2/Qwen3-VL-2B-Instruct
export RUN_ROOT_DIR=/mnt/inspurfs/vla_coop/linzhanhui/runs_inspect/starVLA
export RUN_ID=${SLURM_JOB_ID}_arx_openpi_tube_human_v2_qwen3vl2b_qwenpi_v3_full_z2_statetext_pd16_2n16g_cleanchecked
export RUN_DIR=${RUN_ROOT_DIR}/${RUN_ID}
export STARVLA_TRITON_CACHE_BASE=/tmp/starvla_triton_cache
export CONDA_ENV_DIR=/mnt/petrelfs/linzhanhui/miniconda3/envs/starVLA

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
ALLOCATED_NODELIST=$(IFS=,; echo "${ALLOCATED_NODES[*]}")
export NODELIST="${ALLOCATED_NODELIST}"

mkdir -p /mnt/petrelfs/linzhanhui/slurm_logs "${RUN_DIR}"
cd "${REPO_ROOT}"

source scripts/load_wandb_env.sh

cp "$0" "${RUN_DIR}/"
cp "${CONFIG_YAML}" "${RUN_DIR}/"
scontrol show job "${SLURM_JOB_ID}" > "${RUN_DIR}/scontrol_show_job.txt" || true
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status_short.txt"
{
  echo "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
  echo "total_gpus=${TOTAL_GPUS}"
  echo "gradient_accumulation_steps=${GRAD_ACCUM}"
  echo "total_batch_size=$((PER_DEVICE_BATCH_SIZE * TOTAL_GPUS * GRAD_ACCUM))"
  echo "base_vlm=${BASE_VLM}"
  echo "config_yaml=${CONFIG_YAML}"
  echo "nodes=${NODELIST}"
} > "${RUN_DIR}/batch_math.txt"

echo "RUN_DIR=${RUN_DIR}"
echo "NODELIST=${NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "SLURM_NNODES=${SLURM_NNODES} TOTAL_GPUS=${TOTAL_GPUS}"

for node in "${ALLOCATED_NODES[@]}"; do
  echo "[clean-check] ${node}: clean_process"
  swatch -n "${node}" clean_process
  echo "[clean-check] ${node}: nv"
  swatch -n "${node}" nv
  echo "[clean-check] ${node}: check_empty_run"
  swatch -n "${node}" check_empty_run
done

srun --jobid "${SLURM_JOB_ID}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 env -u SHELLOPTS bash -c '
  set -eo pipefail
  cd "${REPO_ROOT}"
  source /mnt/petrelfs/linzhanhui/miniconda3/bin/activate
  conda activate "${CONDA_ENV_DIR}"
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

  echo "Host=$(hostname) SLURM_PROCID=${SLURM_PROCID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

  accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --machine_rank "${SLURM_PROCID}" \
    --num_machines "${SLURM_NNODES}" \
    --num_processes "${TOTAL_GPUS}" \
    examples/ARXHuman/train_files/train_starvla_rank_triton_cache.py \
    --config_yaml "${CONFIG_YAML}" \
    --framework.name QwenPI_v3 \
    --framework.qwenvl.base_vlm "${BASE_VLM}" \
    --framework.qwenvl.attn_implementation sdpa \
    --framework.qwenvl.ignore_mismatched_sizes false \
    --framework.qwenvl.enable_gradient_checkpointing true \
    --framework.encode_state_as_text true \
    --datasets.vla_data.data_root_dir / \
    --datasets.vla_data.data_mix ARX_openpi_tube_human_v2 \
    --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --datasets.vla_data.num_workers 1 \
    --datasets.vla_data.prefetch_factor 1 \
    --datasets.vla_data.persistent_workers false \
    --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
    --trainer.lora.enabled false \
    --trainer.learning_rate.base 1.0e-05 \
    --trainer.learning_rate.qwen_vl_interface 1.0e-05 \
    --trainer.max_train_steps 100000 \
    --trainer.save_interval 5000 \
    --trainer.max_checkpoints_to_keep 2 \
    --trainer.logging_frequency 10 \
    --trainer.eval_interval 200 \
    --run_root_dir "${RUN_ROOT_DIR}" \
    --run_id "${RUN_ID}" \
    --wandb_project starVLA_vlac
'
