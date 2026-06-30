#!/bin/bash
set -e 

export VLLM_USE_FLASHINFER=0

# --- Directory Configuration ---
PROJECT_ROOT="/path/to/your/project"
CONDA_BASE="/path/to/conda"
MODEL_DIR="/path/to/models/qwen3_14b_sft_checkpoint"
DATA_DIR="/path/to/data/grpo_data"
SAVE_DIR="/path/to/output/runs/qwen3_14b_v1.0_gen_grpo"

# --- Environment Configuration ---
export HF_HOME="/path/to/cache/huggingface"
export MODELSCOPE_CACHE="/path/to/cache/modelscope"
export MEGATRON_LM_PATH="/path/to/build/apex/Megatron-LM"
export PYTHONNOUSERSITE=True
export WANDB_PROJECT=htmlgen-1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600

echo -e "\n🚀 [Phase 1/2] Starting environment configuration and dependency installation..."

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "❌ Error: Cannot find conda.sh, please check your CONDA_BASE path."
    exit 1
fi

conda activate train-svg

# Install rendering dependencies
pip install playwright
if [ ! -d "$HOME/.cache/ms-playwright" ]; then
    playwright install chromium
fi

echo -e "✅ Environment setup complete!"

# =========================================================
# Phase 2: GRPO (Group Relative Policy Optimization)
# =========================================================
echo -e "\n🚀 [Phase 2/2] Preparing to start GRPO training task..."

OUTPUT_NAME=qwen3_14b_v3.1_gen_grpo
WANDB_NAME=${OUTPUT_NAME}/v0
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NODE_ID=${ARNOLD_ID:-0}
LOG_FILE="${SAVE_DIR}/train_ALL_NODE${NODE_ID}_${TIMESTAMP}.log"

mkdir -p "${SAVE_DIR}"
# Redirect output to both log file and terminal
exec > >(tee -a ${LOG_FILE}) 2>&1

set -x 

# Launch Distributed Training
NNODES=${ARNOLD_WORKER_NUM} \
NODE_RANK=${ARNOLD_ID} \
MASTER_ADDR=${ARNOLD_WORKER_0_HOST} \
MASTER_PORT=${ARNOLD_WORKER_0_PORT} \
NPROC_PER_NODE=${ARNOLD_WORKER_GPU} \
${CONDA_BASE}/envs/train-magasvg/bin/megatron rlhf \
    --rlhf_type grpo \
    --model ${MODEL_DIR} \
    --temperature 0.7 \
    --top_p 0.95 \
    --load_safetensors true \
    --save_safetensors true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 8 \
    --pipeline_model_parallel_size 4 \
    --dataset ${DATA_DIR}/hard.jsonl#8000 \
    --max_epochs 1 \
    --global_batch_size 128 \
    --micro_batch_size 1 \
    --num_generations 8 \
    --external_plugins plugin.py \
    --reward_funcs svg_scorer svg_complexity_scorer \
    --save ${SAVE_DIR} \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_max_model_len 10384 \
    --vllm_tensor_parallel_size 8 \
    --max_length 4096 \
    --max_completion_length 7000 \
    --train_type full \
    --lr 1e-6 \
    --bf16 true \
    --beta 0.001 \
    --importance_sampling_level token \
    --epsilon 0.2 \
    --epsilon_high 0.2 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 2 \
    --offload_model true \
    --offload_optimizer true \
    --log_interval 1 \
    --recompute_granularity selective \
    --finetune \
    --attention_backend flash \
    --padding_free true \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_exp_name ${WANDB_NAME} \
    --eval_interval 50 \
    --save_interval 20