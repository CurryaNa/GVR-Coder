#!/bin/bash

# ==============================================================================
# (Curriculum-driven Rejection Sampling Fine-tuning)
# ==============================================================================

export WANDB_PROJECT="svg-gen-curriculum"
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_BIN="megatron_sft_tool"
MODEL_PATH="./models/base_model"
BASE_SAVE_DIR="./runs/svg_gen_v1"


DATA_SIMPLE="simple_data.jsonl"
DATA_MEDIUM="medium_data.jsonl"
DATA_HARD="hard_data.jsonl"

NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"6001"}
NPROC_PER_NODE=${GPUS_PER_NODE:-8}

COMMON_ARGS="--load_safetensors true \
--save_safetensors true \
--tensor_model_parallel_size 8 \
--pipeline_model_parallel_size 1 \
--micro_batch_size 2 \
--global_batch_size 32 \
--max_length 32768 \
--bf16 true \
--use_flash_attn true \
--lr 1e-5 \
--min_lr 1e-6 \
--lr_warmup_iters 10 \
--max_epochs 1 \
--dataset_num_proc 16 \
--no_save_optim true"

# ------------------------------------------------------------------------------
# Stage 1: Basic (100% Simple Data)
# ------------------------------------------------------------------------------
echo ">>> Starting Stage 1: Basic Syntax & Concepts..."

STAGE1_SAVE="${BASE_SAVE_DIR}/stage1_basic"

NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
${TRAIN_BIN} sft \
--model ${MODEL_PATH} \
--dataset ${DATA_SIMPLE} \
--save ${STAGE1_SAVE} \
--wandb_exp_name "stage1_basic" \
${COMMON_ARGS}

# ------------------------------------------------------------------------------
# Stage 2: Generalization (85% Medium + 15% Simple)
# ------------------------------------------------------------------------------
echo ">>> Starting Stage 2: Generalization & Diversity..."

STAGE2_SAVE="${BASE_SAVE_DIR}/stage2_gen"

NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
${TRAIN_BIN} sft \
--model ${STAGE1_SAVE} \
--dataset ${DATA_MEDIUM} ${DATA_SIMPLE} \
--dataset_weights 0.85 0.15 \
--save ${STAGE2_SAVE} \
--wandb_exp_name "stage2_generalization" \
${COMMON_ARGS}

# ------------------------------------------------------------------------------
# Stage 3: Advanced (85% Hard + 15% Medium)
# ------------------------------------------------------------------------------
echo ">>> Starting Stage 3: Complex Reasoning..."

STAGE3_SAVE="${BASE_SAVE_DIR}/stage3_advanced"

NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
${TRAIN_BIN} sft \
--model ${STAGE2_SAVE} \
--dataset ${DATA_HARD} ${DATA_MEDIUM} \
--dataset_weights 0.85 0.15 \
--save ${STAGE3_SAVE} \
--wandb_exp_name "stage3_advanced" \
${COMMON_ARGS}

echo ">>> All stages completed successfully."