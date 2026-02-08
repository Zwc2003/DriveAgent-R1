#!/bin/bash


# The latest vllm==0.7.3 is required for this script: pip3 install vllm==0.7.3
# The latest transformers is required too, install by: pip install git+https://github.com/huggingface/transformers.git@a40f1ac602fe900281722254c52ce3773f28eb0e

EXP_NAME="Qwen2.5-VL-3B-mix-RL-from-SFT"
export WANDB_MODE="offline"
export DEBUG_MODE="true"
export LOG_DIR="path-to/$EXP_NAME/vllm_run_log"
export LOG_PATH="path-to/$EXP_NAME/vllm_run_log/$EXP_NAME.txt"
if [ ! -d "$LOG_DIR" ]; then
 mkdir -p "$LOG_DIR"
fi


QWEN_PATH="path-to-model"
DATASET="path-to-dataset"
OUTPUT_DIR="path-to-output/$EXP_NAME"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="$EXP_NAME"
DS_CONFIG="src/r1-v/local_scripts/zero1_no_optimizer.json"  # Note that other zero setting would meet bugs related to vllm at current stage.
# export VLLM_MAX_MODEL_LEN=6144
# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"


# 5k
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun \
    --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12326" \
    src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${DATASET} \
    --max_prompt_length 4608 \
    --max_completion_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 6272 \
    --max_pixels 259200 \
    --num_train_epochs 1 \
    --run_name ${RUN_NAME} \
    --save_steps 10 \
    --save_total_limit 3 \
    --save_only_model true \
    --report_to none \
    --temperature 1.0 \
    --num_generations 4 \
    --vllm_device "cuda:7" \
    --vllm_gpu_memory_utilization 0.85 \
    --use_vllm true \
    --deepspeed ${DS_CONFIG} \
    --use_3_Stage_CoT true \
    --enable_tool_calls true\
    --mode "mixed"\
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"