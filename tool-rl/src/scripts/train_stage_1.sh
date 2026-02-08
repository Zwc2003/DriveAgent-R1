ACCELERATE_LOG_LEVEL=info 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE="offline"
# export CUDA_HOME=


accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path model_name_or_path \
    --dataset_name dataset_name \
    --learning_rate 2.0e-5 \
    --num_train_epochs 5 \
    --packing \
    --max_seq_length 5120 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 1 \
    --output_dir output_dir  \
    --run_name run_name \
    --attn_implementation flash_attention_2 \
    --report_to none \
