#!/bin/bash

# accelerate launch ../src/train_bash.py \
# CUDA_VISIBLE_DEVICES=0 python -m pdb ./dpo.py \
CUDA_VISIBLE_DEVICES=0 python ./sft.py \
    --stage sft \
    --model_name_or_path /root/share/chatglm2-6b \
    --data_path /root/ChatGLM-RLHF/data/sft_train.json \
    --output_dir ./output_0918/finetune_sft_freeze \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --val_size 0.01 \
    --learning_rate 1e-4 \
    --num_train_epochs 14.0 \
    --load_best_model_at_end \
    --plot_loss \
    --fp16 \
    2>&1 | tee chatglm2-sft-freeze.log
