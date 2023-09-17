#!/bin/bash

# accelerate launch ../src/train_bash.py \
# CUDA_VISIBLE_DEVICES=0 python -m pdb ./dpo.py \
CUDA_VISIBLE_DEVICES=0 python ./dpo.py \
    --model_name_or_path /root/LLaMA-Efficient-Tuning/examples/linewell_chatglm_0915/finetune_sft_lora/exported_model \
    --data_path /root/ChatGLM-RLHF/data/dpo_2000.json \
    --output_dir ./output_0915/finetune_dpo_lora \
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
    --learning_rate 1e-3 \
    --num_train_epochs 100.0 \
    --load_best_model_at_end \
    --plot_loss \
    2>&1 | tee chatglm2-dpo-lora.log
