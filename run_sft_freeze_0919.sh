#!/bin/bash

DATE=0919
STAGE=sft
METHOD=freeze

# accelerate launch ../src/train_bash.py \
CUDA_VISIBLE_DEVICES=1 python ./sft.py \
    --stage $STAGE \
    --model_name_or_path /root/share/chatglm2-6b \
    --data_path /root/ChatGLM-RLHF/data/sft_train.json \
    --output_dir ./output_$DATE/finetune-$STAGE-$METHOD \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
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
    2>&1 | tee chatglm2-$STAGE-$METHOD-$DATE.log
