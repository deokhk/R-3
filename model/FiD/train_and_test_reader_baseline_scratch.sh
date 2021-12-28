#!/bin/bash

python train_reader.py \
        --train_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/DPR_base_scratch/train.json" \
        --eval_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/DPR_base_scratch/dev.json" \
        --model_size base \
        --per_gpu_batch_size 8 \
        --n_context 100 \
        --name 64_DPR_scratch \
        --checkpoint_dir checkpoint \
        --use_checkpoint \
        --lr 0.00005 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --text_maxlength 250 \
        --total_step 15000 \
        --warmup_step 1000 \
        --accumulation_steps 1 \
        --gpus 8 \
        --do_wandb_log

python test_reader.py \
        --model_path checkpoint/64_DPR_scratch/checkpoint/best_dev \
        --eval_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/DPR_base_scratch/test.json" \
        --per_gpu_batch_size 8 \
        --n_context 100 \
        --name 64_DPR_scratch \
        --checkpoint_dir checkpoint \
