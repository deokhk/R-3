#!/bin/bash

python train_reader.py \
        --train_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/train.json" \
        --eval_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/dev.json" \
        --model_size large \
        --model_path "/home/deokhk/research/MultiQA/model/FiD/pretrained_models/nq_reader_large" \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --fine_tune_pretrained_model \
        --name experiment_nq_text_table \
        --checkpoint_dir checkpoint \
        --use_checkpoint \
        --lr 0.00005 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --text_maxlength 250 \
        --total_step 2000 \
        --warmup_step 200 \
        --accumulation_steps 16 \
        --eval_freq 200 
