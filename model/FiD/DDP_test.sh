#!/bin/bash

python train_reader.py \
        --train_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/train.json" \
        --eval_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/dev.json" \
        --model_size base \
        --model_path "/home/deokhk/research/MultiQA/model/FiD/pretrained_models/nq_reader_base" \
        --per_gpu_batch_size 1 \
        --n_context 1 \
        --fine_tune_pretrained_model \
        --name DDP_test \
        --checkpoint_dir checkpoint \
        --use_checkpoint \
        --lr 0.00005 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --text_maxlength 250 \
        --total_step 200 \
        --warmup_step 10 \
        --accumulation_steps 4 \
        --eval_freq 5 \
        --gpus 2
