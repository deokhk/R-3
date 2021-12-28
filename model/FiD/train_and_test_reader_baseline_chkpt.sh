#!/bin/bash
python  train_reader.py \
        --train_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/DPR_base_chkpt/train.json" \
        --eval_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/DPR_base_chkpt/dev.json" \
        --model_size base \
        --per_gpu_batch_size 8 \
        --n_context 100 \
        --checkpoint_dir checkpoint \
        --name 64_DPR_chkpt \
        --use_checkpoint \
        --lr 0.00005 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --text_maxlength 250 \
        --total_step 30000 \
        --warmup_step 2000 \
        --accumulation_steps 2 \
        --eval_freq 1000 \
        --gpus 4 \
        --do_wandb_log

python test_reader.py \
        --model_path checkpoint/64_DPR_chkpt/checkpoint/best_dev \
        --eval_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/DPR_base_chkpt/test.json" \
        --per_gpu_batch_size 8 \
        --n_context 100 \
        --name 64_DPR_chkpt \
        --checkpoint_dir checkpoint \
