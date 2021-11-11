#!/bin/bash

python test_reader.py \
        --model_path checkpoint/experiment_nq_text_table/checkpoint/best_dev \
        --eval_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/test.json" \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name test_plain_reader \
        --checkpoint_dir checkpoint \
