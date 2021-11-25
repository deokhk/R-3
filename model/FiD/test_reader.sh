#!/bin/bash

python test_reader.py \
        --model_path checkpoint/64_multi_available/checkpoint/best_dev \
        --eval_data "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/test.json" \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name 64_multi_available \
        --checkpoint_dir checkpoint \
