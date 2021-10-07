#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 \
train_dense_encoder.py \
train="biencoder_baseline_in_a100_2" \
train_datasets=[nq_train_with_table] \
dev_datasets=[nq_dev_with_table] \
train="biencoder_baseline_in_a100_2" \
model_file="/home/deokhk/research/MultiQA/model/DPR/dpr/data/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp" \
output_dir="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/"
