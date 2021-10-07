#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 \
train_dense_encoder.py \
train="custom_yaml/biencoder_nq_a100_4.yaml" \
train_datasets=[nq_train_with_table] \
dev_datasets=[nq_dev_with_table] \
train="custom_yaml/biencoder_nq_a100_4.yaml" \
model_file="/home/deokhk/research/MultiQA/model/DPR/dpr/data/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp" \
output_dir="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/"
