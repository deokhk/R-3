#!/bin/bash
python train_dense_encoder.py \
train="biencoder_for_development" \
train_datasets=[nq_train_with_table] \
dev_datasets=[nq_dev_with_table] \
model_file="/home/deokhk/research/MultiQA/model/DPR/dpr/data/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp" \
output_dir="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/best_dpr_with_new_hn_ckpt" \
experiment_name="test"
