#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
train="biencoder_for_development" \
train_datasets=[nq_dev_table_only_rel] \
dev_datasets=[nq_dev_table_only_rel] \
train="biencoder_for_development" \
output_dir="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/test" \
experiment_name="test"