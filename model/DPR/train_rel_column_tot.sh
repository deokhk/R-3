#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
train="biencoder_proposed_scratch_single" \
train_datasets=[nq_train_with_table_rel] \
dev_datasets=[nq_dev_with_table_rel] \
train="biencoder_proposed_scratch_single" \
output_dir="/home1/deokhk_1/research/MultiQA/model/DPR/trained_model_checkpoints/column_only/total" \
experiment_name="rel_dpr_column_only_total"