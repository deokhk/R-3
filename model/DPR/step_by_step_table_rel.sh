#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
train="biencoder_proposed_step_by_step" \
train_datasets=[nq_train_only_table_upsampled_rel] \
dev_datasets=[nq_dev_only_table_upsampled_rel] \
output_dir="/home1/deokhk_1/research/MultiQA_data/trained_model_checkpoints/col_row_simul/table" \
model_file="/home1/deokhk_1/research/MultiQA_data/trained_model_checkpoints/column_only/table/dpr_biencoder.19" \
experiment_name="step_by_step_rel_table"