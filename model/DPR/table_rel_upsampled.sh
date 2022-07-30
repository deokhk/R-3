#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
train="biencoder_proposed_scratch_single" \
train_datasets=[nq_train_only_table_upsampled_rel] \
dev_datasets=[nq_dev_only_table_upsampled_rel] \
train="biencoder_proposed_scratch_single" \
output_dir="/home1/deokhk_1/research/MultiQA_data/trained_model_checkpoints/col_row_simul/table" \
experiment_name="rel_dpr_table_upsampled_simul"