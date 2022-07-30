#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python /home1/deokhk_1/research/MultiQA/model/DPR/train_dense_encoder.py \
train="biencoder_proposed_scratch_single" \
train_datasets=[nq_train_total_rel] \
dev_datasets=[nq_dev_total_rel] \
output_dir="/home1/deokhk_1/research/MultiQA_data/trained_model_checkpoints/col_row_simul/total" \
experiment_name="rel_dpr_total_upsampled_simul"