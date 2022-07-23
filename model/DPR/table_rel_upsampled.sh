#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
train="biencoder_proposed_scratch_single" \
train_datasets=[nq_train_table_only_upsampled_rel] \
dev_datasets=[nq_dev_table_only_rel] \
train="biencoder_proposed_scratch_single" \
output_dir="/home1/deokhk_1/research/MultiQA/model/DPR/trained_model_checkpoints/upsample/table" \
experiment_name="rel_dpr_table_upsampled"