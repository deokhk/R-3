#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python /home1/deokhk_1/research/MultiQA/model/DPR/train_dense_encoder.py \
train="biencoder_proposed_scratch_single" \
train_datasets=[nq_train_with_table_upsampled_rel] \
dev_datasets=[nq_dev_with_table_rel] \
output_dir="/home1/deokhk_1/research/MultiQA/model/DPR/trained_model_checkpoints/upsample/total" \
experiment_name="upsample_dpr_total"