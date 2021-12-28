#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
train="biencoder_proposed_chkpt_single" \
train_datasets=[nq_train_with_table_rel] \
dev_datasets=[nq_dev_with_table_rel] \
train="biencoder_proposed_chkpt_single" \
model_file="/home/deokhk/research/MultiQA/model/DPR/dpr/data/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp" \
output_dir="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/rel_dpr_chkpt_same" \
experiment_name="rel_dpr_chkpt_same"