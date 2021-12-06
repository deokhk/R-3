#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 \
train_dense_encoder.py \
train="biencoder_baseline_without_special_token" \
train_datasets=[nq_train_with_table_new_linearization] \
dev_datasets=[nq_dev_with_table_new_linearization] \
train="biencoder_baseline_without_special_token" \
model_file="/home/deokhk/research/MultiQA/model/DPR/dpr/data/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp" \
output_dir="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/best_dpr_new_linearization_chkpt" \
experiment_name="new_linearization_chkpt"