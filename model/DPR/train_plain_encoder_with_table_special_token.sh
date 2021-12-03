#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 \
train_dense_encoder.py \
train="biencoder_baseline_special_token" \
train_datasets=[nq_train_with_table] \
dev_datasets=[nq_dev_with_table] \
train="biencoder_baseline_special_token" \
model_file="/home/deokhk/research/MultiQA/model/DPR/dpr/data/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp" \
output_dir="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/best_dpr_new_chkpt" \
'special_tokens=["[C_SEP]","[V_SEP]","[R_SEP]"]' 
