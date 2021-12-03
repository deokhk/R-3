#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 \
train_dense_encoder.py \
train="biencoder_baseline_special_token_scratch" \
train_datasets=[nq_train_with_table] \
dev_datasets=[nq_dev_with_table] \
train="biencoder_baseline_special_token_scratch" \
output_dir="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/best_dpr_new_scratch" \
experiment_name="" \
'special_tokens=["[C_SEP]","[V_SEP]","[R_SEP]"]' 
