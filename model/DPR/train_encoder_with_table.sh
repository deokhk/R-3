#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=8 \
train_dense_encoder.py \
train="biencoder_from_scratch_2080ti_8" \
train_datasets=[nq_train_with_table] \
dev_datasets=[nq_dev_with_table] \
train="biencoder_from_scratch_2080ti_8" \
output_dir="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/dpr_with_table_scratch/"
