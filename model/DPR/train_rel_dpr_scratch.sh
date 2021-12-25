#!/bin/bash
python train_dense_encoder.py \
train="biencoder_proposed_scratch" \
train_datasets=[nq_train_with_table_rel] \
dev_datasets=[nq_dev_with_table_rel] \
train="biencoder_proposed_scratch" \
output_dir="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/rel_dpr_scratch" \
experiment_name="rel_dpr_scratch" \
column_file_loc="/home/deokhk/research/MultiQA/model/DPR/column_ids_list_without_special_token.pickle" \
row_file_loc="/home/deokhk/research/MultiQA/model/DPR/row_ids_list_without_special_token.pickle"
