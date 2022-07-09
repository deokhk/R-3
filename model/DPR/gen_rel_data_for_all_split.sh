#!/bin/bash

# First, generate nq-rel-augmented-train.json / nq-rel-augmented-dev.json
python generate_encoder_data_with_col_and_row.py \
--train_path="/home/deokhk/research/MultiQA_data/retriever/plain/nq-train-text-table-without-special.json" \
--dev_path="/home/deokhk/research/MultiQA_data/retriever/plain/nq-dev-text-table-without-special.json" \
--out_dir="/home/deokhk/research/MultiQA_data/retriever/relative_embedding" \
--rel_train_file_name="nq-rel-augmented-train.json" \
--rel_dev_file_name="nq-rel-augmented-dev.json"

# Second, generate nq-rel-augmented-train-text.json / nq-rel-augmented-dev-text.json
python generate_encoder_data_with_col_and_row.py \
--train_path="/home/deokhk/research/MultiQA_data/retriever/plain/nq-train-text-without-special.json" \
--dev_path="/home/deokhk/research/MultiQA_data/retriever/plain/nq-dev-text-without-special.json" \
--out_dir="/home/deokhk/research/MultiQA_data/retriever/relative_embedding" \
--rel_train_file_name="nq-rel-augmented-train-text.json" \
--rel_dev_file_name="nq-rel-augmented-dev-text.json"

# Finally, generate nq-rel-augmented-train-table.json / nq-rel-augmented-dev-table.json
python generate_encoder_data_with_col_and_row.py \
--train_path="/home/deokhk/research/MultiQA_data/retriever/plain/table_train_without_special_token.json" \
--dev_path="/home/deokhk/research/MultiQA_data/retriever/plain/table_dev_without_special_token.json" \
--out_dir="/home/deokhk/research/MultiQA_data/retriever/relative_embedding" \
--rel_train_file_name="nq-rel-augmented-train-table.json" \
--rel_dev_file_name="nq-rel-augmented-dev-table.json"
