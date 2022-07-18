#!/bin/bash

# First, convert for total split
python convert_retrieval_to_csv.py \
--train_path "/home/deokhk/research/MultiQA_data/retriever/plain/nq-train-text-table_without_special_token.json" \
--dev_path "/home/deokhk/research/MultiQA_data/retriever/plain/nq-dev-text-table_without_special_token.json" \
--train_out_csv "/home/deokhk/research/MultiQA_data/retriever/plain/nq-train-text-table_without_special_token.csv" \
--dev_out_csv "/home/deokhk/research/MultiQA_data/retriever/plain/nq-dev-text-table_without_special_token.csv" 

# Second, convert for text split
python convert_retrieval_to_csv.py \
--train_path "/home/deokhk/research/MultiQA_data/retriever/plain/nq-train-text-without-special.json" \
--dev_path "/home/deokhk/research/MultiQA_data/retriever/plain/nq-dev-text-without-special.json" \
--train_out_csv "/home/deokhk/research/MultiQA_data/retriever/plain/nq-train-text-without-special.csv" \
--dev_out_csv "/home/deokhk/research/MultiQA_data/retriever/plain/nq-dev-text-without-special.csv" 

# First, convert for table split
python convert_retrieval_to_csv.py \
--train_path "/home/deokhk/research/MultiQA_data/retriever/plain/table_train_without_special_token.json" \
--dev_path "/home/deokhk/research/MultiQA_data/retriever/plain/table_dev_without_special_token.json" \
--train_out_csv "/home/deokhk/research/MultiQA_data/retriever/plain/table_train_without_special_token.csv" \
--dev_out_csv "/home/deokhk/research/MultiQA_data/retriever/plain/table_dev_without_special_token.csv" 
