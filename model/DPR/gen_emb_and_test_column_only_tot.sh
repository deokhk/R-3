#!/bin/bash

# Generate dense embedding
python generate_dense_embeddings.py \
        model_file="/home1/deokhk_1/research/MultiQA/model/DPR/trained_model_checkpoints/column_only/total_best/dpr_biencoder.37" \
        ctx_src=dpr_wiki_with_table_col_row \
        shard_id=0 num_shards=1 \
        out_file="/home1/deokhk_1/research/MultiQA_data/dense_embeddings/column_only_tot/"	

# Evaluate the model on total split
python dense_retriever.py \
	model_file="/home1/deokhk_1/research/MultiQA/model/DPR/trained_model_checkpoints/column_only/total_best/dpr_biencoder.37" \
	qa_dataset=nq_dev_with_table_csv \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home1/deokhk_1/research/MultiQA_data/dense_embeddings/column_only_tot/_0"] \
	out_file="/home1/deokhk_1/research/MultiQA_data/reader_ctx/col_total/c_reader_total_scratch.json"

# Evaluate the model on table split
python dense_retriever.py \
	model_file="/home1/deokhk_1/research/MultiQA/model/DPR/trained_model_checkpoints/column_only/total_best/dpr_biencoder.37" \
	qa_dataset=nq_dev_table_csv \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home1/deokhk_1/research/MultiQA_data/dense_embeddings/column_only_tot/_0"] \
	out_file="/home1/deokhk_1/research/MultiQA_data/reader_ctx/col_total/c_reader_table_scratch.json"