#!/bin/bash

# Generate dense embedding
python generate_dense_embeddings.py \
        model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/exp_source/total_best/dpr_biencoder.36" \
        ctx_src=dpr_wiki_with_table_col_row \
        shard_id=0 num_shards=1 \
        out_file="/home/deokhk/research/MultiQA_data/dense_embeddings/rel_total/"	

# Evaluate the model on total split
python dense_retriever.py \
	model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/exp_source/total_best/dpr_biencoder.36" \
	qa_dataset=nq_dev_with_table_csv \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA_data/dense_embeddings/rel_total/_0"] \
	out_file="/home/deokhk/research/MultiQA_data/reader_ctx/rel_total/c_reader_total_scratch.json"

# Evaluate the model on text split
python dense_retriever.py \
	model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/exp_source/total_best/dpr_biencoder.36" \
	qa_dataset=nq_dev_text_csv \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA_data/dense_embeddings/rel_total/_0"] \
	out_file="/home/deokhk/research/MultiQA_data/reader_ctx/rel_total/c_reader_text_scratch.json"

# Evaluate the model on table split
python dense_retriever.py \
	model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/exp_source/total_best/dpr_biencoder.36" \
	qa_dataset=nq_dev_table_csv \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA_data/dense_embeddings/rel_total/_0"] \
	out_file="/home/deokhk/research/MultiQA_data/reader_ctx/rel_total/c_reader_table_scratch.json"