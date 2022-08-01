#!/bin/bash

# Generate dense embedding
python generate_dense_embeddings.py \
        model_file="/home1/deokhk_1/research/MultiQA_data/trained_model_checkpoints/column_only/table_best/dpr_biencoder.31" \
        ctx_src=dpr_wiki_with_table_col_row \
        shard_id=0 num_shards=1 \
        out_file="/home1/deokhk_1/research/MultiQA_data/dense_embeddings/rel_table/"	


# Evaluate the model on table split
python dense_retriever.py \
	model_file="/home1/deokhk_1/research/MultiQA_data/trained_model_checkpoints/column_only/table_best/dpr_biencoder.31" \
	qa_dataset=nq_test_table \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home1/deokhk_1/research/MultiQA_data/dense_embeddings/rel_table/_0"] \
	out_file="/home1/deokhk_1/research/MultiQA_data/reader_ctx/rel_table/c_reader_table_scratch.json"

# Evaluate the model on total split
python dense_retriever.py \
	model_file="/home1/deokhk_1/research/MultiQA_data/trained_model_checkpoints/column_only/table_best/dpr_biencoder.31" \
	qa_dataset=nq_test_total \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home1/deokhk_1/research/MultiQA_data/dense_embeddings/rel_table/_0"] \
	out_file="/home1/deokhk_1/research/MultiQA_data/reader_ctx/rel_table/c_reader_total_scratch.json"

# Evaluate the model on text split
python dense_retriever.py \
	model_file="/home1/deokhk_1/research/MultiQA_data/trained_model_checkpoints/column_only/table_best/dpr_biencoder.31" \
	qa_dataset=nq_test_text \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home1/deokhk_1/research/MultiQA_data/dense_embeddings/rel_table/_0"] \
	out_file="/home1/deokhk_1/research/MultiQA_data/reader_ctx/rel_table/c_reader_text_scratch.json"