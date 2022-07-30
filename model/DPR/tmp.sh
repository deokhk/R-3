#!/bin/bash
python dense_retriever.py \
	model_file="/home1/deokhk_1/research/MultiQA/model/DPR/trained_model_checkpoints/column_only/table_best/dpr_biencoder.39" \
	qa_dataset=nq_dev_table_csv \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home1/deokhk_1/research/MultiQA_data/dense_embeddings/column_only_tb/_0"] \
	out_file="/home1/deokhk_1/research/MultiQA_data/reader_ctx/col_table/c_reader_table_scratch.json"