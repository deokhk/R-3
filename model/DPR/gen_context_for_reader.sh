#!/bin/bash

python convert_retrieval_to_csv.py \
	--train_path "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-train-text-table-without-special.json" \
	--dev_path "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-dev-text-table-without-special.json" \
	--train_out_csv "/home/deokhk/research/MultiQA/model/DPR/downloads/data/retriever/qas/nq-train-text-table-without-special.csv" \
	--dev_out_csv "/home/deokhk/research/MultiQA/model/DPR/downloads/data/retriever/qas/nq-dev-text-table-without-special.csv"

python dense_retriever.py \
	model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/best_dpr_new_linearization_chkpt/dpr_biencoder.44" \
	qa_dataset=nq_train_with_table_csv \
	ctx_datatsets=[dpr_wiki_with_table] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/from_new_linearization_chkpt/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/context_for_reader/c_reader_train.json"

python dense_retriever.py \
	model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/best_dpr_new_linearization_chkpt/dpr_biencoder.44" \
	qa_dataset=nq_dev_with_table_csv \
	ctx_datatsets=[dpr_wiki_with_table] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/from_new_linearization_chkpt/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/context_for_reader/c_reader_dev.json"

python dense_retriever.py \
	model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/best_dpr_new_linearization_chkpt/dpr_biencoder.44" \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki_with_table] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/from_new_linearization_chkpt/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/context_for_reader/c_reader_test.json"

python gen_reader_dataset.py