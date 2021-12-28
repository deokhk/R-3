#!/bin/bash

# Generate dense embedding
python generate_dense_embeddings.py \
        model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/rel_dpr_chkpt_same/dpr_biencoder.50" \
        ctx_src=dpr_wiki_with_table_col_row \
        shard_id=0 num_shards=1 \
        out_file="/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/rel_dpr_chkpt_same/"	
# Evaluate the model & Generate reader dataset

python dense_retriever.py \
	model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/rel_dpr_chkpt_same/dpr_biencoder.50" \
	qa_dataset=nq_train_with_table_csv \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/rel_dpr_chkpt_same/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/proposed_chkpt_reader_context/c_reader_train_scratch.json"

python dense_retriever.py \
	model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/rel_dpr_chkpt_same/dpr_biencoder.50" \
	qa_dataset=nq_dev_with_table_csv \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/rel_dpr_chkpt_same/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/proposed_chkpt_reader_context/c_reader_dev_scratch.json"

python dense_retriever.py \
	model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/rel_dpr_chkpt_same/dpr_biencoder.50" \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki_with_table_col_row] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/rel_dpr_chkpt_same/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/proposed_chkpt_reader_context/c_reader_test_scratch.json"

python gen_reader_dataset.py \
        --train_context_path "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/proposed_chkpt_reader_context/c_reader_train_scratch.json" \
        --dev_context_path "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/proposed_chkpt_reader_context/c_reader_dev_scratch.json" \
        --test_context_path "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/proposed_chkpt_reader_context/c_reader_test_scratch.json" \
        --saved_path "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/rel_dpr_chkpt_same/"