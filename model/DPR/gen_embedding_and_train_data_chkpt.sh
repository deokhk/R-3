#!/bin/bash

#Convert retrieval file into..
python convert_retrieval_to_csv.py \
    --train_path "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-train-text-table-fin.json" \
    --dev_path "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-dev-text-table-fin.json" \
    --train_out_csv "/home/deokhk/research/MultiQA/model/DPR/downloads/data/retriever/qas/nq-train-text-table-without-special.csv" \
    --dev_out_csv "/home/deokhk/research/MultiQA/model/DPR/downloads/data/retriever/qas/nq-dev-text-table-without-special.csv"

# Generate dense embedding
python generate_dense_embeddings.py \
        model_file="/home/deokhk/research/DPR/output_dir/DPR_base_chkpt/dpr_biencoder.49" \
        ctx_src=dpr_wiki_with_table \
        shard_id=0 num_shards=1 \
        out_file="/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/DPR_base_chkpt/"	
# Evaluate the model & Generate reader dataset

python dense_retriever.py \
	model_file="/home/deokhk/research/DPR/output_dir/DPR_base_chkpt/dpr_biencoder.49" \
	qa_dataset=nq_train_with_table_csv \
	ctx_datatsets=[dpr_wiki_with_table] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/DPR_base_chkpt/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_chkpt_reader_context/c_reader_train_scratch.json"

python dense_retriever.py \
	model_file="/home/deokhk/research/DPR/output_dir/DPR_base_chkpt/dpr_biencoder.49" \
	qa_dataset=nq_dev_with_table_csv \
	ctx_datatsets=[dpr_wiki_with_table] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/DPR_base_chkpt/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_chkpt_reader_context/c_reader_dev_scratch.json"

python dense_retriever.py \
	model_file="/home/deokhk/research/DPR/output_dir/DPR_base_chkpt/dpr_biencoder.49" \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki_with_table] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/DPR_base_chkpt/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_chkpt_reader_context/c_reader_test_scratch.json"

python gen_reader_dataset.py \
        --train_context_path "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_chkpt_reader_context/c_reader_train_scratch.json" \
        --dev_context_path "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_chkpt_reader_context/c_reader_dev_scratch.json" \
        --test_context_path "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_chkpt_reader_context/c_reader_test_scratch.json" \
        --saved_path "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/DPR_base_chkpt/"