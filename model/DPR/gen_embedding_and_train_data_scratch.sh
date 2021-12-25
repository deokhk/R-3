#!/bin/bash

# Generate dense embedding
python generate_dense_embeddings.py \
        model_file="/home/deokhk/research/DPR/output_dir/DPR_base_scratch/dpr_biencoder.37" \
        ctx_src=dpr_wiki_with_table \
        shard_id=0 num_shards=1 \
        out_file="/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/DPR_base_scratch/"	
# Evaluate the model & Generate reader dataset

python dense_retriever.py \
	model_file="/home/deokhk/research/DPR/output_dir/DPR_base_scratch/dpr_biencoder.37" \
	qa_dataset=nq_train_with_table_csv \
	ctx_datatsets=[dpr_wiki_with_table] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/DPR_base_scratch/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_scratch_reader_context/c_reader_train_scratch.json"

python dense_retriever.py \
	model_file="/home/deokhk/research/DPR/output_dir/DPR_base_scratch/dpr_biencoder.37" \
	qa_dataset=nq_dev_with_table_csv \
	ctx_datatsets=[dpr_wiki_with_table] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/DPR_base_scratch/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_scratch_reader_context/c_reader_dev_scratch.json"

python dense_retriever.py \
	model_file="/home/deokhk/research/DPR/output_dir/DPR_base_scratch/dpr_biencoder.37" \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki_with_table] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/DPR_base_scratch/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_scratch_reader_context/c_reader_test_scratch.json"

python gen_reader_dataset.py \
        --train_context_path "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_scratch_reader_context/c_reader_train_scratch.json" \
        --dev_context_path "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_scratch_reader_context/c_reader_dev_scratch.json" \
        --test_context_path "/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/baseline_scratch_reader_context/c_reader_test_scratch.json" \
        --saved_path "/home/deokhk/research/MultiQA/model/DPR/reader_dataset/DPR_base_scratch/"