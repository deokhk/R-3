python dense_retriever.py \
	model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/best_dpr_with_table_ckpt/dpr_biencoder.34" \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki_with_table] \
	encoded_ctx_files=["/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/from_ckpt/_0"] \
	out_file="/home/deokhk/research/MultiQA/model/DPR/retrieval_eval_output/from_ckpt"
