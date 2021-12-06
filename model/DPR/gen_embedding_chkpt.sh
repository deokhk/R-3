#!/bin/bash
python generate_dense_embeddings.py \
        model_file="/home/deokhk/research/MultiQA/model/DPR/trained_model_checkpoints/best_dpr_new_linearization_chkpt/dpr_biencoder.44" \
        ctx_src=dpr_wiki_with_table \
        shard_id=0 num_shards=1 \
        out_file="/home/deokhk/research/MultiQA/model/DPR/dense_embeddings/from_new_linearization_chkpt/"	
