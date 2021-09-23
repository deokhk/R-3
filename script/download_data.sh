#!/bin/bash

python /home/deokhk/research/MultiQA/model/DPR/dpr/data/download_data.py --resource "data.retriever.nq-dev"
python /home/deokhk/research/MultiQA/model/DPR/dpr/data/download_data.py --resource "data.retriever.nq-train"
python /home/deokhk/research/MultiQA/model/DPR/dpr/data/download_data.py --resource "data.retriever.qas.nq-dev"
python /home/deokhk/research/MultiQA/model/DPR/dpr/data/download_data.py --resource "data.retriever.nq-test"
python /home/deokhk/research/MultiQA/model/DPR/dpr/data/download_data.py --resource "data.retriever.nq-train"
python /home/deokhk/research/MultiQA/model/DPR/dpr/data/download_data.py --resource "data.wikipedia_split.psgs_w100"