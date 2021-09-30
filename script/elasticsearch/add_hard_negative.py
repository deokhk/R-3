from datetime import datetime
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
import csv
import json
from tqdm import tqdm

def gen_tb_passages(table_passage_loc):
    passages = open(table_passage_loc, "r")
    read_passages = csv.reader(passages, delimiter="\t")
    
    for passage in tqdm(read_passages):
        title = passage[2]
        content = passage[1]
        psg = title + " [SEP] " + content
        yield {
            "_index" : "table_passages",
            "passage": psg
        }

    
es = Elasticsearch()
table_passage_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/wikipedia_split/table_w100.tsv"
bulk(es, gen_tb_passages(table_passage_loc)) # Index table passages. Wait a while till 1255730 passages are indexed..

with open("/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/table_train.json", "r") as f:
    table_train = json.load(f)

updated_table_train = []
for qa_pair in tqdm(table_train):
    question = qa_pair["question"]
    answers = qa_pair["answers"]
    query = {
        "query": {
            "match": {
                "passage" : question
            }
        }
    }
    res = es.search(index="table_passages", size=50, body = query)
    matched_passages = res["hits"]["hits"]
    # filter passage that does contain the answers.
    hn_passage = ""
    title = ""
    for passage in matched_passages:
        found = True
        passage_content = passage["_source"]["passage"]
        for a in answers:
            if a in passage_content:
                found = False
                break
        if found == True:
            x = passage_content.split("[SEP]", maxsplit=1)
            title = x[0].strip()
            hn_passage = x[1].strip()
            break

    updated_qa_pair = qa_pair
    hn_ctxs = [{"title": title, "text": hn_passage}]
    updated_qa_pair["hard_negative_ctxs"] = hn_ctxs
    updated_table_train.append(updated_qa_pair)

with open("table_train_updated.json", "w") as f:
    json.dump(updated_table_train, f)

print("Adding a hard negative passage to table train qa pair completed!")

with open("/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/table_dev.json", "r") as f:
    table_dev = json.load(f)

updated_table_dev = []
for qa_pair in tqdm(table_dev):
    question = qa_pair["question"]
    answers = qa_pair["answers"]
    query = {
        "query": {
            "match": {
                "passage" : question
            }
        }
    }
    res = es.search(index="table_passages", size=50, body = query)
    matched_passages = res["hits"]["hits"]
    # filter passage that does contain the answers.
    hn_passage = ""
    title = ""
    for passage in matched_passages:
        found = True
        passage_content = passage["_source"]["passage"]
        for a in answers:
            if a in passage_content:
                found = False
                break
        if found == True:
            x = passage_content.split("[SEP]", maxsplit=1)
            title = x[0].strip()
            hn_passage = x[1].strip()
            break

    updated_qa_pair = qa_pair
    hn_ctxs = [{"title": title, "text": hn_passage}]
    updated_qa_pair["hard_negative_ctxs"] = hn_ctxs
    updated_table_dev.append(updated_qa_pair)

with open("table_train_updated.json", "w") as f:
    json.dump(updated_table_dev, f)

print("Adding a hard negative passage to table dev qa pair completed!")

