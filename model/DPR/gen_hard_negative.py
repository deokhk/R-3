from datetime import datetime
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
import csv
import json
import argparse
import os
from tqdm import tqdm
from dpr.data.qa_validation import has_answer
from transformers.tokenization_bert import BertTokenizer
from generate_table_passages_renew import _add_special_tokens

"""
Generate hard negative passages for retrieval training datasets, given a passages.
Make sure to run elasticsearch before.
"""

def gen_passages(passage_loc):
    passages = open(passage_loc, "r")
    read_passages = csv.reader(passages, delimiter="\t")
    
    for passage in tqdm(read_passages):
        title = passage[2]
        content = passage[1]
        psg = title + " [SEP] " + content
        yield {
            "_index" : "wikipedia",
            "passage": psg
        }

    
def main(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    es = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True)
    passage_loc = args.passage_loc
    bulk(es, gen_passages(passage_loc)) # Index table passages. Wait a while till all passages are indexed..

    with open(args.train_table_without_hn, "r") as f:
        train_table_without_hn = json.load(f)

    with open(args.train_text, "r") as f:
        train_text = json.load(f)
    
    # Generate text to psg_id mapping for put passage id and origin into the passage in the generated train file.
    text_to_psg_id = {}
    with open(passage_loc, 'r') as file:
        # Table passage file doesn't have a header.
        table_passages = csv.reader(file, delimiter="\t")
        for row in table_passages:
            psg_id = row[0]
            text = row[1].strip()
            text_to_psg_id[text] = psg_id

    train_text_table = train_table_without_hn + train_text
    updated_train = []
    cnt = 0
    for qa_pair in tqdm(train_text_table):
        question = qa_pair["question"]
        answers = qa_pair["answers"]
        query = {
            "query": {
                "match": {
                    "passage" : question
                }
            }
        }
        res = es.search(index="wikipedia", size=100, body = query)
        matched_passages = res["hits"]["hits"]
        # filter passage that does contain the answers.
        hn_passage = ""
        title = ""
        for passage in matched_passages:
            passage_content = passage["_source"]["passage"]
            answer_in_content = has_answer(answers, passage_content, tokenizer, "string")
            if answer_in_content:
                continue

            x = passage_content.split("[SEP]", maxsplit=1)
            title = x[0].strip()
            hn_passage = x[1].strip()
            break

        updated_qa_pair = qa_pair
        if hn_passage in text_to_psg_id:
            pid = text_to_psg_id[hn_passage.strip()]

            hn_ctxs = [{"title": title, "text": hn_passage, "passage_id": pid}]
            updated_qa_pair["hard_negative_ctxs"] = hn_ctxs
            updated_train.append(updated_qa_pair)
        else:
            cnt+=1
            print(f"Failed to find hard negative passages!. Total failed : {cnt}")


    print("Generate a hard negative passage to train data completed!")
    print(f"Total :{len(train_text_table)}, not generated :{cnt}")

    with open(args.output_path + "nq-train-text-table_without_special_token.json", "w") as f:
        json.dump(updated_train, f)

    print("Now generating a hard negative passages for dev data.")
    with open(args.dev_table_without_hn, "r") as f:
        dev_table_without_hn = json.load(f)
    with open(args.dev_text, "r") as f:
        dev_text = json.load(f)
    
    dev_text_table = dev_table_without_hn + dev_text

    updated_table_dev = []
    cnt = 0
    for qa_pair in tqdm(dev_text_table):
        question = qa_pair["question"]
        answers = qa_pair["answers"]
        query = {
            "query": {
                "match": {
                    "passage" : question
                }
            }
        }
        res = es.search(index="wikipedia", size=100, body = query)
        matched_passages = res["hits"]["hits"]
        # filter passage that does contain the answers.
        hn_passage = ""
        title = ""
        for passage in matched_passages:
            passage_content = passage["_source"]["passage"]
            answer_in_content = has_answer(answers, passage_content, tokenizer, "string")
            if answer_in_content:
                continue

            x = passage_content.split("[SEP]", maxsplit=1)
            title = x[0].strip()
            hn_passage = x[1].strip()
            break
        
        updated_qa_pair = qa_pair
        if hn_passage in text_to_psg_id:
            pid = text_to_psg_id[hn_passage.strip()]

            hn_ctxs = [{"title": title, "text": hn_passage, "passage_id": pid}]
            updated_qa_pair["hard_negative_ctxs"] = hn_ctxs
            updated_table_dev.append(updated_qa_pair)
        else:
            cnt+=1
            print(f"Failed to find hard negative passages!. Total failed : {cnt}")

    with open(args.output_path + "nq-dev-text-table_without_special_token.json", "w") as f:
        json.dump(updated_table_dev, f)

    print("Generate a hard negative passage to dev data completed!")
    print(f"Total :{len(dev_text_table)}, not generated :{cnt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adds hard negative to table training data.")
    parser.add_argument('--train_table_without_hn', help='path to input table training data file', default='/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/table_train_without_special_token.json')
    parser.add_argument('--dev_table_without_hn', help='path to input table dev data file', default='/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/table_dev_without_special_token.json')
    parser.add_argument('--train_text', help='path to nq-train.json', default='/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-train.json')
    parser.add_argument('--dev_text', help='path to nq-dev.json', default='/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-dev.json')
    parser.add_argument('--output_path',  default='/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/', help='path to output directory where augmented training/dev data will be added')
    parser.add_argument('--passage_loc',  default='/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/wikipedia_split/psg_table_w100_without_special_token.tsv', help='path to the passage file')
    args = parser.parse_args()
    main(args)