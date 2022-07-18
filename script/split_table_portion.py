import json 

train_path = "/home/deokhk/research/MultiQA_data/retriever/plain/nq-train-text-table_without_special_token.json"
dev_path = "/home/deokhk/research/MultiQA_data/retriever/plain/nq-dev-text-table_without_special_token.json"

with open(train_path, 'r') as f:
    train = json.load(f)
with open(dev_path, 'r') as f:
    dev = json.load(f)

train_table_only = []
for qapair in train:
    pos = qapair["positive_ctxs"]
    psg = pos[0]
    if int(psg["passage_id"]) >= 21015325:
        train_table_only.append(qapair)
with open("/home/deokhk/research/MultiQA_data/retriever/plain/table_train_without_special_token.json" , 'w') as f:
    json.dump(train_table_only, f)

dev_table_only = []
for qapair in dev:
    pos = qapair["positive_ctxs"]
    psg = pos[0]
    if int(psg["passage_id"]) >= 21015325:
        dev_table_only.append(qapair)
with open("/home/deokhk/research/MultiQA_data/retriever/plain/table_dev_without_special_token.json" , 'w') as f:
    json.dump(dev_table_only, f)
