import json 

train_path = "/home/deokhk/research/MultiQA_data/retriever/plain/nq-train-text-table-without-special.json"
dev_path = "/home/deokhk/research/MultiQA_data/retriever/plain/nq-dev-text-table-without-special.json"

with open(train_path, 'r') as f:
    train = json.load(f)
with open(dev_path, 'r') as f:
    dev = json.load(f)

train_text_only = []
for qapair in train:
    pos = qapair["positive_ctxs"]
    psg = pos[0]
    if int(psg["passage_id"]) < 21015325:
        train_text_only.append(qapair)
with open("/home/deokhk/research/MultiQA_data/retriever/plain/nq-train-text-without-special.json" , 'w') as f:
    json.dump(train_text_only, f)

dev_text_only = []
for qapair in dev:
    pos = qapair["positive_ctxs"]
    psg = pos[0]
    if int(psg["passage_id"]) < 21015325:
        dev_text_only.append(qapair)
with open("/home/deokhk/research/MultiQA_data/retriever/plain/nq-dev-text-without-special.json" , 'w') as f:
    json.dump(dev_text_only, f)
