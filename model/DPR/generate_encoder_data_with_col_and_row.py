import argparse
import json
import pickle
import logging
import os
from dpr.options import setup_logger



def main(args, logger):
    with open(args.train_path, 'r') as f:
        train = json.load(f)
    with open(args.dev_path, 'r') as f:
        dev = json.load(f)
    logger.info("Loaded train/dev json files. Length of train file")

    with open(args.column_file_loc, 'rb') as f:
        column = pickle.load(f)
    
    with open(args.row_file_loc, 'rb') as f:
        row = pickle.load(f)
    logger.info(f"Loaded colum/row files. Length of column file: {len(column)}, Length of row file: {len(row)}")
    
    for qapair in train:
        pos = qapair["positive_ctxs"]
        neg = qapair["hard_negative_ctxs"]
        try:
            for psg in pos:
                if int(psg["passage_id"]) >= 21015325:
                        psg["column_id"] = column[int(psg["passage_id"]) - 21015325]
                        psg["row_id"] = row[int(psg["passage_id"]) - 21015325]
                else:
                        psg["column_id"] = None
                        psg["row_id"] = None
            for psg in neg:
                if int(psg["passage_id"]) >= 21015325:
                    psg["column_id"] = column[int(psg["passage_id"]) - 21015325]
                    psg["row_id"] = row[int(psg["passage_id"]) - 21015325]
                else:
                    psg["column_id"] = None
                    psg["row_id"] = None
        except:
            import pdb
            pdb.set_trace()
    with open(os.path.join(args.out_dir, args.rel_train_file_name) , 'w') as f:
        json.dump(train, f)

    logger.info("Generated augmented train dataset.")

    for qapair in dev:
        pos = qapair["positive_ctxs"]
        neg = qapair["hard_negative_ctxs"]
        try:
            for psg in pos:
                if int(psg["passage_id"]) >= 21015325:
                    psg["column_id"] = column[int(psg["passage_id"]) - 21015325]
                    psg["row_id"] = row[int(psg["passage_id"]) - 21015325]
                else:
                    psg["column_id"] = None
                    psg["row_id"] = None
            for psg in neg:
                if int(psg["passage_id"]) >= 21015325:
                    psg["column_id"] = column[int(psg["passage_id"]) - 21015325]
                    psg["row_id"] = row[int(psg["passage_id"]) - 21015325]
                else:
                    psg["column_id"] = None
                    psg["row_id"] = None
        except:
            import pdb 
            pdb.set_trace()
    with open(os.path.join(args.out_dir,args.rel_dev_file_name) , 'w') as f:
        json.dump(dev, f)

    logger.info("Generated augmented dev dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help="path to the retrieval json train file", 
        default="/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-train-text-table-fin.json")
    parser.add_argument("--dev_path", help="path to the retrieval dev json file",
        default="/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-dev-text-table-fin.json")
    parser.add_argument("--column_file_loc", help="path to column file",
        default="/home/deokhk/research/MultiQA/model/DPR/column_ids_list_without_special_token.pickle")
    parser.add_argument("--row_file_loc", help="path to row file",
        default="/home/deokhk/research/MultiQA/model/DPR/row_ids_list_without_special_token.pickle")
    parser.add_argument("--out_dir", help="output path where the augmented encoder train/dev will be saved",
        default="/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/")
    parser.add_argument("--rel_train_file_name", help="name of the output train file", default="nq-rel-augmented-train.json")
    parser.add_argument("--rel_dev_file_name", help="name of the output dev file", default="nq-rel-augmented-dev.json")

    args = parser.parse_args()
    logger = logging.getLogger()
    setup_logger(logger)

    main(args, logger)
