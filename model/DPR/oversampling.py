import json 
import argparse 
import os 
import logging
from dpr.options import setup_logger 



def main(args, logger):
    with open(args.total_train_path, 'r') as f:
        text_and_table = json.load(f)
    with open(args.table_train_path, 'r') as f:
        table = json.load(f)
    
    upsample_rate = int(len(text_and_table) / len(table))
    logger.info("Loaded text+table and table json files.")
    logger.info(f"Length of text+table train file: {len(text_and_table)}")
    logger.info(f"Length of table train file: {len(table)}")
    logger.info(f"Upsample rate: {upsample_rate}")


    upsampled_portion = table*upsample_rate
    text_and_table_upsampled = text_and_table + upsampled_portion
    table_upsampled = table + upsampled_portion

    with open(os.path.join(args.out_dir, "nq_train_with_table_rel_upsampled.json"), 'w') as f:
        json.dump(text_and_table_upsampled, f)
    
    logger.info(f"Write upsampled table+text dataset. Length of the dataset:{len(text_and_table_upsampled)}")
    with open(os.path.join(args.out_dir, "nq_train_table_only_rel_upsampled.json"), 'w') as f:
        json.dump(table_upsampled, f)
    logger.info(f"Write upsampled table dataset. Length of the dataset: {len(table_upsampled)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_train_path", help="path to the retrieval text+table json train file", 
        default="/home1/deokhk_1/research/MultiQA_data/retriever/relative_embedding/nq-rel-augmented-train.json")
    parser.add_argument("--table_train_path", help="path to the retrieval table-only json file",
        default="/home1/deokhk_1/research/MultiQA_data/retriever/relative_embedding/nq-rel-augmented-train-table.json")
    parser.add_argument("--out_dir", help="output path where the augmented encoder train/dev will be saved",
        default="/home1/deokhk_1/research/MultiQA_data/retriever/relative_embedding/")

    args = parser.parse_args()
    logger = logging.getLogger()
    setup_logger(logger)

    main(args, logger)
