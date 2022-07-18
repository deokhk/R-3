import argparse
import json
import csv
import pickle
import logging
import pandas as pd
from dpr.options import setup_logger

def main(args, logger):

    with open(args.column_file_loc, 'rb') as f:
        column_list = pickle.load(f)
    
    with open(args.row_file_loc, 'rb') as f:
        row_list = pickle.load(f)
    logger.info("Loaded column/row files.")

    with open(args.context_path, 'r') as csvinput:
        with open(args.output_dir + "psg_table_w100_with_relational_info.tsv", 'w') as csvoutput:
            writer = csv.writer(csvoutput, delimiter='\t', lineterminator='\n')
            reader = csv.reader(csvinput, delimiter='\t')

            all = []
            row = next(reader)
            row.append("column_ids")
            row.append("row_ids")
            all.append(row)

            for row in reader:
                if int(row[0]) >= 21015325:
                    # table passages
                    row.append(column_list[int(row[0])-21015325])
                    row.append(row_list[int(row[0])-21015325])
                else:
                    # text passages
                    row.append(None)
                    row.append(None)
                all.append(row)
            writer.writerows(all)
    logger.info("Done writing augmented csv file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_path", help="path to the csv context file ",
        default = "/home/deokhk/research/MultiQA_data/wikipedia_split/psg_table_w100_without_special_token.tsv")
    parser.add_argument("--column_file_loc", help="path to column file",
        default="/home/deokhk/research/MultiQA/model/DPR/column_ids_list_without_special_token.pickle")
    parser.add_argument("--row_file_loc", help="path to row file",
        default="/home/deokhk/research/MultiQA/model/DPR/row_ids_list_without_special_token.pickle")
    parser.add_argument("--output_dir", help="output path where the augmented context will be saved",
        default="/home/deokhk/research/MultiQA_data/wikipedia_split/")
    args = parser.parse_args()
    logger = logging.getLogger()
    setup_logger(logger)

    main(args, logger)