import argparse
from custom_util import convert_json_to_qas_tsv

def main(args):
    convert_json_to_qas_tsv(args.train_path, args.train_out_csv)
    print("Converted retrieval train file from json to csv format")
    convert_json_to_qas_tsv(args.dev_path, args.dev_out_csv)
    print("Converted retrieval dev file from json to csv format")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help="path to the retrieval json train file")
    parser.add_argument("--dev_path", help="path to the retrieval dev json file")
    parser.add_argument("--train_out_csv", help="path to output train file in csv format")
    parser.add_argument("--dev_out_csv", help="path to output dev file in csv format")
    args = parser.parse_args()
    main(args)
