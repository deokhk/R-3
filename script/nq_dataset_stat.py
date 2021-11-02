import json
import csv

def compare_nq_original_and_retriever_dataset(original_file_loc, dpr_file_loc):
    print("Compare {} file".format(original_file_loc[-10:]))
    with open(original_file_loc, 'r') as ori_file:
        reader = csv.reader(ori_file)
        lines = len(list(reader))
    print("Number of q/a pair in original file: {}".format(lines))

    with open(dpr_file_loc, 'r') as dpr_file:
        ori_data = json.load(dpr_file)
        lines = len(ori_data)
    print("Number of q/a pair in dpr-used file: {}".format(lines))

def print_dataset_key_stat(dataset):
    single_data = dataset[0]
    for key in single_data.keys():
        if not isinstance(single_data[key], list):
            print(f"Name of the key: {key}, value: {single_data[key]}")
        else:
            print(f"Name of the key: {key}, number of elem in it: {len(single_data[key])}")


def main():
    dev_dataset_loc = "/home/deokhk/research/multiqa/model/DPR/dpr/downloads/data/retriever/nq-dev.json"
    dev_dataset_ori_loc = "/home/deokhk/research/multiqa/model/DPR/dpr/downloads/data/retriever/qas/nq-dev.csv"
    
    # with open(dev_dataset_loc, 'r') as file:
    #     dev_data = json.load(file)

    # print("Successfully loaded the dev data")
    # print_dataset_key_stat(dev_data)
    compare_nq_original_and_retriever_dataset(dev_dataset_ori_loc, dev_dataset_loc)

    train_dataset_loc = "/home/deokhk/research/multiqa/model/DPR/dpr/downloads/data/retriever/nq-train.json"
    train_dataset_ori_loc = "/home/deokhk/research/multiqa/model/DPR/dpr/downloads/data/retriever/qas/nq-train.csv"
    
    compare_nq_original_and_retriever_dataset(train_dataset_ori_loc, train_dataset_loc)

if __name__ == '__main__':
    main()