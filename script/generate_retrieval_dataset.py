import json
import csv
import pickle

def generate_retrieval_data(qa_list, nq_open_loc):
    nq_open_data = open(nq_open_loc)
    csvreader = csv.reader(nq_open_data)

def calc_q_intersection_dpr_nq_and_table(dpr_loc, table_qa_list):
    """
    Calculate the number of questions in dpr training data that overlapped with NQ q/a pair with answers in table (table_qa_list) 
    """

    print(f"Now checking the overlapped data b/w {dpr_loc[-10:]} and merged_interaction data.")
    with open(dpr_loc, 'r') as dpr_file:
        dpr_data = json.load(dpr_file)
    dpr_q_list = []
    for data in dpr_data:
        dpr_q_list.append(data["question"].strip())
    print("Number of questions in dpr-{} dataset:{}".format(dpr_loc[-10:], len(dpr_q_list)))
    print("Number of overlapped question b/w the two: {}".format(len(list(set(table_qa_list) & set(dpr_q_list)))))


def calc_q_intersection_ori_nq_and_table(nq_loc, table_qa_list):
    """
    Calculate the number of questions in original nq data that overlapped with NQ q/a pair with answers in table (table_qa_list) 
    """

    print(f"Now checking the overlapped data b/w {nq_loc[-10:]} and merged_interaction data.")
    original_nq_q_list = []
    with open(nq_loc, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            question = row[0]
            original_nq_q_list.append(question.strip())
    print("Number of questions in nq-{} dataset:{}".format(nq_loc[-5:], len(original_nq_q_list)))
    print("Number of overlapped question b/w the two: {}".format(len(list(set(table_qa_list) & set(original_nq_q_list)))))

def filter_table_qa_dataset(table_q_list, nq_open_train_loc, nq_open_dev_loc):
    """
    Filter table QA dataset that belongs to NQ_open_train, NQ_open_dev only.
    """
    print("==== Now filtering table_QA dataset ====")
    print(f"Length before filtering: {len(table_q_list)}")
    original_nq_open_q_list = []
    with open(nq_open_train_loc, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            question = row[0]
            original_nq_open_q_list.append(question.strip())
    
    with open(nq_open_dev_loc, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            question = row[0]
            original_nq_open_q_list.append(question.strip())
    
    filtered_list = list(set(table_q_list) & set(original_nq_open_q_list))
    print(f"Length after filtering: {len(filtered_list)}")
    
    return filtered_list

def extract_q_from_interaction_file_and_save(merged_nq_table_data_loc):
    """
    From merged interaction file, extract question from each q/a pairs and save file with only questions.
    Return: table_q_list
    """
    with open(merged_nq_table_data_loc, 'r') as interaction_file:
        interaction_list = json.load(interaction_file)
    print(f"Successfully loaded the interaction file. Length of the list: {len(interaction_list)}")
    
    table_q_list = []
    for it_object in interaction_list:
        it_object = json.loads(it_object)
        question_v = it_object["questions"][0]["originalText"].strip()
        table_q_list.append(question_v) 
    print("Number of questions collected from interaction file: {}".format(len(table_q_list)))
    
    print("Saving interaction file with question only ...  ")
    interaction_q_file = open("interaction_q_file", "w")
    for elem in table_q_list:
        interaction_q_file.write(elem + "\n")
    interaction_q_file.close()
    return table_q_list


def calc_intersection(table_q_list, DPR_NQ_train_loc, DPR_NQ_dev_loc, NQ_open_train_loc, NQ_open_dev_loc, NQ_open_test_loc):
    """
    Calculate intersection b/w table_q_list and
    1. DPR_NQ_train
    2. DPR_NQ_dev
    3. NQ_open_train
    4. NQ_open_dev
    5. NQ_open_test
    , respectively.
    """
    calc_q_intersection_dpr_nq_and_table(DPR_NQ_train_loc, table_q_list)
    calc_q_intersection_dpr_nq_and_table(DPR_NQ_dev_loc, table_q_list)
    calc_q_intersection_ori_nq_and_table(NQ_open_train_loc, table_q_list)
    calc_q_intersection_ori_nq_and_table(NQ_open_dev_loc, table_q_list)
    calc_q_intersection_ori_nq_and_table(NQ_open_test_loc, table_q_list)

def visualize_overlapped_qa_pair(dpr_loc, filtered_table_q_list):
    """
    Print and save overlapped pairs in DPR_NQ data and filtered table_qa_data.
    """
    with open(dpr_loc, 'r') as dpr_file:
        dpr_data = json.load(dpr_file)
    dpr_q_list = []
    for data in dpr_data:
        dpr_q_list.append(data["question"].strip())
    
    n = int(input("How many overlapped pair do you want to see: "))
    i = 0
    for q in filtered_table_q_list:
        if q in dpr_q_list:
            # We pick this as a overlapped qa pair.
            for data in dpr_data:
                if data["question"].strip() == q:
                    print("====== In DPR ======")
                    print(data)
                    print("====== In Table QA ======")
                    print()
                else:
                    print("Didn't find the matching q/a pair. Something went wrong!")
            i+=1
            if i>n:
                break
        

def main():
    merged_nq_table_data_loc = "/home/deokhk/research/MultiQA/dataset/NQ_tables/interactions/merged_interaction.json"
    dpr_nq_train_data_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-train.json"
    dpr_nq_dev_data_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-dev.json"
    original_nq_train_data_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/qas/nq-train.csv"
    original_nq_dev_data_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/qas/nq-dev.csv"
    original_nq_test_data_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/qas/nq-test.csv"

    table_q_list = extract_q_from_interaction_file_and_save(merged_nq_table_data_loc)
    filtered_table_q_list = filter_table_qa_dataset(table_q_list, original_nq_train_data_loc, original_nq_dev_data_loc)
    
    calc_intersection(filtered_table_q_list, dpr_nq_train_data_loc, dpr_nq_dev_data_loc, original_nq_train_data_loc, original_nq_dev_data_loc, original_nq_test_data_loc)
    
        
if __name__ == '__main__':
    main()