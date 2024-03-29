import json
import csv
import pickle
from transformers import BertTokenizerFast
from generate_table_passages import gen_table_passages
from tqdm import tqdm


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

def filter_and_save_table_qa_dataset(table_q_list, nq_open_train_loc, nq_open_dev_loc, merged_interaction_loc):
    """
    Filter table QA dataset that belongs to NQ_open_train, NQ_open_dev only.
    Return: filtered interaction list with question only.
    Save: filtered interaction itself.
    """
    print("==== Now filtering table_QA dataset ====")
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
    
    with open(merged_interaction_loc, 'r') as f:
        merged_interaction_list = json.load(f)
    
    filtered_interaction_list = []

    for it_object in merged_interaction_list:
        it_object = json.loads(it_object)
        question = it_object["questions"][0]["originalText"]
        if question.strip() in filtered_list:
            filtered_interaction_list.append(it_object)

    with open("/home/deokhk/research/MultiQA/dataset/NQ_tables/interactions/filtered_interaction.json", "w+") as f:
        json.dump(filtered_interaction_list, f)
    print("Saving of filtered interaction list completed. Length : {}".format(len(filtered_interaction_list)))
    return filtered_interaction_list

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

def visualize_overlapped_qa_pair(dpr_loc, filtered_table_q_list, merged_interaction_loc):
    """
    Print and save overlapped pairs in DPR_NQ data and filtered table_qa_data.
    """
    with open(dpr_loc, 'r') as dpr_file:
        dpr_data = json.load(dpr_file)
    with open(merged_interaction_loc, 'r') as interaction_file:
        merged_interaction_data = json.load(interaction_file)

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
        
def remove_duplicated_qa_pair(dpr_train_loc, dpr_dev_loc, filtered_interaction):
    """
    Remove qa pair from filtered_interaction where it is already exists in DPR
    """
    print("==== Now removing duplicated qa pair in filtered_list ====")
    with open(dpr_train_loc, 'r') as f:
        dpr_train = json.load(f)
    with open(dpr_dev_loc, 'r') as f:
        dpr_dev = json.load(f)
    
    dpr_q = set()
    for data in dpr_train:
        dpr_q.add(data["question"].strip())
    for data in dpr_dev:
        dpr_q.add(data["question"].strip())
    
    duplicated_removed_interaction = []
    print(f"Number of QA pair before duplication removed: {len(filtered_interaction)}")
    for it_object in filtered_interaction:
        q = it_object["questions"][0]["originalText"].strip()
        if q not in dpr_q:
            duplicated_removed_interaction.append(it_object)
    print(f"Number of QA pair after duplication removed: {len(duplicated_removed_interaction)}")
    with open("/home/deokhk/research/MultiQA/dataset/NQ_tables/interactions/dup_removed_interaction.json", "w+") as f:
        json.dump(duplicated_removed_interaction, f)
    print("Saving of duplicated_removed_interaction completed")

    return duplicated_removed_interaction

def generate_retrieval_data_without_hn(dup_removed_interaction, tokenizer):
    """
    Generate table retrieval data, without hard negatives.
    Return: 
    [
        {
            "question": "....",
            "answers": ["...", "...", "..."],
            "positive_ctxs": [{
                "title": "...",
                "text": "...."
            }]
        },
        ...
    ]
    """
    print("==== Now Generating table retrieval dataset without hard negatives. ====")

    tb_retrieval_data = []
    count = 0
    for it_object in tqdm(dup_removed_interaction):
        data = {}
        table = it_object["table"]
        title = table["documentTitle"]
        question = it_object["questions"][0]["originalText"]
        answers = it_object["questions"][0]["answer"]["answerTexts"]
        
        data["question"] = question
        data["answers"] = answers
        
        positive_context = {}
        linearized_column, psg_list = gen_table_passages(table, tokenizer)
        gold_psg = ""
        max_answer_num = 0
        for psg in psg_list:
            psg = tokenizer.decode(tokenizer.convert_tokens_to_ids(psg))
            num_answers_found = 0
            for answer in answers:
                if answer in psg:
                    num_answers_found+=1
            if num_answers_found > max_answer_num:
                max_answer_num = num_answers_found
                gold_psg = psg
        
        gold_text = linearized_column + " [SEP] " + gold_psg
        positive_context["title"] = title
        positive_context["text"] = gold_text
        data["positive_ctxs"] = [positive_context]
        tb_retrieval_data.append(data)
        
    dlen = len(tb_retrieval_data)
    train_data = tb_retrieval_data[0:(dlen//10)*9]
    dev_data = tb_retrieval_data[(dlen//10)*9:]
    with open("/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/table_train.json", "w") as f:
        json.dump(train_data, f)
    with open("/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/table_dev.json", "w") as f:
        json.dump(dev_data, f)
    print("Saving of retrieval data done.")
    print(f"Number of training data: Train: {len(train_data)} , Dev: {len(dev_data)}")


def main():
    merged_nq_table_data_loc = "/home/deokhk/research/MultiQA/dataset/NQ_tables/interactions/merged_interaction.json"
    dpr_nq_train_data_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-train.json"
    dpr_nq_dev_data_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-dev.json"
    original_nq_train_data_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/qas/nq-train.csv"
    original_nq_dev_data_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/qas/nq-dev.csv"
    original_nq_test_data_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/qas/nq-test.csv"

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased", additional_special_tokens = ['[C_SEP]', '[V_SEP]', '[R_SEP]'])
    table_q_list = extract_q_from_interaction_file_and_save(merged_nq_table_data_loc)
    filtered_interaction = filter_and_save_table_qa_dataset(table_q_list, original_nq_train_data_loc, original_nq_dev_data_loc, merged_nq_table_data_loc)
    dup_removed_interaction = remove_duplicated_qa_pair(dpr_nq_train_data_loc, dpr_nq_dev_data_loc, filtered_interaction)    
    generate_retrieval_data_without_hn(dup_removed_interaction, tokenizer)
if __name__ == '__main__':
    main()