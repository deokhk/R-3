import json
import pickle
import csv
from unicodedata import category 
from transformers.tokenization_bert import BertTokenizer
from tqdm import tqdm
from dpr.data.qa_validation import answer_count

def filter_table(table_list):
    """
    filter out table which does not satisfy these desiderata:
    1. Number of row <= 100
    2. Number of column (header element) <= 50
    3. Number of cell values <= 500
    """
    filtered_table = []
    for tb in table_list:
        tb_object = json.loads(tb)
        row_num = len(tb_object['rows'])
        header_num = len(tb_object['columns'])
        value_num = len(tb_object['rows']) * len(tb_object['columns'])
        if row_num <= 100 and header_num <= 50 and value_num <= 500:
            filtered_table.append(tb_object)
    print("Remaining table size: {}".format(len(filtered_table)))
    with open('filtered_table.json', 'w') as f:
        json.dump(filtered_table, f)

def interactions_visualize_and_merge(interaction_loc):
    """
    Print a single interaction json object.
    We print train, dev, test interactions
    Return the merged interaction
    """
    train_it_loc = interaction_loc + "/train.jsonl"
    dev_it_loc = interaction_loc + "/dev.jsonl"
    test_it_loc = interaction_loc + "/test.jsonl"
    interaction_names = ["train", "dev", "test"]
    interaction_locations = [train_it_loc, dev_it_loc, test_it_loc]
    merged_list = []
    for (name, location) in zip(interaction_names, interaction_locations):
        print(f"=== Interaction_{name} ===")
        with open(location, 'r') as interaction_file:
            interaction_list = list(interaction_file)
        for it_object in interaction_list:
            it_object = json.loads(it_object)
            for key in it_object.keys():
                print("{}: {}".format(key, it_object[key]))
            break
        merged_list += interaction_list
    print("Merged list created. Total length: {}".format(len(merged_list)))
    return merged_list

def filter_interactions(merged_interactions, filtered_table):
    """
    Filter q/a pairs if its context is not in the filtered tables.
    """
    filtered_table_ids = set()
    for tb in filtered_table:
        filtered_table_ids.add(tb["tableId"])

    filtered_interactions = []
    print(f"Interaction size before filtering: {len(merged_interactions)}")
    for it in merged_interactions:
        it = json.loads(it)
        if it["table"]["tableId"] in filtered_table_ids:
            filtered_interactions.append(it)
    print(f"Interaction size after filtering : {len(filtered_interactions)}")
    return filtered_interactions

def generate_table_qa_interactions(filtered_interactions, dpr_train_loc, dpr_dev_loc, nq_open_train_loc, nq_open_dev_loc):
    """
    filter & split filtered interactions.
    table_train = (NQ_open_train – DPR_train) ∩ TAPAS_filtered
    table_dev = (NQ_dev_train – DPR_dev) ∩ TAPAS_filtered
    """
    table_q = set()
    for it in filtered_interactions:
        question_v = it["questions"][0]["originalText"].strip()
        table_q.add(question_v) 
    print("Number of questions collected from interaction file: {}".format(len(table_q)))

    with open(dpr_train_loc, 'r') as f:
        dpr_train = json.load(f)
    with open(dpr_dev_loc, 'r') as f:
        dpr_dev = json.load(f)

    dpr_q_train = set()
    dpr_q_dev = set()
    for data in dpr_train:
        dpr_q_train.add(data["question"].strip())
    for data in dpr_dev:
        dpr_q_dev.add(data["question"].strip())

    nq_open_q_train = set()
    nq_open_q_dev = set()

    with open(nq_open_train_loc, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            question = row[0]
            nq_open_q_train.add(question.strip())

    with open(nq_open_dev_loc, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            question = row[0]
            nq_open_q_dev.add(question.strip())

    table_train_q = (nq_open_q_train.difference(dpr_q_train)).intersection(table_q) 
    table_dev_q = (nq_open_q_dev.difference(dpr_q_dev)).intersection(table_q) 

    table_train = []
    table_dev = []
    for it in filtered_interactions:
        question_v = it["questions"][0]["originalText"].strip()
        if question_v in table_train_q:
            table_train.append(it)
        elif question_v in table_dev_q:
            table_dev.append(it)
        else:
            pass
    
    print(f"Number of q/a pairs in table_train:{len(table_train)}")
    print(f"Number of q/a pairs in table_dev:{len(table_dev)}")
    return (table_train, table_dev)

def _add_special_tokens(tokenizer, special_tokens):
    print("Adding special tokens %s", special_tokens)
    special_tokens_num = len(special_tokens)
    # TODO: this is a hack-y logic that uses some private tokenizer structure which can be changed in HF code
    assert special_tokens_num < 50
    unused_ids = [tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)]
    print(f"Utilizing the following unused token ids {unused_ids}")

    for idx, id in enumerate(unused_ids):
        del tokenizer.vocab["[unused{}]".format(idx)]
        tokenizer.vocab[special_tokens[idx]] = id
        tokenizer.ids_to_tokens[id] = special_tokens[idx]

    tokenizer._additional_special_tokens = list(special_tokens)
    print(
        f"Added special tokenizer.additional_special_tokens {tokenizer.additional_special_tokens}"
    )
    print(f"Tokenizer's all_special_tokens {tokenizer.all_special_tokens}")

def linearize_column(column_list):
    """
    Given a column list, return linearized column string.
    """
    linearized_column =""
    cl = len(column_list)
    for i, column in enumerate(column_list):
        linearized_column +=column
        if i != cl-1:
            linearized_column+=" "
    return linearized_column

def gen_table_passages(table, tokenizer):
    """
    Given a table, linearize the contents of table and create passage with 100 tokens each.
    Also, generate column, row ids and segment_ids for each passage (NOTE: passage with form "columns [SEP] values")
    Return: (linearized_column, psg_list, column_ids_list, row_ids_list, segment_id_list)
    """
    # 우선 [C_SEP], [V_SEP], [R_SEP]과 같은 special token을 끼워넣은 상태로 column/row/segment id를 생성하고
    # 추후 위 special token들을 [C_SEP], [V_SEP]의 경우 ","로, [R_SEP]의 경우 "\n" 으로 변화시키자.
    columns =[]
    for i, column in enumerate(table["columns"]):
        columns.append(column["text"])
        if i != len(table["columns"])-1:
            columns.append("[C_SEP]")
    linearized_column = linearize_column(columns)
    
    preceding_title_string = " [CLS] " + table["documentTitle"] + " [SEP] "
    passage_schema_string = linearized_column + " [SEP] "
    preceding_title = tokenizer.tokenize(preceding_title_string)
    schema = tokenizer.tokenize(passage_schema_string)
    
    value_column_ids = []
    value_row_ids = []
    linearized_value = ""
    row_ids_for_schema = [0 for i in range(len(preceding_title+schema))]
    column_ids_for_schema = [0 for i in range(len(preceding_title))]
    schema_column_id = 1
    
    for tok in schema:
        if tok == "[C_SEP]":
            schema_column_id+=1
            column_ids_for_schema.append(0)
        elif tok == "[SEP]":
            if tok == schema[-1]:
                column_ids_for_schema.append(0)
                break
            else:
                raise ValueError("Schema string contains [SEP] in the position that is not the end.")
        else:
            column_ids_for_schema.append(schema_column_id)
    
    rows = table["rows"]
    for i, row in enumerate(rows):
        row_value = row["cells"]
        for j, value in enumerate(row_value):
            linearized_value += value["text"] 
            if j != len(row_value)-1:
                linearized_value += " [V_SEP] "
        if i != len(rows)-1:
            linearized_value += " [R_SEP] "
    linearized_value = tokenizer.tokenize(linearized_value)

    # generate column ids for values
    value_column_id = 1
    for tok in linearized_value:
        if tok == "[R_SEP]":
            value_column_id = 1
            value_column_ids.append(0)
        elif tok == "[V_SEP]":
            value_column_id+=1
            value_column_ids.append(0)
        else:
            value_column_ids.append(value_column_id)
    # generate row ids for values
    value_row_id = 1
    for tok in linearized_value:
        if tok == "[R_SEP]":
            value_row_id+=1
            value_row_ids.append(0)
        elif tok == "[V_SEP]":
            value_row_ids.append(0)
        else:
            value_row_ids.append(value_row_id)
    
    # split the linearized value into passages, each with 100 token.
    vlen = len(linearized_value)
    quotient = vlen // 100
    remainder = vlen % 100
    psg_list = []
    column_ids_list = []
    row_ids_list = []

    assert len(linearized_value) == len(value_column_ids) == len(value_row_ids), "Linearized value and value column/row must be the same length"
    """
    Check whether the generate row/column ids satisfy those conditions
    1. Number of row <= 100
    2. Number of column (header element) <= 50
    """
    assert max(value_row_ids) <= 100, "Violate 1. Number of row <= 100"
    assert max(value_column_ids) <= 50, "Violate 2. Number of column <= 50"

    if quotient == 0:
        psg_list.append(linearized_value[0:remainder])
        column_ids_list.append(column_ids_for_schema + value_column_ids[0:remainder])
        row_ids_list.append(row_ids_for_schema + value_row_ids[0:remainder])
    else:
        if remainder == 0:
            for i in range(quotient):
                psg_list.append(linearized_value[i*100:(i+1)*100])
                column_ids_list.append(column_ids_for_schema + value_column_ids[i*100:(i+1)*100])
                row_ids_list.append(row_ids_for_schema + value_row_ids[i*100:(i+1)*100])
        else:
            for i in range(quotient):
                psg_list.append(linearized_value[i*100:(i+1)*100])
                column_ids_list.append(column_ids_for_schema + value_column_ids[i*100:(i+1)*100])
                row_ids_list.append(row_ids_for_schema + value_row_ids[i*100:(i+1)*100])
                if i == quotient-1:
                    psg_list.append(linearized_value[(i+1)*100:(i+1)*100+remainder])
                    column_ids_list.append(column_ids_for_schema + value_column_ids[(i+1)*100:(i+1)*100+remainder])
                    row_ids_list.append(row_ids_for_schema + value_row_ids[(i+1)*100:(i+1)*100+remainder])


    # Replace [C_SEP] to ","
    linearized_column = linearized_column.replace("[C_SEP]", ",")
    # Replace [V_SEP] to "," and [R_SEP] to "[SEP]"
    value_replacements = {"[V_SEP]":",","[R_SEP]":"[SEP]"}
    replacer = value_replacements.get
    for idx, psg in enumerate(psg_list):
        psg_list[idx] = [replacer(elem,elem) for elem in psg]
    return (linearized_column, psg_list, column_ids_list, row_ids_list)

def generate_retrieval_data_without_hn(interactions, tokenizer, type, table_passage_loc):
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

    # Generate text to psg_id mapping for put passage id into the passage in the generated train file.
    text_to_psg_id = {}
    with open(table_passage_loc, 'r') as file:
        # Table passage file doesn't have a header.
        table_passages = csv.reader(file, delimiter="\t")
        for row in table_passages:
            psg_id = row[0]
            text = row[1].strip()
            text_to_psg_id[text] = psg_id

    tb_retrieval_data = []
    count = 0
    for it_object in tqdm(interactions):
        data = {}
        table = it_object["table"]
        title = table["documentTitle"]
        question = it_object["questions"][0]["originalText"]
        answers = it_object["questions"][0]["answer"]["answerTexts"]
        
        data["question"] = question
        data["answers"] = answers
        
        positive_context = {}
        linearized_column, psg_list, _, _, _  = gen_table_passages(table, tokenizer)
        gold_psg = ""
        max_answer_num = 0
        for psg in psg_list:
            psg = tokenizer.decode(tokenizer.convert_tokens_to_ids(psg))
            num_answers_found = answer_count(answers, psg, tokenizer)
            if num_answers_found > max_answer_num:
                max_answer_num = num_answers_found
                gold_psg = psg
        
        gold_text = linearized_column + " [SEP] " + gold_psg
        positive_context["title"] = title
        positive_context["text"] = gold_text
        is_gold_exist = True
        if gold_text.strip() in text_to_psg_id:
            positive_context["passage_id"] = text_to_psg_id[gold_text.strip()]
        else:
            is_gold_exist = False
        data["positive_ctxs"] = [positive_context]
        if is_gold_exist:
            tb_retrieval_data.append(data)

    if type == "train":
        with open("table_train_without_special_token.json", "w") as f:
            json.dump(tb_retrieval_data, f)
    elif type == "dev":
        with open("table_dev_without_special_token.json", "w") as f:
            json.dump(tb_retrieval_data, f)
    print("Saving of retrieval data done.")
    print(f"Number of q/a pairs in {type} : {len(tb_retrieval_data)}")

def main():
    table_loc = "/home/deokhk/research/MultiQA/dataset/NQ_tables/tables/tables.jsonl"
    interaction_loc = "/home/deokhk/research/MultiQA/dataset/NQ_tables/interactions"
    with open(table_loc, 'r') as tb_file:
        tb_list = list(tb_file)
    # filter_table(tb_list)
    filtered_table_loc = "/home/deokhk/research/MultiQA/script/filtered_table.json"
    with open(filtered_table_loc, 'r') as data:
        filtered_tables = json.load(data)
    merged_interactions = interactions_visualize_and_merge(interaction_loc)
    filtered_interactions = filter_interactions(merged_interactions, filtered_tables)

    dpr_train_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-train.json"
    dpr_dev_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/nq-dev.json"
    nq_open_train_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/qas/nq-train.csv"
    nq_open_dev_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/retriever/qas/nq-dev.csv"
    table_train_interactions, table_dev_interactions = generate_table_qa_interactions(filtered_interactions, dpr_train_loc, dpr_dev_loc, nq_open_train_loc, nq_open_dev_loc)

    # Generate table passages & column/row/segment ids for each passage
    # This additional token is required to create temporary tokenized passage for row/column/segment id creations
    print("Now generating table passages.")
    special_tokens = ["[C_SEP]", "[V_SEP]", "[R_SEP]"]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    _add_special_tokens(tokenizer, special_tokens)

    count = 0
    psg_count = 21015325 # Since the last number of wikipedia split is 21015324

    f = open("/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/wikipedia_split/table_w100_without_special_token.tsv", "wt")
    tsv_writer = csv.writer(f, delimiter='\t')

    total_column_ids_list = []
    total_row_ids_list = []
    for table in tqdm(filtered_tables):
        linearized_column, psg_list, column_ids_list, row_ids_list = gen_table_passages(table, tokenizer)
        for psg in psg_list:
            psg = tokenizer.decode(tokenizer.convert_tokens_to_ids(psg))
            tsv_writer.writerow([psg_count, linearized_column + " [SEP] " + psg, table["documentTitle"]])
            psg_count+=1
        total_column_ids_list += column_ids_list
        total_row_ids_list += row_ids_list
        count+=1

    print(f"Total number of table linearized :{count}")
    print(f"Total passage created : {psg_count-21015325}")

    with open("column_ids_list_without_special_token.pickle", "wb") as fw:
        pickle.dump(total_column_ids_list, fw)
    print(f"Saved column ids for passages")

    with open("row_ids_list_without_special_token.pickle", "wb") as fw:
        pickle.dump(total_row_ids_list, fw)
    print(f"Saved row ids for passages")


    f.close()

    table_passage_loc = "/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/wikipedia_split/table_w100_without_special_token.tsv"
    generate_retrieval_data_without_hn(table_train_interactions, tokenizer, "train", table_passage_loc)
    generate_retrieval_data_without_hn(table_dev_interactions, tokenizer, "dev", table_passage_loc)

if __name__ == "__main__":
    main()
    

