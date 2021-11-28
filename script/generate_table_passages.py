from transformers import BertTokenizerFast
import json
import csv
from tqdm import tqdm
from knockknock import slack_sender


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
    Return: (linearized_column, psg_list)
    """
    columns =[]
    for i, column in enumerate(table["columns"]):
        columns.append(column["text"])
        if i != len(table["columns"])-1:
            columns.append("[C_SEP]")
    linearized_column = linearize_column(columns)

    linearized_value = ""
    rows = table["rows"]
    for i, row in enumerate(rows):
        row_value = row["cells"]
        for j, value in enumerate(row_value):
            linearized_value += value["text"] 
            if j != len(row_value)-1:
                linearized_value += "[V_SEP]"
        if i != len(rows)-1:
            linearized_value += "[R_SEP]"
    linearized_value = tokenizer.tokenize(linearized_value)
    
    # split the linearized value into passages, each with 100 token.
    vlen = len(linearized_value)
    quotient = vlen // 100
    remainder = vlen % 100
    psg_list = []
    
    if quotient == 0:
        psg_list.append(linearized_value[0:remainder])
    else:
        for i in range(quotient+1):
            psg_list.append(linearized_value[i*100:(i+1)*100])
            if i == quotient and remainder:
                psg_list.append(linearized_value[(i+1)*100:(i+1)*100 + remainder])
    return (linearized_column, psg_list)

# TODO: make sure to remove webhook when this releases publicly.
@slack_sender(webhook_url="https://hooks.slack.com/services/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", channel="knock_knock")
def main():
    print("Now generating table passages.")    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased", additional_special_tokens = ['[C_SEP]', '[V_SEP]', '[R_SEP]'])
    table_data_loc = "/home/deokhk/research/MultiQA/dataset/NQ_tables/tables/tables.jsonl"
    with open(table_data_loc, 'r') as table_file:
        table_datas = list(table_file)
        
    count = 0
    psg_count = 21015325 # Since the last number of wikipedia split is 21015324

    f = open("/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/wikipedia_split/table_w100.tsv", "wt")
    tsv_writer = csv.writer(f, delimiter='\t')

    for table in tqdm(table_datas):
        table = json.loads(table)
        linearized_column, psg_list = gen_table_passages(table, tokenizer)
        for psg in psg_list:
            psg = tokenizer.decode(tokenizer.convert_tokens_to_ids(psg))
            tsv_writer.writerow([psg_count, linearized_column + " [SEP] " + psg, table["documentTitle"]])
            psg_count+=1
        
        count+=1
    print(f"Total number of table linearized :{count}")
    print(f"Total passage created : {psg_count-21015325}")
    f.close()

if __name__ == '__main__':
    main()