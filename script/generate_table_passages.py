from transformers import BertTokenizerFast
import json
import csv
from time import sleep
from tqdm import tqdm


def linearize_column(column_list):
    linearized_column =""
    cl = len(column_list)
    for i, column in enumerate(column_list):
        linearized_column +=column
        if i != cl-1:
            linearized_column+=" "
    return linearized_column

def main():    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased", additional_special_tokens = ['[C_SEP]', '[R_SEP]'])
    table_data_loc = "/home/deokhk/research/MultiQA/dataset/NQ_tables/tables/tables.jsonl"
    with open(table_data_loc, 'r') as table_file:
        table_datas = list(table_file)

    count = 0
    psg_count = 21015325 # Since the last number of wikipedia split is 21015324

    f = open("/home/deokhk/research/MultiQA/model/DPR/dpr/downloads/data/wikipedia_split/table_w100.tsv", "wt")
    tsv_writer = csv.writer(f, delimiter='\t')

    for table in tqdm(table_datas):
        sleep(0.25)
        table = json.loads(table)
        title = table["documentTitle"]
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
                    linearized_value += "[R_SEP]"
            if i != len(rows)-1:
                # Since "\n" is considered as whitespace, we first set delimter for each sentence as "[CLS]"
                # And replace it to "\n" afterward.
                linearized_value += "[CLS]"
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
        
        for psg in psg_list:
            psg = tokenizer.decode(tokenizer.convert_tokens_to_ids(psg))
            psg = psg.replace("[CLS]", "\n")
            tsv_writer.writerow([psg_count, linearized_column + "[SEP]" + psg, title])
            psg_count+=1
        
            
        count+=1
        if count == 2:
            break
    print(f"Total number of table linearized :{count}")
    print(f"Total passage created : {psg_count-21015325}")
    f.close()

if __name__ == '__main__':
    main()