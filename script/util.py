import json
import pandas as pd

def print_table_object(table_list):
    """
    Print the table json object at corresponding index.
    """
    
    tb_num = int(input("Type the index of the table you want to see: "))
    for i, tb_object in enumerate(table_list):
        if i>tb_num:
            break
        if i==tb_num:
            tb_object = json.loads(tb_object)
            for key in tb_object.keys():
                print("{}: {}".format(key, tb_object[key]))

def extract_table_dataframe_from_interaction(interaction_object):
    """
    Given an interaction object, extract a table dataframe object
    """
    table = interaction_object["table"]
    columns = table["columns"]
    column_list = []
    for column in columns:
        column_list.append(column["text"])
    
    rows = table["rows"]
    rows_list = []
    for row in rows:
        cells = row["cells"]
        row_values = []
        for cell in cells:
            row_values.append(cell["text"])
        rows_list.append(row_values)
    frame = pd.DataFrame(rows_list, columns=column_list)
    return frame
