import json

def calc_header_statistics(table_list):
    """

    return header statistics.
    header statistics is a dictionary where key is a number of elements in the header
    and the value is a number of tables having the same number of elements in the header as the key.
    
    """
    header_num_from_each_tables = []
    for tb in table_list:
        tb_object = json.loads(tb)
        header_num = len(tb_object['columns'])
        header_num_from_each_tables.append(header_num)
    unique_elem = set(header_num_from_each_tables)
    header_stat = {}
    for elem in unique_elem:
        header_stat[elem] = header_num_from_each_tables.count(elem)
        
    return header_stat

def calc_row_statistics(table_list):
    """

    return row statistics.
    row statistics is a dictionary where key is a number of elements in the row
    and the value is a number of tables having the same number of elements in the row as the key.

    """
    row_num_from_each_tables = []
    for tb in table_list:
        tb_object = json.loads(tb)
        row_num = len(tb_object['rows'])
        row_num_from_each_tables.append(row_num)
    unique_elem = set(row_num_from_each_tables)
    row_stat = {}
    for elem in unique_elem:
        row_stat[elem] = row_num_from_each_tables.count(elem)

    return row_stat

def calc_value_statistics(table_list):
    """

    return value statistics.
    value statistics is a dictionary where key is a number of cells in the table
    and the value is a number of tables having the same number of cells as the key.

    """
    value_num_from_each_tables = []
    for tb in table_list:
        tb_object = json.loads(tb)
        value_num = len(tb_object['rows']) * len(tb_object['columns'])
        value_num_from_each_tables.append(value_num)
    unique_elem = set(value_num_from_each_tables)
    value_stat = {}
    for elem in unique_elem:
        value_stat[elem] = value_num_from_each_tables.count(elem)

    return value_stat


def cumulative_percentage_row(row_stat):
    """
    Print the cumulative percentage of row statistics up to a given integer n.
    """
    print("===Calculate cumulative percentage of row===")
    n = int(input('input n: '))
    num_tables = 0
    num_tables_below_n = 0
    for i in sorted(row_stat.keys()):
        if i<=n:
            num_tables_below_n += row_stat[i]
        num_tables += row_stat[i]
    cum_percentage = (num_tables_below_n / num_tables) * 100
    print("Cumulative percentage of row up to {} is {}%.".format(n, cum_percentage))


def cumulative_percentage_header(header_stat):
    """
    Print the cumulative percentage of header statistics up to a given integer n.
    """
    print("===Calculate cumulative percentage of header===")
    n = int(input('input n: '))
    num_tables = 0
    num_tables_below_n = 0
    for i in sorted(header_stat.keys()):
        if i<=n:
            num_tables_below_n += header_stat[i]
        num_tables += header_stat[i]
    cum_percentage = (num_tables_below_n / num_tables) * 100
    print("Cumulative percentage of header up to {} is {}%.".format(n, cum_percentage))
    
def cumulative_percentage_value(value_stat):
    """
    Print the cumulative percentage of value statistics up to a given integer n.
    """
    print("===Calculate cumulative percentage of value===")
    n = int(input('input n: '))
    num_tables = 0
    num_tables_below_n = 0
    for i in sorted(value_stat.keys()):
        if i<=n:
            num_tables_below_n += value_stat[i]
        num_tables += value_stat[i]
    cum_percentage = (num_tables_below_n / num_tables) * 100
    print("Cumulative percentage of value up to {} is {}%.".format(n, cum_percentage))

def table_visualize(table_list):
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

def interactions_visualize_and_merge(interaction_loc):
    """
    Print a single interaction json object.
    We print train, dev, test interactions
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
    with open('merged_interaction.json', 'w') as f:
        json.dump(merged_list, f)
    print("Merged list saved. Total length: {}".format(len(merged_list)))

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

def main():
    table_loc = "/home/deokhk/research/multiqa/dataset/NQ_tables/tables/tables.jsonl"
    interaction_loc = "/home/deokhk/research/multiqa/dataset/NQ_tables/interactions"
    with open(table_loc, 'r') as tb_file:
        tb_list = list(tb_file)

    # print("Successfully loaded the table file")
    # table_visualize(tb_list)
    # header_stat = calc_header_statistics(tb_list)
    # row_stat = calc_row_statistics(tb_list)
    # value_stat = calc_value_statistics(tb_list)

    # cumulative_percentage_row(row_stat)
    # cumulative_percentage_header(header_stat)
    # cumulative_percentage_value(value_stat)
    # interactions_visualize_and_merge(interaction_loc)

    filter_table(tb_list)

if __name__ == '__main__':
    main()