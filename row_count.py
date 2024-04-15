import sqlite3
import pandas as pd

def count_rows_in_table(table_name, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    
    row_count = cursor.fetchone()[0]
    
    conn.close()
    
    return row_count



def check_for_duplicates(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Define the columns to check for duplicates
    columns_to_check = ['time_step', 'neuron', 'optimizer', 'patience', 'epoch', 'batch_size', 'activation', 'loss_function']

    # Construct the SQL query to check for duplicates
    query = f'''
            SELECT COUNT(*)
            FROM (
                SELECT COUNT(*) AS cnt
                FROM {table_name}
                GROUP BY {', '.join(columns_to_check)}
                HAVING cnt > 1
            ) AS duplicates
            '''

    cursor.execute(query)
    result = cursor.fetchone()[0]

    return result


def fetch_data_by_activation(db_path, table_name, activation):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Define the columns to fetch
    columns_to_fetch = ['id', 'epoch_used', 'total_time', 'start_date', 'start_timestamp', 'finish_date', 'finish_timestamp']

    # Construct the SQL query to fetch data where 'activation' == 'sigmoid'
    query = f'''
            SELECT {', '.join(columns_to_fetch)}
            FROM {table_name}
            WHERE activation = ?
            '''

    cursor.execute(query, (activation,))
    data = cursor.fetchall()

    df = pd.DataFrame(data, columns=columns_to_fetch)

    return df


def fetch_N_data(db_path, table_name, limit):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Define the columns to fetch
    columns_to_fetch = ['id', 'epoch_used', 'total_time', 'start_date', 'start_timestamp', 'finish_date', 'finish_timestamp']

    # Construct the SQL query to fetch data with a limit
    query = f'''
            SELECT {', '.join(columns_to_fetch)}
            FROM {table_name}
            ORDER BY id
            LIMIT ?
            '''

    cursor.execute(query, (limit,))
    data = cursor.fetchall()

    df = pd.DataFrame(data, columns=columns_to_fetch)

    return df



if __name__ == '__main__':
    table_name = 'LSTM_models'
    db_path = '/Volumes/DATA/LSTM/LSTM.db'

    # row_count = count_rows_in_table(table_name, db_path)
    # print(f"Number of rows in '{table_name}': {row_count}")

    # duplicates = check_for_duplicates(db_path, table_name)
    # print(f"Duplicates found: {duplicates}")

    # sigmoid_data = fetch_data_by_activation(db_path, table_name, 'sigmoid')
    # print(sigmoid_data)

    first_data = fetch_N_data(db_path, table_name, 45)
    print(first_data)


