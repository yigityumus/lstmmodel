import os
import json
import sqlite3
from datetime import datetime


# Function to read JSON files from a folder
def read_json_files_from_folder(folder_path):
    json_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            json_files.append(file_path)
    return json_files


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data


def read_from_text_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')

            i = int(parts[0].split('/')[0])

            # Extracting time as minutes and seconds
            time_as_minutes = int(parts[3][:-1])
            time_as_seconds = int(parts[4][:-2])
            total_time = time_as_minutes * 60 + time_as_seconds
            
            # Extracting date and time separately
            finish_date_str = parts[7]
            finish_time_str = parts[8]
            finish_datetime_str = f"{finish_date_str} {finish_time_str}"
            finish_date = datetime.strptime(finish_datetime_str, '%Y-%m-%d %H:%M:%S')
            
            finish_timestamp = int(finish_date.timestamp())
            epoch_used = int(parts[9][1:-1].split('/')[0])
            
            # Calculate start_timestamp and start_date
            start_timestamp = finish_timestamp - total_time
            start_date = datetime.fromtimestamp(start_timestamp)
            
            
            data.append((i, epoch_used, total_time, start_date, start_timestamp, finish_date, finish_timestamp))
    return data


def save_to_database(json_data, text_data, db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    table_name = "LSTM_models"
    
    # # Create table with auto-incrementing ID
    # cursor.execute('''CREATE TABLE IF NOT EXISTS {} (
    #                     id INTEGER PRIMARY KEY AUTOINCREMENT,
    #                     train_rate REAL,
    #                     time_step INT,
    #                     neuron INT,
    #                     dropout_rate REAL,
    #                     optimizer TEXT,
    #                     patience INT,
    #                     epoch INT,
    #                     batch_size INT,
    #                     activation TEXT,
    #                     kernel_regularizer REAL,
    #                     loss_function TEXT,
    #                     test_loss REAL,
    #                     close_data TEXT,
    #                     train_predict TEXT,
    #                     test_predict TEXT,
    #                     predictions TEXT,
    #                     epoch_used INT,
    #                     total_time INT,
    #                     start_date TEXT,
    #                     start_timestamp INT,
    #                     finish_date TEXT,
    #                     finish_timestamp INT
    #                 )'''.format(table_name))

    # # Insert data from JSON files
    # total_json = len(json_data)
    # for index, item in enumerate(json_data, 1):
    #     try:
    #         json_data = read_json_file(item)

    #         params = json_data['params']
    #         test_loss = json_data['test_loss']
    #         close_data = json.dumps(json_data['close_data'])
    #         train_predict = json.dumps(json_data['train_predict'])
    #         test_predict = json.dumps(json_data['test_predict'])
    #         predictions = json.dumps(json_data['predictions'])

    #         cursor.execute('''INSERT INTO {} (
    #                         train_rate, time_step, neuron, dropout_rate, optimizer, 
    #                         patience, epoch, batch_size, activation, kernel_regularizer, 
    #                         loss_function, test_loss, close_data, train_predict, test_predict, predictions, 
    #                         epoch_used, total_time, start_date, start_timestamp, finish_date, finish_timestamp
    #                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''.format(table_name), 
    #                    (params['train_rate'], params['time_step'], params['neuron'], params['dropout_rate'], 
    #                     params['optimizer'], params['patience'], params['epoch'], params['batch_size'], 
    #                     params['activation'], params['kernel_regularizer'], params['loss_function'], 
    #                     test_loss, close_data, train_predict, test_predict, predictions,
    #                     None, None, None, None, None, None))  # Fill placeholder for text data
            
    #         # Print progress
    #         print(f"Inserted JSON data: {index}/{total_json}")
    #     except json.JSONDecodeError as e:
    #         print(f"Error parsing JSON: {e}")
    
    
    # Insert data from text file
    total_text = len(text_data)
    for index, row in enumerate(text_data, 1):
        cursor.execute('''UPDATE {}
                            SET epoch_used=?, total_time=?, start_date=?, start_timestamp=?, finish_date=?, finish_timestamp=?  
                            WHERE id=?'''.format(table_name), 
                    (row[1], row[2], row[3], row[4], row[5], row[6], row[0]))
            
        # Print progress
        print(f"Updated text data: {index}/{total_text}")

    conn.commit()
    conn.close()



if __name__ == '__main__':
    # Example usage
    folder_path = '/Users/yigityumus/model_data_files'
    text_file_path = '/Users/yigityumus/Desktop/times_and_epochs.txt'
    db_name = "/Volumes/DATA/LSTM_DATABASE/LSTM.db"
    
    text_data = read_from_text_file(text_file_path)
    print("text data has been successfully handled.")

    json_files = read_json_files_from_folder(folder_path)
    print(json_files[0])
    print('json files has been successfully handled')

    print("starting to save to the database")
    save_to_database(json_files, text_data, db_name)
