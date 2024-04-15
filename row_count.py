import sqlite3

def count_rows_in_table(table_name, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Execute SQL query to count rows in the table
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    
    # Fetch the result
    row_count = cursor.fetchone()[0]
    
    # Close the connection
    conn.close()
    
    return row_count

# Example usage:

if __name__ == '__main__':
    table_name = 'LSTM_models'
    db_path = '/Volumes/DATA/LSTM/LSTM.db'
    row_count = count_rows_in_table(table_name, db_path)
    print(f"Number of rows in '{table_name}': {row_count}")
