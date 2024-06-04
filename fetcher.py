from lstm_database import LSTMDatabase
import os
import sqlite3


if __name__ == '__main__':
    DATABASE_NAME = os.path.join('output', 'Models.db')
    bilstm_ids = [40, 48, 50]
    lstm_ids = [41, 65]
    security = "ETHUSDT_1h_FUTURES"
    bilstm_table = f"{security}_BILSTM"
    lstm_table = f"{security}_LSTM"
    conn = sqlite3.connect(DATABASE_NAME)
    bilstm_results = []
    lstm_results = []
    


    for id in bilstm_ids:
        query = f"SELECT epoch_used from {bilstm_table} where id={id}"
        curr = conn.cursor()
        curr.execute(query)
        results = curr.fetchall()
        bilstm_results.append(results)
    
    for id in lstm_ids:
        query = f"SELECT epoch_used from {lstm_table} where id={id}"
        curr = conn.cursor()
        curr.execute(query)
        results = curr.fetchall()
        lstm_results.append(results)
    
    print(bilstm_results)
    print(lstm_results)