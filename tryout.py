import sqlite3
import os
from model import Model
import pandas as pd
import numpy as np
from lstm_plotter import LSTMPlotter


def calculate_residuals(control_file: str, predictions: np.ndarray, time_step: int):
    try:
        control_df = Model.load_csv(control_file)
        control_df = control_df.head(time_step)
        control_df = control_df[['close']]

        if len(control_df) != len(predictions):
            print("Error: Lengths of control_df and predictions do not match.")
            exit(1)

        # Assuming control_df has only one column
        actual_values = control_df.values.reshape(-1, 1)
        actual_values = [item[0] for item in actual_values]
        
        residuals = actual_values - predictions

        residuals_percentage = (residuals / actual_values) * 100

        return {
            'residuals': residuals,
            'residuals_percentage': residuals_percentage,
        }
    except Exception as e:
        print(f"An error occurred when calculating residuals: {str(e)}")


if __name__ == '__main__':
    csv_path = os.path.join('input', 'csv_files', 'ETHUSDT_1h_afterwards.csv')
    residuals_path = os.path.join('output', 'ETHUSDT_1h_FUTURES', 'fig', 'residuals')
    plotter = LSTMPlotter()
    model_types = ['LSTM', 'BILSTM']

    params = Model.load_json_data('new_params.json')
    model_params_str = Model.dict_to_tuple(params['model_params'])

    columns = list(params['model_params'].keys())
    columns.insert(0, 'id')
    columns.append('predictions')
    columns_str = ', '.join(columns)


    DATABASE_NAME = os.path.join('output', 'Models.db')
    conn = sqlite3.connect(DATABASE_NAME)

    lstm_ids = [41, 65]
    bilstm_ids = [40, 48, 50]
    lstm_residuals = []
    lstm_residuals_percentages = []
    bilstm_residuals = []
    bilstm_residuals_percentages = []

    cursor = conn.cursor()

    table_name = f"ETHUSDT_1h_FUTURES_LSTM"
    for id in lstm_ids:
        query = f'SELECT {columns_str} FROM {table_name} where id={id}'

        cursor.execute(query)
        results = cursor.fetchall()
        i = 0

        for row in results:
            params_str = str(row[1:])
            predictions = np.array(eval(row[-1]))
            time_step = int(row[2])
            residuals = calculate_residuals(csv_path, predictions, time_step)
            lstm_residuals.append(residuals['residuals'])
            lstm_residuals_percentages.append(residuals['residuals_percentage'])

            # plotter.plot_residuals(residuals_path, model_type, i+1, params_str, predictions, residuals['residuals'], residuals['residuals_percentage'])
            # i += 1
    
    table_name = f"ETHUSDT_1h_FUTURES_BILSTM"
    for id in bilstm_ids:
        query = f'SELECT {columns_str} FROM {table_name} where id={id}'

        cursor.execute(query)
        results = cursor.fetchall()
        i = 0

        for row in results:
            params_str = str(row[1:])
            predictions = np.array(eval(row[-1]))
            time_step = int(row[2])
            residuals = calculate_residuals(csv_path, predictions, time_step)
            bilstm_residuals.append(residuals['residuals'])
            bilstm_residuals_percentages.append(residuals['residuals_percentage'])

            # plotter.plot_residuals(residuals_path, model_type, i+1, params_str, predictions, residuals['residuals'], residuals['residuals_percentage'])
            # i += 1
    

    # print(bilstm_residuals)
    for residual_percentages in bilstm_residuals_percentages:
        print(np.mean((residual_percentages)))
    for residual_percentages in lstm_residuals_percentages:
        print(np.mean((residual_percentages)))

        




        

