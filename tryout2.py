import sqlite3
import os
from model import Model
import pandas as pd
import numpy as np



if __name__ == '__main__':
    model_types = ['LSTM', 'BILSTM']

    DATABASE_NAME = os.path.join('output', 'Models.db')
    conn = sqlite3.connect(DATABASE_NAME)

    cursor = conn.cursor()
    security = "ETHUSDT_1h_FUTURES"
    query = f"SELECT mae_score, mse_score, r2_score FROM {security}_LSTM where time_step=72"
    cursor.execute(query)
    results = cursor.fetchall()

    mae_scores = [item[0] for item in results]
    mse_scores = [item[1] for item in results]
    r2_scores = [item[2] for item in results]
    mae_means = np.mean(mae_scores)
    mse_means = np.mean(mse_scores) / 1000000
    r2_means = np.mean(r2_scores) / 1000000
    mae_max = max(mae_scores)
    mse_max = max(mse_scores) / 1000000
    r2_max_abs = max(np.abs(r2_scores)) / 1000000
    print(f"Mae Scores Means: {mae_means:.2f}    Mae Scores Max: {mae_max:.2f}")
    print(f"Mse Scores Means: {mse_means:.3f}M    Mse Scores Max: {mse_max:.3f}M")
    print(f"R2 Scores Means: {r2_means:.2f}M    R2 Scores Max (abs): {r2_max_abs:.2f}M")





        

