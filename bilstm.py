import pandas as pd
import numpy as np
import datetime as dt

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Input, LSTM, Bidirectional
from keras.regularizers import l2

import itertools
import os
import itertools
import time
import sys

from functions import (generate_dataset, 
                       load_params_from_json,
                       append_to_times_and_epochs,
                       determine_frequency
                       )
from lstm_plotter import LSTMPlotter
from lstm_database import LSTMDatabase




if __name__ == '__main__':

    current_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Current time: {current_time}")

    model_type = sys.argv[1].upper()
    

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Set TensorFlow logging level to suppress informational messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
    # 0: all messages, 1: INFO messages, 2: INFO and WARNING messages, 3: INFO, WARNING, and ERROR messages


    parameters = load_params_from_json('new_params.json')
    parameter_list = parameters['parameter_list']
    interval = parameters['data']['interval']
    DATABASE_NAME = parameters['database_name']


    ticker_fullname = f"{parameters['data']['security']}_{interval}_{parameters['data']['data_type']}"

    csv_path = os.path.join('csv_files', f'{ticker_fullname}.csv')
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date']) # To fix the x axis dates on the graphs. Store them as pd.datetime instead of str.
    # print(df.head())

    close_data = df[['close']]
    print(f"close_data shape: {close_data.shape}")

    param_combinations = list(itertools.product(*parameter_list.values()))
    total_combinations = len(param_combinations)
    print(f"param_combinations: {total_combinations} different options.")


    # Iterate over all parameter combinations
    for i, params in enumerate(param_combinations):
        start_time = time.time()
        
        train_rate, time_step, neuron, dropout_rate, optimizer, patience, epoch, batch_size, activation, kernel_regularizer, loss_function = params

        
        train_size = int(len(close_data) * train_rate)
        train_data = close_data[:train_size]
        test_data = close_data[train_size:]
        
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        train_data_scaled = scaler.fit_transform(train_data)
        test_data_scaled = scaler.transform(test_data)

        # Generate datasets
        X_train, y_train = generate_dataset(train_data_scaled, time_step)
        X_test, y_test = generate_dataset(test_data_scaled, time_step)


        # - - - - - - - - - - - - - - - - - - - M O D E L - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        core_layers_bilstm = [
            Input(shape=(None, 1)),
            Bidirectional(LSTM(neuron, activation=activation, kernel_regularizer=l2(kernel_regularizer))),
            Dropout(dropout_rate),
            Dense(time_step)
        ]

        core_layers_lstm = [
            Input(shape=(None, 1)),
            LSTM(neuron, activation=activation, kernel_regularizer=l2(kernel_regularizer)),
            Dropout(dropout_rate),
            Dense(time_step)
        ]

        if model_type == 'BILSTM':
            model = Sequential(core_layers_bilstm)
        elif model_type == 'LSTM':
            model = Sequential(core_layers_lstm)
        else:
            print(f"Unknown model_type: {model_type}")
            exit(0)
        model.compile(loss=loss_function, optimizer=optimizer)
        
        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=epoch, batch_size=batch_size, verbose=1, callbacks=[early_stopping])

        LSTMPlotter.plot_training_validation_loss(model_type, history.history['loss'], history.history['val_loss'], params)

        # - - - - - - - - - - - - I N F E R E N C E S   A N D   P R E D I C T I O N S - - - - - - - - - - - - - - - - - - 

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        
        test_loss = model.evaluate(X_test, y_test.reshape(y_test.shape[0], y_test.shape[1])) # TODO research default
        print('Test Loss:', test_loss)


        train_date = df['date'].iloc[time_step : time_step+len(train_predict)]
        test_date = df['date'].iloc[len(train_predict) + 2*time_step + time_step : len(train_predict) + 2*time_step + time_step + len(test_predict)]
        
        LSTMPlotter.plot_close_and_predictions(model_type, df, close_data, train_date, train_predict, test_date, test_predict, params)

        # - - - - - - - - - - - - - - - - - - - - F U T U R E   P R E D I C T I O N S - - - - - - - - - - - - - - - - - -
        last_data = test_data_scaled[-time_step:]
        last_data = last_data.reshape(1, time_step, 1)
        # TODO - Why does it have to be [[[]]]
            
        predictions = model.predict(last_data)
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)

        

        last_date = df['date'].iloc[-1]
        freq = determine_frequency(last_date, interval)

        future_dates = pd.date_range(start=last_date, periods=time_step+1, freq=freq)[1:]
                
        
        LSTMPlotter.plot_future_predictions(model_type, future_dates, predictions, params)
        
        # Print progress
        end_time = time.time()
        
        times = LSTMModelHelper.calculate_times(start_time, end_time)
        table_name = f"{ticker_fullname}_{model_type}"

        params_dict = LSTMModelHelper.params_to_dict(params, parameter_list)

        saved_data = LSTMModelHelper.save_data(times, history, test_loss, close_data, train_predict, test_predict, predictions, params_dict, table_name)

        db = LSTMDatabase(DATABASE_NAME)
        db.save_data(saved_data['params_dict'], saved_data['training_data'], saved_data['table_name'])


        # Convert elapsed time to minutes and seconds
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes = int(elapsed_time_seconds // 60)
        elapsed_seconds = int(elapsed_time_seconds % 60)

        current_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{i+1}/{total_combinations} finished in {elapsed_minutes}m {elapsed_seconds}s. Current time: {current_time}')
        
        # Append progress information to file
        append_to_times_and_epochs(i, total_combinations, elapsed_minutes, elapsed_seconds, current_time, saved_data, model_type)
