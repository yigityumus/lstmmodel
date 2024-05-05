import datetime as dt
import os
import pandas as pd
import numpy as np
import itertools
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dropout, Dense, Input
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import json
import time
from keras.src.callbacks import History

from functions import determine_frequency, append_to_times_and_epochs
from lstm_model_helper import LSTMModelHelper
from lstm_database import LSTMDatabase



class Model:
    def __init__(self, model_type, params_json):
        self.model_type = model_type
        self.params_json = params_json
        self.output_folder = 'output'
 

    def load_params(self):
        params = Model.load_json_data(self.params_json)

        self.model_params = params['model_params']
        self.input_params = params['data']
        self.database_name = os.path.join(self.output_folder, params['database_name'])
    

    def get_input_data(self):
        self.ticker_name = self.get_ticker_name()
        csv_path = os.path.join('csv_files', f'{self.ticker_name}.csv')

        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date']) # To fix the x axis dates on the graphs. Store them as pd.datetime instead of str.

        self.close_data = df[['close']]
        return df
        # print(f"close_data shape: {self.close_data.shape}")
    
    def get_ticker_name(self):
        return f"{self.input_params['security']}_{self.input_params['interval']}_{self.input_params['data_type']}"

    
    @staticmethod
    def load_json_data(json_file: str) -> dict:
        """
        Load parameters from a JSON file.

        Parameters:
        - json_file (str): Path to the JSON file containing parameters.

        Returns:
        - dict: Dictionary containing parameter configurations.
        """
        try:
            with open(json_file, 'r') as file:
                params = json.load(file)

            return params

        except FileNotFoundError:
            print(f"File '{json_file}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file '{json_file}'. Please check the file format.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    @staticmethod
    def reshape_data(X: np.ndarray) -> np.ndarray:
        """
        Reshape the input data array to have the correct dimensions for LSTM input.

        Parameters:
        - X (numpy.ndarray): Input data array.

        Returns:
        - numpy.ndarray: Reshaped data array.
        """
        assert isinstance(X, np.ndarray), "Input data must be a numpy array."
        assert len(X.shape) >= 2, 'Input data must be at least 2-dimensional.'

        try:
            # Reshape the data array
            return X.reshape(X.shape[0], X.shape[1], 1)
        except Exception as e:
            print("An error occurred in reshape_data: ", e)
            return None
        

    @staticmethod
    def create_dataset(dataset: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Create input-output pairs from the dataset.

        Parameters:
        - dataset (numpy.ndarray): Input dataset.
        - time_step (int): Number of time steps to use for each input sequence.

        Returns:
        - numpy.ndarray, numpy.ndarray: Input-output pairs.
        """
        assert isinstance(dataset, np.ndarray), "Input dataset must be a numpy array."
        assert isinstance(time_step, int) and time_step > 0, "time_step must be a positive integer."

        try:
            dataX, dataY = [], []
            for i in range(len(dataset)-2*time_step-1):
                a = dataset[i:(i+time_step), 0]
                dataX.append(a)
                b = dataset[(i+time_step):(i+2*time_step), 0]
                dataY.append(b)
            return np.array(dataX), np.array(dataY)
        except Exception as e:
            print("An error occurred in create_dataset: ", e)
            return None, None
    

    @staticmethod
    def generate_dataset(dataset: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate input-output pairs for LSTM training from the dataset.

        Parameters:
        - dataset (numpy.ndarray): Input dataset.
        - time_step (int): Number of time steps to use for each input sequence.

        Returns:
        - numpy.ndarray, numpy.ndarray: Input-output pairs for LSTM training.
        """
        assert isinstance(dataset, np.ndarray), "Input dataset must be a numpy array."
        assert isinstance(time_step, int) and time_step > 0, "time_step must be a positive integer."

        try:
            X, y = Model.create_dataset(dataset, time_step)
            X = Model.reshape_data(X)
            return X, y
        except Exception as e:
            print(f"An error occurred in generate_dataset: {e}")
            return None, None
    

    def prepare_model(self, params):
        train_size = int(len(self.close_data) * params['train_rate'])

        train_data = np.array(self.close_data[:train_size])
        test_data = np.array(self.close_data[train_size:])

        train_data = np.array(train_data)
        test_data = np.array(test_data)

        scaler = MinMaxScaler(feature_range=(0, 1))

        self.train_data_scaled = scaler.fit_transform(train_data)
        self.test_data_scaled = scaler.transform(test_data)

        # Generate datasets
        self.X_train, self.y_train = self.generate_dataset(self.train_data_scaled, params['time_step'])
        self.X_test, self.y_test = self.generate_dataset(self.test_data_scaled, params['time_step'])

        return scaler
    

    def build_model(self, params: dict):
        core_layers_bilstm = [
            Input(shape=(None, 1)),
            Bidirectional(LSTM(params['neuron'], activation=params['activation'], 
                            kernel_regularizer=l2(params['kernel_regularizer']))),
            Dropout(params['dropout_rate']),
            Dense(params['time_step'])
        ]

        core_layers_lstm = [
            Input(shape=(None, 1)),
            LSTM(params['neuron'], activation=params['activation'], 
                kernel_regularizer=l2(params['kernel_regularizer'])),
            Dropout(params['dropout_rate']),
            Dense(params['time_step'])
        ]

        if self.model_type == 'BILSTM':
            model = Sequential(core_layers_bilstm)
        elif self.model_type == 'LSTM':
            model = Sequential(core_layers_lstm)
        else:
            print(f"Unknown model_type: {self.model_type}")
            exit(0)
        model.compile(loss=params['loss_function'], optimizer=params['optimizer'])
        return model


    def train_model(self, model, params):
        early_stopping = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)
        history = model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                            epochs=params['epoch'], batch_size=params['batch_size'], verbose=1, callbacks=[early_stopping])
        return history


    def evaluate_model(self, model, scaler):
        train_predict = model.predict(self.X_train)
        test_predict = model.predict(self.X_test)
        
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        
        test_loss = model.evaluate(self.X_test, self.y_test.reshape(self.y_test.shape[0], self.y_test.shape[1]))

        return {
            'train_predict': train_predict,
            'test_predict': test_predict,
            'test_loss': test_loss
        }
    
    def calculate_dates(self, df, params, model_evaluation):
        time_step = params['time_step']
        train_predict = model_evaluation['train_predict']
        test_predict = model_evaluation['test_predict']

        train_date = df['date'].iloc[time_step : time_step+len(train_predict)]
        test_date = df['date'].iloc[len(train_predict) + 2*time_step + time_step : len(train_predict) + 2*time_step + time_step + len(test_predict)]


    def predict(self, model, params, scaler):
        last_data = self.test_data_scaled[-params['time_step']:]
        last_data = last_data.reshape(1, params['time_step'], 1)

        predictions = model.predict(last_data)
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        return predictions
    

    def save_results(self, times: dict, history: History, model_evaluation: np.ndarray, predictions: np.ndarray, params: dict) -> dict:
        """
        Saves model data to the database.

        Parameters:
        - Various parameters related to model configuration and training.
        """

        data = {
            'params_dict': params,
            'training_data': {
                'start_timestamp': times['start_timestamp'],
                'end_timestamp': times['end_timestamp'],
                'total_time': times['total_time'],
                'start_date': times['start_date'],
                'end_date': times['end_date'],
                'epoch_used': len(history.history['loss']),
                'test_loss': model_evaluation['test_loss'],

                'training_loss': json.dumps(history.history['loss']),
                'validation_loss': json.dumps(history.history['val_loss']),
                'close_data': json.dumps([item for sublist in self.close_data.values.tolist() for item in sublist]),
                'train_predict': json.dumps(model_evaluation['train_predict'].tolist()),
                'test_predict': json.dumps(model_evaluation['test_predict'].tolist()),
                'predictions': json.dumps([item for sublist in predictions.tolist() for item in sublist])
            },
            'table_name': f'{self.ticker_name}_{self.model_type}'
        }

        return data


    def run(self):
        self.load_params()
        df = self.get_input_data()

        param_keys = list(self.model_params.keys())
        param_values = list(self.model_params.values())
        param_combinations = [dict(zip(param_keys, values)) for values in itertools.product(*param_values)]

        # param_combinations = list(itertools.product(*self.model_params.values()))
        total_combinations = len(param_combinations)
        print(f"Total combinations: {total_combinations} different combinations")

        for i, params in enumerate(param_combinations):
            start_time = time.time()

            scaler = self.prepare_model(params)

            modelhelper = LSTMModelHelper(self.ticker_name)

            # - - - - - - - - - - - - - - - - - - - M O D E L - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

            model = self.build_model(params)
            history = self.train_model(model, params)

            modelhelper.plot_training_validation_loss(i, self.model_type, history.history['loss'], history.history['val_loss'], params)

            #  - - - - - - - - - - - - I N F E R E N C E S   A N D   P R E D I C T I O N S - - - - - - - - - - - - - - - - - - 

            model_evaluation = self.evaluate_model(model, scaler)

            time_step = params['time_step']
            train_predict = model_evaluation['train_predict']
            test_predict = model_evaluation['test_predict']
            train_date = df['date'].iloc[time_step : time_step+len(train_predict)]
            test_date = df['date'].iloc[len(train_predict) + 2*time_step + time_step : len(train_predict) + 2*time_step + time_step + len(test_predict)]

            modelhelper.plot_close_and_predictions(i, self.model_type, df, self.close_data, train_date, train_predict, test_date, test_predict, params)

             # - - - - - - - - - - - - - - - - - - - - F U T U R E   P R E D I C T I O N S - - - - - - - - - - - - - - - - - -

            predictions = self.predict(model, params, scaler)

            last_date = df['date'].iloc[-1]
            freq = determine_frequency(last_date, self.input_params['interval'])

            future_dates = pd.date_range(start=last_date, periods=time_step+1, freq=freq)[1:]
            modelhelper.plot_future_predictions(i, self.model_type, future_dates, predictions, params)

            end_time = time.time()

            times = LSTMModelHelper.calculate_times(start_time, end_time)


            data = self.save_results(times, history, model_evaluation, predictions, params)
            db = LSTMDatabase(self.database_name)
            db.save_data(data['params_dict'], data['training_data'], data['table_name'])

            # Convert elapsed time to minutes and seconds
            elapsed_time_seconds = end_time - start_time
            elapsed_minutes = int(elapsed_time_seconds // 60)
            elapsed_seconds = int(elapsed_time_seconds % 60)

            current_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'{i+1}/{total_combinations} finished in {elapsed_minutes}m {elapsed_seconds}s. Current time: {current_time}')
            
            # Append progress information to file
            text_file_folder = os.path.join(self.output_folder, self.ticker_name)
            params_tuple = LSTMModelHelper.dict_to_tuple(params)
            append_to_times_and_epochs(i, total_combinations, params_tuple, elapsed_minutes, elapsed_seconds, current_time, data, self.model_type, text_file_folder, self.ticker_name)

