import datetime as dt
import os
import pandas as pd
import numpy as np
import itertools
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dropout, Dense, Input
from keras.callbacks import EarlyStopping, TensorBoard
from keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import json
import time
from keras.src.callbacks import History

from functions import determine_frequency, append_to_times_and_epochs
from lstm_model_helper import LSTMPlotter
from lstm_database import LSTMDatabase

from typing import Dict, Any, Tuple
from datetime import datetime



class Model:
    def __init__(self, model_type, params_json):
        self.model_type = model_type
        self.params_json = params_json
        self.input_folder = 'input'
        self.output_folder = 'output_modified'
    

    @staticmethod
    def create_folder(folder_path: str):
        """
        Create a folder.

        Parameters:
        - folder_path (str): The path to the folder to be created.
        """
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                # print(f"Folder '{folder_path}' created successfully.")
            except Exception as e:
                print(f"An error occurred while creating folder '{folder_path}': {e}")
                exit(1)
        else:
            print(f"Folder '{folder_path}' already exists.")
    

    def create_initial_folders(self):
        """
        Create initial folders for output.

        Parameters:
        - ticker_name (str): Name of the ticker.
        """

        ticker_path = os.path.join(self.output_folder, self.ticker_name)
        self.fig_path = os.path.join(ticker_path, "fig")
        self.logs_path = os.path.join(ticker_path, "logs")
        self.model_path = os.path.join(ticker_path, "model")

        # TODO - Make this code idiotproof by using lists below.
        # fig_folders = ["training_validation_loss", "inferences_and_predictions", "future_predictions", "residual"]
        # logs_folders = ['my_logs']
        # model_folders = []

        folders = {
            ticker_path: {
                "fig": {
                    "training_validation_loss": None,
                    "inferences_and_predictions": None,
                    "future_predictions": None,
                    "residual": None
                },
                "logs": {
                    "my_logs": None
                },
                "model": None
            }
        }

        try:
            for folder, subfolders in folders.items():
                try:
                    Model.create_folder(folder)
                except Exception as e:
                    print(f"An error occurred while creating folder '{folder}': {e}")
                for subfolder, subsubfolders in subfolders.items():
                    subfolder_path = os.path.join(folder, subfolder)
                    try:
                        Model.create_folder(subfolder_path)
                    except Exception as e:
                        print(f"An error occurred while creating folder '{subfolder_path}': {e}")
                    if subsubfolders:
                        for subsubfolder in subsubfolders:
                            try:
                                Model.create_folder(os.path.join(subfolder_path, subsubfolder))
                            except Exception as e:
                                print(f"An error occurred while creating folder '{os.path.join(subfolder_path, subsubfolder)}': {e}")
            print("All initial folders created successfully")
        except Exception as e:
            print(f"An error occurred while creating initial folders: {e}")
            exit(1)
 

    @staticmethod
    def dict_to_tuple(dict: dict) -> tuple:
        """
        Convert a dictionary to a tuple.

        Parameters:
        - dict (dict): The dictionary to convert.

        Returns:
        - tuple: The tuple containing dictionary values.
        """
        return tuple(dict.values())


    def load_params(self):
        """
        Load parameters from a JSON file.

        This function loads parameters required for the model and input data from a JSON file.
        
        Returns:
        None
        
        Raises:
        FileNotFoundError: If the JSON file specified in self.params_json does not exist.
        KeyError: If the required keys are not found in the loaded JSON.
        """
        # Error validation for file existence
        if not os.path.exists(self.params_json):
            raise FileNotFoundError(f"JSON file '{self.params_json}' not found.")

        params = Model.load_json_data(self.params_json)

        # Type validation for params
        if not isinstance(params, dict):
            raise TypeError("Expected 'params' to be a dictionary.")

        # Type validation for keys
        required_keys = ['model_params', 'data', 'database_name']
        for key in required_keys:
            if key not in params:
                raise KeyError(f"Key '{key}' not found in the loaded JSON.")
        
        # Assigning values to class attributes
        self.model_params = params['model_params']
        self.input_params = params['data']
        self.database_name = os.path.join(self.output_folder, params['database_name'])
    

    def get_input_data(self) -> pd.DataFrame:
        """
        Get input data from a CSV file.

        This function reads input data from a CSV file corresponding to the ticker name.
        
        Returns:
        pd.DataFrame or None: The DataFrame containing input data, or None if an error occurred.
        
        Raises:
        FileNotFoundError: If the CSV file corresponding to the ticker name does not exist.
        TypeError: If the loaded data is not of DataFrame type.
        """
        # Getting ticker name
        self.ticker_name = self.get_ticker_name()
        
        # Constructing CSV path
        csv_path = os.path.join(self.input_folder, 'csv_files')
        ticker_csv_path = os.path.join(csv_path, f'{self.ticker_name}.csv')

        # Error validation for file existence
        if not os.path.exists(ticker_csv_path):
            raise FileNotFoundError(f"CSV file '{ticker_csv_path}' not found.")
        
        try:
            df = pd.read_csv(ticker_csv_path)
            
            # Converting date column to datetime to fix the x axis dates on the graphs. Store them as pd.datetime instead of str.
            df['date'] = pd.to_datetime(df['date'])

            # Extracting close data
            self.close_data = df[['close']]
            # print(f"close_data shape: {self.close_data.shape}")

            return df
        except Exception as e:
            print(f"An error occurred in get_input_data function: {e}")
            exit(1)


    def get_ticker_name(self) -> str:
        """
        Get the ticker name based on input parameters.

        This function constructs the ticker name using security, interval, and data_type from input_params.

        Returns:
        str: The constructed ticker name.
        """
        try:
            # Retrieving required parameters from input_params
            security = self.input_params['security']
            interval = self.input_params['interval']
            data_type = self.input_params['data_type']

            # Constructing the ticker name
            return f"{security}_{interval}_{data_type}"
        except KeyError as e:
            raise KeyError(f"KeyError: {e}. Required key not found in input_params.") from e
        except Exception as e:
            raise Exception(f"An error occurred while constructing the ticker name: {e}") from e

    
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
            exit(1)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file '{json_file}'. Please check the file format.")
            exit(1)
        except Exception as e:
            print(f"An error occurred in load_json_data function: {e}")
            exit(1)
    

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
            X, y = [], []
            # TODO - Evaluate which one is better choice
            # for i in range(len(dataset)-2*time_step) # I removed the -1.
            for i in range(len(dataset) - 2 * time_step - 1):
                X.append(dataset[i:(i + time_step), 0])
                y.append(dataset[(i + time_step):(i + 2 * time_step), 0])
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            # X.reshape(X.shape[0], time_step, X.shape[1]) # Alternate code, with same output according to GPT
            return X, y
        except Exception as e:
            print(f"An error occurred in generate_dataset: {e}")
            exit(1)
    
    
    def prepare_model(self, params: Dict[str, Any]) -> MinMaxScaler:
        """
        Prepare data and scaler for the model training.

        Parameters:
        - params (dict): Parameters for data preparation and model training.

        Returns:
        - sklearn.preprocessing.MinMaxScaler: Scaler used for data normalization.
        """
        assert isinstance(params, dict), "params must be a dictionary."
        assert 'train_rate' in params and 'time_step' in params, "train_rate and time_step are required parameters in params."

        train_data, test_data = train_test_split(self.close_data.values.reshape(-1, 1), train_size=params['train_rate'], shuffle=False)
        # # Original code, CHANGED, but they return the exact same results
        # train_size = int(len(self.close_data) * params['train_rate'])
        # train_data = np.array(self.close_data[:train_size])
        # test_data = np.array(self.close_data[train_size:])

        scaler = MinMaxScaler()

        self.train_data_scaled = scaler.fit_transform(train_data)
        self.test_data_scaled = scaler.transform(test_data)

        # Generate datasets
        self.X_train, self.y_train = self.generate_dataset(self.train_data_scaled, params['time_step'])
        self.X_test, self.y_test = self.generate_dataset(self.test_data_scaled, params['time_step'])

        return scaler
    

    def build_model(self, params: Dict[str, Any]) -> Sequential:
        """
        Build a neural network model based on the provided parameters.

        Parameters:
        - params (Dict[str, any]): Parameters for building the model.

        Returns:
        - keras.models.Sequential: Compiled neural network model.
        """
        assert isinstance(params, dict), "params must be a dictionary."
        assert 'dropout_rate' in params and 'neuron' in params and 'activation' in params \
            and 'kernel_regularizer' in params and 'loss_function' in params and 'optimizer' in params \
            and 'time_step' in params, "Required parameters are missing in params."

        core_layers = [
            Input(shape=(None, 1)),
            Dropout(params['dropout_rate'], name="dropout_1"),
            Dense(params['time_step'], name="dense_1")
        ]

        if self.model_type == 'BILSTM':
            core_layers.insert(1, Bidirectional(LSTM(params['neuron'], activation=params['activation'], 
                                                     kernel_regularizer=l2(params['kernel_regularizer'])), name="bilstm_1"))
        elif self.model_type == 'LSTM':
            core_layers.insert(1, LSTM(params['neuron'], activation=params['activation'], 
                                       kernel_regularizer=l2(params['kernel_regularizer'])), name="lstm_1")
        else:
            print(f"Unknown model_type: {self.model_type}")
            exit(0)
            
        model = Sequential(core_layers)
        model.compile(loss=params['loss_function'], optimizer=params['optimizer'])
        return model

    
    def train_model(self, model: Sequential, params: Dict[str, Any], run_name: str) -> History:
        """
        Train a neural network model.

        Parameters:
        - model (Sequential): The neural network model to be trained.
        - params (Dict[str, Any]): Parameters for training the model.
        - run_name (str): Name of the training run for logging purposes.

        Returns:
        - keras.src.callbacks.History: Training history containing loss and metric values.
        """
        assert isinstance(params, dict), "params must be a dictionary."
        assert 'patience' in params and 'epoch' in params and 'batch_size' in params, "Required parameters are missing in params."

        early_stopping = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)
        log_dir = os.path.join(self.logs_path, run_name)
        logger = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            write_steps_per_second=True,
            update_freq='epoch',
            profile_batch=f"0, {params['batch_size']}"
        )
        history = model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
                            epochs=params['epoch'], batch_size=params['batch_size'], verbose=1, callbacks=[early_stopping, logger])
        return history


    def evaluate_model(self, model: Sequential, scaler: MinMaxScaler) -> Dict[str, np.ndarray]:
        """
        Evaluate a trained model.

        Parameters:
        - model (keras.models.Sequential): The trained neural network model to be evaluated.
        - scaler (sklearn.preprocessing.MinMaxScaler): The scaler used for normalization during training.

        Returns:
        - Dict[str, np.ndarray]: A dictionary containing evaluation results, including train and test predictions and test loss.
        """
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
    
    
    def train_test_dates(self, df: pd.DataFrame, params: Dict[str, Any], model_evaluation: dict) -> Tuple[pd.Series, pd.Series]:
        """
        Extract train and test dates.

        Parameters:
        - df (pd.DataFrame): DataFrame containing date information.
        - params (dict): Parameters used for modeling.
        - model_evaluation (dict): Evaluation results of the trained model.

        Returns:
        - Tuple[pd.Series, pd.Series]: Train and test dates.
        """
        time_step = params['time_step']
        train_predict = model_evaluation['train_predict']
        test_predict = model_evaluation['test_predict']

        train_date = df['date'].iloc[time_step : time_step + len(train_predict)]
        # TODO - Take a good look at the line below, make sure that it's right and optimized.
        test_date = df['date'].iloc[len(train_predict) + 2*time_step + time_step : len(train_predict) + 2*time_step + time_step + len(test_predict)]

        return train_date, test_date


    def future_predicts(self, model: Sequential, params: Dict[str, Any], scaler: MinMaxScaler) -> np.ndarray:
        """
        Generate future predictions using the trained model.

        Parameters:
        - model (keras.models.Sequential): The trained neural network model.
        - params (dict): Parameters used for modeling.
        - scaler (sklearn.preprocessing.MinMaxScaler): The scaler used for normalization during training.

        Returns:
        - np.ndarray: Future predictions.
        """
        last_data = self.test_data_scaled[-params['time_step']:]
        last_data = last_data.reshape(1, params['time_step'], 1)

        predictions = model.predict(last_data)
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        return predictions


    def calculate_metrics(self, y_test: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.

        Parameters:
        - y_test: True target values.
        - predictions: Predicted values.

        Returns:
        - Dict[str, float]: Evaluation metrics including MAE, MSE, and R-squared.
        """
        # Calculate metrics
        mae_score = mean_absolute_error(y_test, predictions)
        mse_score = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return {
            'mae_score': mae_score,
            'mse_score': mse_score,
            'r2_score': r2
        }
    

    def append_to_times_and_epochs(self, folder_path: str, index: int, total_combinations: int, elapsed_minutes: int, elapsed_seconds: int, params_str: str, saved_data: Dict[str, Any]):
        """
        Append the progress information to 'times_and_epochs.txt'.

        Parameters:
        - folder_path (str): The path to the folder containing the log file.
        - index (int): Current iteration index.
        - total_combinations (int): Total number of combinations.
        - elapsed_minutes (int): Current iteration elapsed_minutes.
        - elapsed_seconds (int): Current iteration elapsed_seconds.
        - params_str (str): String representation of model parameters.
        - saved_data (Dict[str, Any]): Dictionary containing saved data.
        """
        # Assert statements to validate input parameters
        assert isinstance(folder_path, str), "folder_path must be a string."
        assert isinstance(index, int), "index must be an integer."
        assert isinstance(total_combinations, int), "total_combinations must be an integer."
        assert isinstance(elapsed_minutes, int), "elapsed_minutes must be an integer."
        assert isinstance(elapsed_seconds, int), "elapsed_seconds must be an integer."
        assert isinstance(params_str, str), "params_str must be a string."
        assert isinstance(saved_data, dict), "saved_data must be a dictionary."

        try:
            text_file = os.path.join(folder_path, f'{self.ticker_name}_logs.txt')
            epoch_used, epoch = saved_data['training_data']['epoch_used'], saved_data['params_dict']['epoch']
            current_time = LSTMPlotter.get_current_time()
            with open(text_file, 'a+') as file:
                file.write(f"[{self.model_type}] {index+1}/{total_combinations} finished in {elapsed_minutes}m {elapsed_seconds}s. Params: {params_str} Current time: {current_time} ({epoch_used}/{epoch} epoch)\n")
            print(f"Data appended to {text_file} successfully.")
        except Exception as e:
            print(f"An error occurred while appending data to {text_file}: {str(e)}")

    @staticmethod
    def calculate_times(start_time: float, end_time: float) -> dict:
        """
        Calculate various time-related values based on start and end timestamps.

        Parameters:
        - start_time (float): Start timestamp.
        - end_time (float): End timestamp.

        Returns:
        - dict: Dictionary containing calculated time values.
        """
        assert isinstance(start_time, float), "start_time must be an float."
        assert isinstance(end_time, float), "end_time must be an float."

        try:
            # Convert timestamps to datetime objects
            start_date = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
            end_date = datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print("An error occurred when converting timestamps to dates: ", e)
            return None

        try:
            # Calculate total time in seconds
            total_time = end_time - start_time
        except Exception as e:
            print("An error occurred when calculating total time: ", e)
            return None

        return {
            'start_timestamp': start_time,
            'end_timestamp': end_time,
            'total_time': total_time,
            'start_date': start_date,
            'end_date': end_date
        }
    

    def save_results(self, times: dict, history: History, model_evaluation: np.ndarray, evaluation_metrics: Dict[str, Any], predictions: np.ndarray, params: Dict[str, Any]) -> dict:
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

                'mae_score': evaluation_metrics['mae_score'],
                'mse_score': evaluation_metrics['mse_score'],
                'r2_score': evaluation_metrics['r2_score'],

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
        fig_folders = ["training_validation_loss", "inferences_and_predictions", "future_predictions", "residual"]
        logs_folders = ['my_logs']
        self.load_params()
        df = self.get_input_data()
        self.create_initial_folders()

        param_keys = list(self.model_params.keys())
        param_values = list(self.model_params.values())
        param_combinations = [dict(zip(param_keys, values)) for values in itertools.product(*param_values)]

        # param_combinations = list(itertools.product(*self.model_params.values()))
        total_combinations = len(param_combinations)
        print(f"Total combinations: {total_combinations} different combinations")

        for i, params in enumerate(param_combinations):
            params_str = str(Model.dict_to_tuple(params))
            RUN_NAME = f"run_{params_str}"

            start_time = time.time()

            scaler = self.prepare_model(params)

            plotter = LSTMPlotter()

            # - - - - - - - - - - - - - - - - - - - M O D E L - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

            model = self.build_model(params)
            history = self.train_model(model, params, RUN_NAME)

            tra_val_path = os.path.join(self.fig_path, fig_folders[0])
            plotter.plot_training_validation_loss(tra_val_path, self.model_type, i, params_str, history.history['loss'], history.history['val_loss'])

            #  - - - - - - - - - - - - I N F E R E N C E S   A N D   P R E D I C T I O N S - - - - - - - - - - - - - - - - - - 

            model_evaluation = self.evaluate_model(model, scaler)
            evaluation_metrics = self.calculate_metrics(self.y_test, model_evaluation['test_predict'])

            train_date, test_date = self.train_test_dates(df, params, model_evaluation)

            close_pred_path = os.path.join(self.fig_path, fig_folders[1])
            plotter.plot_close_and_predictions(close_pred_path, self.model_type, i, params_str, df, self.close_data, train_date, model_evaluation['train_predict'], test_date, model_evaluation['test_predict'])

             # - - - - - - - - - - - - - - - - - - - - F U T U R E   P R E D I C T I O N S - - - - - - - - - - - - - - - - - -

            predictions = self.future_predicts(model, params, scaler)

            last_date = df['date'].iloc[-1]
            freq = determine_frequency(last_date, self.input_params['interval'])

            future_dates = pd.date_range(start=last_date, periods=params['time_step']+1, freq=freq)[1:]
            future_pred_path = os.path.join(self.fig_path, fig_folders[2])
            plotter.plot_future_predictions(future_pred_path, self.model_type, i, params_str, future_dates, predictions)

            end_time = time.time()

            times = self.calculate_times(start_time, end_time)


            data = self.save_results(times, history, model_evaluation, evaluation_metrics, predictions, params)
            db = LSTMDatabase(self.database_name)
            db.save_data(data['params_dict'], data['training_data'], data['table_name'])

            # Convert elapsed time to minutes and seconds
            elapsed_time_seconds = end_time - start_time
            elapsed_minutes = int(elapsed_time_seconds // 60)
            elapsed_seconds = int(elapsed_time_seconds % 60)

            current_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'{i+1}/{total_combinations} finished in {elapsed_minutes}m {elapsed_seconds}s. Current time: {current_time}')
            
            # Append progress information to file
            # text_file_folder = os.path.join(self.output_folder, self.ticker_name)
            ticker_model_path = os.path.join(self.model_path, self.ticker_name)
            Model.create_folder(ticker_model_path)
            model_path = os.path.join(ticker_model_path, f"{self.model_type}_{params_str}.keras")
            model.save(model_path)
            # append_to_times_and_epochs(i, total_combinations, params_str, elapsed_minutes, elapsed_seconds, current_time, data, self.model_type, text_file_folder, self.ticker_name)
            text_folder_path = os.path.join(self.logs_path, logs_folders[0])
            self.append_to_times_and_epochs(text_folder_path, i, total_combinations, elapsed_minutes, elapsed_seconds, params_str, data)

