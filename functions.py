import itertools
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import json

# Define parameter lists
parameter_list = {
    'train_rate': [0.75],  # 0.7, 0.8
    'time_step': [6, 24, 72, 168],
    'neuron': [25, 50, 100],
    'dropout_rate': [0.2],  # 0.1, 0.3
    'optimizer': ['adam', 'rmsprop'],
    'patience': [7],  # 13
    'epoch': [50, 100, 150],
    'batch_size': [100],
    'activation': ['sigmoid'],  # sigmoid
    'kernel_regularizer': [0.01],
    'loss_function': ['mean_absolute_error', 'mean_squared_error'],  # 'huber_loss', 'logcosh'
}

# Generate all combinations of parameters
param_combinations = list(itertools.product(*parameter_list.values()))

def params_to_dict(param_combinations: List[tuple], parameter_list: dict) -> List[dict]:
    """
    Convert parameter combinations from tuples to dictionaries.

    Parameters:
    - param_combinations (List[tuple]): List of parameter combinations as tuples.
    - parameter_list (dict): Dictionary containing parameter names and their corresponding values.
    """
    assert isinstance(param_combinations, list), 'param_combinations must be a list.'
    assert isinstance(parameter_list, dict), 'parameter_list must be a dict.'

    try:
        return [{key: value for key, value in zip(parameter_list.keys(), param)} for param in param_combinations[0]]
    except Exception as e:
        print("An error occurred when converting params tuples to dicts: ", e)
        return None


def tuple_to_dict(single_tuple: tuple, parameter_list: dict) -> dict:
    """
    Convert a single tuple to a dictionary using parameter names as keys.

    Parameters:
    - single_tuple (tuple): The tuple to be converted.
    - parameter_list (dict): Dictionary containing parameter names and their corresponding values.

    Returns:
    - dict: The converted dictionary.
    """
    assert isinstance(single_tuple, tuple), 'single_tuple must be a tuple.'
    assert isinstance(parameter_list, dict), 'parameter_list must be a dict.'

    try:
        return {key: value for key, value in zip(parameter_list.keys(), single_tuple)}
    except Exception as e:
        print("An error occurred when converting tuple to dict: ", e)
        return None


def get_current_time() -> str:
    try:
        return datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    except Exception as e:
        print("An error occurred when calculating current time: ", e)
        return None


def create_folder(folder_name: str, target_path: str = '/Volumes/DATA/LSTM/') -> None:
    """
    Create a folder in the specified path.

    Parameters:
    - folder_name (str): Name of the folder to be created.
    - target_path (str): Path of the folder to be created.
    """
    assert isinstance(folder_name, str), 'folder_name must be a string.'
    assert isinstance(target_path, str), 'target_path must be a string.'

    try:
        # Combine the target path with the provided folder name
        folder_path = os.path.join(target_path, target_path)

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_name}' created successfully at: {folder_path}")
        else:
            print(f"Folder '{folder_name}' already exists at: {folder_path}")
    except Exception as e:
        print(f"An error occurred when creating '{folder_name}' folder in {target_path} path: ", e)



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
        X, y = create_dataset(dataset, time_step)
        X = reshape_data(X)
        return X, y
    except Exception as e:
        print(f"An error occurred in generate_dataset: {e}")
        return None, None

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


def training_validation_loss_plot(training_loss: List[float], validation_loss: List[float], params: Tuple) -> None:
    """
    Plot training and validation loss.

    Parameters:
    - training_loss (List[float]): List of training loss values.
    - validation_loss (List[float]): List of validation loss values.
    - params (tuple): tuple that contains model parameters.
    """
    assert isinstance(training_loss, list), 'training_loss must be a list.'
    assert isinstance(validation_loss, list), 'validation_loss must be a list.'
    assert isinstance(params, tuple), 'params must be a tuple.'
    assert len(training_loss) == len(validation_loss), 'Lengths of training_loss and validation_loss must be the same.'

    try:        
        plt.figure(figsize=(12, 6))
        plt.plot(len(training_loss), training_loss, 'r', label='Training loss')
        plt.plot(len(training_loss), validation_loss, 'b', label='Validation loss')
        plt.title(f'Training and validation loss for: {params} (date: {get_current_time()})')
        plt.legend(loc=0)
        save_image('tra_val_loss', params)
        plt.close()
    except Exception as e:
        print("An error occurred when creating training and validation plot: ", e)


def close_and_predictions_plot(df: pd.DataFrame, close_data: List[float], train_date: List[str], 
                               train_predict: List[float], test_date: List[str], test_predict: List[float], 
                               params: Tuple) -> None:
    """
    Plot close values and predictions.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing date and close data.
    - close_data (List[float]): Original close data.
    - train_date (List[str]): Dates for training data.
    - train_predict (List[float]): Predictions for training data.
    - test_date (List[str]): Dates for test data.
    - test_predict (List[float]): Predictions for test data.
    - params (tuple): Model parameters.
    """
    assert isinstance(df, pd.DataFrame), 'df must be a dictionary.'
    assert isinstance(close_data, list), 'close_data must be a list.'
    assert isinstance(train_date, list), 'train_date must be a list.'
    assert isinstance(train_predict, list), 'train_predict must be a list.'
    assert isinstance(test_date, list), 'test_date must be a list.'
    assert isinstance(test_predict, list), 'test_predict must be a list.'
    assert isinstance(params, str), 'params must be a string.'

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], close_data, label='Original Close')
        plt.plot(train_date, train_predict[:,-1], label='Training Predictions')
        plt.plot(test_date, test_predict[:,-1], label='Test Predictions')
        plt.xlabel('Time')
        plt.ylabel('Close Value')
        plt.title(f'Close Values vs. Predictions {params}')
        plt.legend()
        save_image('close_and_predictions', params)
        plt.close()
    except Exception as e:
        print("An error occurred when creating close and predictions plot: ", e)


def future_plot(future_dates: List[str], predictions: List[float], params: Tuple) -> None:
    """
    Plot future price predictions.

    Parameters:
    - future_dates (List[str]): Dates for future predictions.
    - predictions (List[float]): Future price predictions.
    - params (tuple): Model parameters.
    """
    assert isinstance(future_dates, list), "future_dates must be a list."
    assert isinstance(predictions, list), "predictions must be a list."
    assert isinstance(params, tuple), "params must be a tuple."

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(future_dates, predictions, label='Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Future Price Predictions {params} ({get_current_time()})')
        plt.legend()
        save_image('future_predictions', params)
        plt.close()
    except Exception as e:
        print("An error occurred when creating future predictions plot: ", e)


def save_image(folder_name: str, params: Tuple) -> None:
    """
    Save the current plot as an image.

    Parameters:
    - folder_name (str): Name of the folder where the image will be saved.
    - params (tuple): Tuple containing parameters used for the plot.
    """

    assert isinstance(folder_name, tuple), "folder_name must be str."
    assert isinstance(params, tuple), "params must be a tuple."

    try:
        create_folder(f'images/{folder_name}')
        plt.savefig(f'images/{folder_name}/{folder_name}_{params}.png')
    except Exception as e:
        print(f"An error occurred when saving the {folder_name} plot: {str(e)}")


def load_params_from_json(json_file: str) -> dict:
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
        
        if not isinstance(params, dict):
            raise TypeError("JSON file should contain a dictionary.")

        parameter_list = params.get('parameter_list')
        security = params.get('security')
        interval = params.get('interval')
        database_name = params.get('database_name')

        if not isinstance(parameter_list, dict):
            raise TypeError("'parameter_list' in JSON file should be a dictionary.")
        if not isinstance(security, str):
            raise TypeError("'security' in JSON file should be a string.")
        
        return parameter_list, security, interval, database_name

    except FileNotFoundError:
        print(f"File '{json_file}' not found.")
        return None, None, None, None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file '{json_file}'. Please check the file format.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None
    

def append_to_times_and_epochs(i: int, total_combinations: int, elapsed_minutes: int, elapsed_seconds: int, current_time: str, saved_data: dict) -> None:
    """
    Append the progress information to 'times_and_epochs.txt'.

    Parameters:
    - i (int): Current iteration.
    - total_combinations (int): Total number of combinations.
    - elapsed_minutes (int): Elapsed minutes.
    - elapsed_seconds (int): Elapsed seconds.
    - current_time (str): Current time.
    - saved_data (dict): Dictionary containing saved data.
    """
    try:
        with open('times_and_epochs.txt', 'a+') as file:
            file.write(f'{i+1}/{total_combinations} finished in {elapsed_minutes}m {elapsed_seconds}s. Current time: {current_time} ({saved_data["training_data"]["epoch_used"]}/{saved_data["params_dict"]["epoch"]} epoch)\n')
        print("Data appended to 'times_and_epochs.txt' successfully.")
    except Exception as e:
        print(f"An error occurred while appending data to 'times_and_epochs.txt': {str(e)}")



def determine_frequency(interval: str) -> str:
    """
    Determine the frequency of timestamps based on the interval string.

    Args:
        interval (str): Interval string in the format "{x}{y}", where "x" is an integer and "y" represents the interval type.

    Returns:
        str: Frequency of timestamps.
    """
    try:
        # Validate input data type
        if not isinstance(interval, str):
            raise TypeError("Interval must be a string.")
        if not interval:
            raise ValueError("Invalid interval format. It should not be empty.")

        # Parse the integer part
        x = ''
        for char in interval:
            if char.isdigit():
                x += char
            else:
                break
        
        # Parse the string part
        y = interval[len(x):]

        # Mapping of interval type to frequency string
        interval_mapping = {
            'mo': 'M',    # Monthly
            'w': 'W',     # Weekly
            'd': 'D',     # Calendar Daily  # Use `B` for business day
            'h': 'h',     # Hourly
            'm': 'min',   # Minutely
            's': 's',     # Secondly
            'ms': 'ms'    # Millisecondly
        }

        # Check if the interval type is valid
        if y in interval_mapping:
            assert int(x) >= 1, 'Integer part of interval should be at least 1.'
            if int(x) == 1:
                return f"{interval_mapping[y]}"
            return f"{int(x)}{interval_mapping[y]}"
        else:
            raise ValueError(f"Invalid interval type: {y}")

    except TypeError as te:
        print(f"TypeError: {te}")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print('An error occurred when determining frequency: ', e)