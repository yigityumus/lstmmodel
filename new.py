import itertools
from datetime import datetime
import os
import numpy as np
from keras.src.callbacks import History
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

# Define parameter lists
parameter_lists = {
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
param_combinations = list(itertools.product(*parameter_lists.values()))

def params_to_dict(param_combinations: List[tuple]) -> List[dict]:
    assert isinstance(param_combinations, list), 'param_combinations must be a list.'

    try:
        return [{key: value for key, value in zip(parameter_lists.keys(), param)} for param in param_combinations]
    except Exception as e:
        print("An error occurred when converting params tuples to dicts: ", e)
        return None

def get_current_time() -> str:
    try:
        return datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    except Exception as e:
        print("An error occurred when calculating current time: ", e)
        return None


def create_folder(folder_name: str) -> None:
    assert isinstance(folder_name, str), 'folder_name must be a string.'

    try:
        # Get the current directory
        current_directory = os.getcwd()
        
        # Combine the current directory path with the provided folder name
        folder_path = os.path.join(current_directory, folder_name)
        
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_name}' created successfully at: {folder_path}")
        else:
            print(f"Folder '{folder_name}' already exists at: {folder_path}")
    except Exception as e:
        print(f"An error occurred when creating '{folder_name}' folder: ", e)



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


def training_validation_plot(training_loss: List[float], validation_loss: List[float], params: Tuple) -> None:
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
        epochs = range(len(training_loss))
        
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, training_loss, 'r', label='Training loss')
        plt.plot(epochs, validation_loss, 'b', label='Validation loss')
        plt.title(f'Training and validation loss for: {params} (date: {get_current_time()})')
        plt.legend(loc=0)

        folder_name = 'images/training_val'
        create_folder(folder_name)

        plt.savefig(f'{folder_name}/tra_val_loss_{params}.png') 
        plt.close()
    except Exception as e:
        print("An error occurred when creating training and validation plot: ", e)


def close_and_predictions_plot(df: pd.DataFrame, close_data: List[float], train_date: List[str], 
                               train_predict: List[float], test_date: List[str], test_predict: List[float], 
                               params: Tuple) -> None:
    """
    Plot close values and predictions.

    Parameters:
    - df (dict): DataFrame containing date and close data.
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

        folder_name = 'images/close_and_predictions'
        create_folder(folder_name)

        plt.savefig(f'{folder_name}/close_and_predictions_{params}.png')
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
        
        folder_name = 'images/future_predictions'
        create_folder(folder_name)

        plt.savefig(f'{folder_name}/future_predictions_{params}.png')
        plt.close()
    except Exception as e:
        print("An error occurred when creating future predictions plot: ", e)
